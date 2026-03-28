import asyncio
import logging
import os
from typing import Optional

from fastapi import HTTPException
from google.cloud import firestore, storage

from .models import ExtractionResult, StoryMeta, VoiceAssignmentResult

project_id = os.getenv("GCP_PROJECT_ID")
bucket_name = os.getenv("GCS_BUCKET_NAME")
database_id = os.getenv("GCP_DATABASE_ID", "(default)")

db = firestore.AsyncClient(project=project_id, database=database_id)
gcs = storage.Client(project=project_id) if bucket_name else None
bucket = gcs.bucket(bucket_name) if gcs else None


async def save_meta(story_id: str, meta: StoryMeta) -> None:
    doc_ref = db.collection("stories").document(story_id)
    await doc_ref.set({"meta": meta.model_dump()}, merge=True)


async def save_extraction(story_id: str, result: ExtractionResult) -> None:
    doc_ref = db.collection("stories").document(story_id)
    await doc_ref.set({"extraction": result.model_dump()}, merge=True)


async def save_voices(story_id: str, result: VoiceAssignmentResult) -> None:
    doc_ref = db.collection("stories").document(story_id)
    await doc_ref.set({"voices": result.model_dump()}, merge=True)


async def _get_doc_field(story_id: str, field: str, label: str) -> dict:
    doc = await db.collection("stories").document(story_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found.")
    data = doc.to_dict()
    if not data or field not in data:
        raise HTTPException(status_code=404, detail=f"{label} not found for story '{story_id}'.")
    return data[field]


async def load_meta(story_id: str) -> StoryMeta:
    data = await _get_doc_field(story_id, "meta", "Meta")
    return StoryMeta.model_validate(data)


async def load_extraction(story_id: str) -> ExtractionResult:
    data = await _get_doc_field(story_id, "extraction", "Extraction")
    return ExtractionResult.model_validate(data)


async def load_voices(story_id: str) -> VoiceAssignmentResult:
    data = await _get_doc_field(story_id, "voices", "Voice assignment")
    return VoiceAssignmentResult.model_validate(data)


def _get_blob(blob_name: str):
    if not bucket:
        logging.warning("GCS_BUCKET_NAME not set, cannot access storage")
        return None
    return bucket.blob(blob_name)


async def has_segment_audio(story_id: str, index: int) -> bool:
    blob = _get_blob(f"{story_id}/audio_{index}.wav")
    if not blob:
        return False
    return await asyncio.to_thread(blob.exists)


async def load_segment_audio(story_id: str, index: int) -> Optional[bytes]:
    blob = _get_blob(f"{story_id}/audio_{index}.wav")
    if blob and await asyncio.to_thread(blob.exists):
        return await asyncio.to_thread(blob.download_as_bytes)
    return None


async def save_segment_audio(story_id: str, index: int, wav_bytes: bytes) -> None:
    blob = _get_blob(f"{story_id}/audio_{index}.wav")
    if blob:
        await asyncio.to_thread(blob.upload_from_string, wav_bytes, content_type="audio/wav")


async def has_music(story_id: str, emotion: str) -> bool:
    blob = _get_blob(f"{story_id}/music_{emotion}.mp3")
    if not blob:
        return False
    return await asyncio.to_thread(blob.exists)


async def save_music(story_id: str, emotion: str, data: bytes) -> None:
    blob = _get_blob(f"{story_id}/music_{emotion}.mp3")
    if blob:
        await asyncio.to_thread(blob.upload_from_string, data, content_type="audio/mpeg")


async def load_music(story_id: str, emotion: str) -> Optional[bytes]:
    blob = _get_blob(f"{story_id}/music_{emotion}.mp3")
    if blob and await asyncio.to_thread(blob.exists):
        return await asyncio.to_thread(blob.download_as_bytes)
    return None


async def list_stories() -> list[StoryMeta]:
    stories = []
    docs = db.collection("stories").stream()
    async for doc in docs:
        data = doc.to_dict()
        if data and "meta" in data:
            try:
                stories.append(StoryMeta.model_validate(data["meta"]))
            except Exception:
                pass
    stories.sort(key=lambda s: s.created_at, reverse=True)
    return stories


async def load_full_session(story_id: str) -> dict:
    doc = await db.collection("stories").document(story_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Story not found")
    data = doc.to_dict()

    meta = data.get("meta")
    extraction = data.get("extraction")
    voices = data.get("voices")

    has_audio = await has_segment_audio(story_id, 0)

    return {
        **meta,
        "characters": extraction.get("characters", []) if extraction else [],
        "segments": extraction.get("segments", []) if extraction else [],
        "voice_assignments": voices.get("assignments", []) if voices else None,
        "has_audio": has_audio,
    }
