import json
from pathlib import Path

from fastapi import HTTPException

from .models import ExtractionResult, StoryMeta, VoiceAssignmentResult

STORY_DIR = Path("data")


def story_path(story_id: str) -> Path:
    return STORY_DIR / story_id


def _read(path: Path, label: str):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{label} not found for story '{path.parent.name}'.")
    return json.loads(path.read_text())


def save_meta(story_id: str, meta: StoryMeta) -> None:
    p = story_path(story_id)
    p.mkdir(parents=True, exist_ok=True)
    (p / "meta.json").write_text(meta.model_dump_json(indent=2))


def save_extraction(story_id: str, result: ExtractionResult) -> None:
    (story_path(story_id) / "extraction.json").write_text(result.model_dump_json(indent=2))


def save_voices(story_id: str, result: VoiceAssignmentResult) -> None:
    (story_path(story_id) / "voices.json").write_text(result.model_dump_json(indent=2))


def load_meta(story_id: str) -> StoryMeta:
    return StoryMeta.model_validate(_read(story_path(story_id) / "meta.json", "Meta"))


def load_extraction(story_id: str) -> ExtractionResult:
    return ExtractionResult.model_validate(_read(story_path(story_id) / "extraction.json", "Extraction"))


def load_voices(story_id: str) -> VoiceAssignmentResult:
    return VoiceAssignmentResult.model_validate(_read(story_path(story_id) / "voices.json", "Voice assignment"))


def segment_audio_path(story_id: str, index: int) -> Path:
    return story_path(story_id) / f"audio_{index}.wav"


def save_segment_audio(story_id: str, index: int, wav_bytes: bytes) -> None:
    segment_audio_path(story_id, index).write_bytes(wav_bytes)


def list_stories() -> list[StoryMeta]:
    if not STORY_DIR.exists():
        return []
    stories = []
    for d in sorted(STORY_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        meta_file = d / "meta.json"
        if d.is_dir() and meta_file.exists():
            try:
                stories.append(StoryMeta.model_validate(json.loads(meta_file.read_text())))
            except Exception:
                pass
    return stories


def load_full_session(story_id: str) -> dict:
    meta = load_meta(story_id)
    extraction = load_extraction(story_id)
    voices_path = story_path(story_id) / "voices.json"
    voices = load_voices(story_id) if voices_path.exists() else None

    return {
        **meta.model_dump(),
        "characters": [c.model_dump() for c in extraction.characters],
        "segments": [s.model_dump() for s in extraction.segments],
        "voice_assignments": [a.model_dump() for a in voices.assignments] if voices else None,
        "has_audio": segment_audio_path(story_id, 0).exists(),
    }
