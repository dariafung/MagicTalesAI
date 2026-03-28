import asyncio
import os

from dotenv import load_dotenv
from fastapi import HTTPException
from google import genai
from google.genai import types

from .chunker import merge_results, split_into_chunks
from .models import ExtractionResult
from .prompts import SYSTEM_INSTRUCTION, build_user_prompt

load_dotenv()

_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    response_mime_type="application/json",
    response_schema=ExtractionResult,
    temperature=0.1,
    max_output_tokens=65536,
)

MODEL = "gemini-2.5-flash"

async def _extract_chunk(text: str) -> dict:
    """
    Processes a single chunk of story text to extract characters and segments.
    Note: Timeouts are intentionally omitted to support large book processing 
    which may exceed default duration limits.
    """
    response = await _client.aio.models.generate_content(
        model=MODEL,
        contents=build_user_prompt(text),
        config=_CONFIG,
    )

    candidate = response.candidates[0] if response.candidates else None
    if candidate is None:
        raise HTTPException(status_code=502, detail="No response from the AI model.")

    finish_reason = candidate.finish_reason.name
    if finish_reason == "SAFETY":
        raise HTTPException(
            status_code=502,
            detail="Content was blocked by safety filters.",
        )
    if finish_reason == "MAX_TOKENS":
        raise HTTPException(
            status_code=422,
            detail="A section of the text is too long to process. Try breaking it into smaller parts.",
        )

    if not response.text:
        raise HTTPException(status_code=502, detail="Received an empty response from the AI model.")

    return ExtractionResult.model_validate_json(response.text).model_dump()


async def extract_characters_from_story(story_text: str) -> ExtractionResult:
    chunks = split_into_chunks(story_text)

    if len(chunks) == 1:
        return ExtractionResult.model_validate(await _extract_chunk(chunks[0]))

    results = []
    for chunk in chunks:
        results.append(await _extract_chunk(chunk))

    return ExtractionResult.model_validate(merge_results(results))
