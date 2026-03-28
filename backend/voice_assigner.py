from collections import Counter

from fastapi import HTTPException
from google import genai
from google.genai import types

from .gemini_client import MODEL, _client
from .models import ExtractionResult, GeminiVoice, VoiceAssignmentResult

_VOICE_PROFILES = "\n".join([
    f"- {v.value}: {desc}" for v, desc in [
        (GeminiVoice.Puck,   "male — young, playful, mischievous (good for children or trickster characters)"),
        (GeminiVoice.Charon, "male — deep, authoritative, commanding (good for rulers, judges, villains)"),
        (GeminiVoice.Kore,   "female — warm, maternal, expressive (good for nurturing or emotional characters)"),
        (GeminiVoice.Fenrir, "male — gruff, dramatic, intense (good for fierce or conflicted characters)"),
        (GeminiVoice.Aoede,  "female — smooth, measured, unhurried (the ideal Narrator voice)"),
        (GeminiVoice.Leda,   "female — light, cheerful, innocent (good for young or optimistic characters)"),
        (GeminiVoice.Orus,   "male — neutral, reliable, clear (good for steady or secondary characters)"),
        (GeminiVoice.Zephyr, "female — bright, energetic, youthful (good for lively or romantic characters)"),
    ]
])

_SYSTEM_INSTRUCTION = f"""You are a casting director for an AI audiobook.
Your job is to assign one voice to each character based on their personality, gender, and role.

Available voices:
{_VOICE_PROFILES}

Rules:
1. Assign exactly one voice to each character in the list. Match the voice's gender to the character's gender whenever possible.
2. The character named "Narrator" MUST always be assigned the voice "Aoede".
3. MINIMIZE voice repetition. Work through the character list and keep a mental tally of
   which voices you have already used. Assign a fresh voice whenever one is available.
4. Only reuse a voice when you have exhausted all 8 options. When reuse is unavoidable,
   only pair characters who appear in completely separate scenes and never speak to each other.
5. Never give the same voice to two characters who interact directly in the story.
6. Base your decision on the character's description AND their emotional range in the story.
7. speaking_style must be a short phrase (3–6 words) like "slow and deliberate" or "quick and panicked".
8. rationale must be one sentence explaining why this voice fits this specific character."""


def _build_casting_prompt(extraction: ExtractionResult) -> str:
    """
    Constructs a detailed prompt for Gemini to assign voices.
    Includes character descriptions, roles, genders, and dominant emotions
    extracted from the story segments to help Gemini make an informed choice.
    """
    # Count most common emotions per speaker to give Gemini richer signal
    emotion_counts: dict[str, Counter] = {}
    for seg in extraction.segments:
        # Normalize speaker names to ensure consistent emotion counting
        speaker = seg.speaker.strip().lower()
        emotion_counts.setdefault(speaker, Counter())[seg.emotion] += 1

    lines = ["Cast the following characters:\n"]
    for char in extraction.characters:
        speaker = char.name.strip().lower()
        # Retrieve the top 3 most common emotions for this character
        top_emotions = [e for e, _ in emotion_counts.get(speaker, Counter()).most_common(3)]
        emotion_str = ", ".join(top_emotions) if top_emotions else "unknown"
        lines.append(
            f"- {char.name} | gender: {char.gender} | role: {char.role} | {char.description} | "
            f"dominant emotions: {emotion_str}"
        )
    return "\n".join(lines)


_CONFIG = types.GenerateContentConfig(
    system_instruction=_SYSTEM_INSTRUCTION,
    response_mime_type="application/json",
    response_schema=VoiceAssignmentResult,
    temperature=0.3,
)


def _deduplicate(result: VoiceAssignmentResult) -> VoiceAssignmentResult:
    """
    Post-process Gemini's assignments to enforce minimal repetition.
    
    Logic:
    1. Always lock the "Narrator" to the "Aoede" voice.
    2. Ensure each character gets a unique voice if possible from the pool of 8.
    3. If more than 8 characters exist, reuse voices only after the pool is exhausted.
    4. Sorting ensures the Narrator is processed first to reserve Aoede.
    """
    all_voices = [v.value for v in GeminiVoice]
    # Narrator is always Aoede — remove it from the available pool for others
    free = [v for v in all_voices if v != GeminiVoice.Aoede.value]

    # Sort: narrator first (priority 0), then everyone else (priority 1)
    assignments = sorted(
        result.assignments,
        key=lambda a: 0 if a.character_name.strip().lower() == "narrator" else 1
    )

    used: set[str] = set()
    fixed = []
    for a in assignments:
        # Handle the Narrator explicitly
        if a.character_name.strip().lower() == "narrator":
            fixed.append(a.model_copy(update={"voice_name": GeminiVoice.Aoede}))
            used.add(GeminiVoice.Aoede.value)
            continue

        # If Gemini's suggested voice hasn't been used yet, keep it
        if a.voice_name.value not in used:
            used.add(a.voice_name.value)
            fixed.append(a)
        else:
            # Otherwise, pick the next available unused voice from the pool
            next_voice = next((v for v in free if v not in used), None)
            if next_voice is None:
                # All voices in the pool have been used at least once.
                # Reset the 'used' tracker (keeping Aoede reserved) to start a second round of assignments.
                used = {GeminiVoice.Aoede.value}
                next_voice = next(v for v in free if v not in used)
            
            used.add(next_voice)
            fixed.append(a.model_copy(update={"voice_name": GeminiVoice(next_voice)}))

    return VoiceAssignmentResult(assignments=fixed)


async def assign_voices(extraction: ExtractionResult) -> VoiceAssignmentResult:
    """
    Orchestrates the voice assignment process:
    1. Builds a context-rich prompt.
    2. Calls Gemini for initial voice casting.
    3. Post-processes results to ensure unique voices and consistent Narrator assignment.
    """
    prompt = _build_casting_prompt(extraction)
    response = await _client.aio.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=_CONFIG,
    )

    candidate = response.candidates[0] if response.candidates else None
    if candidate is None or not response.text:
        raise HTTPException(status_code=502, detail="No response from the AI model.")

    result = VoiceAssignmentResult.model_validate_json(response.text)
    return _deduplicate(result)
