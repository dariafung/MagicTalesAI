from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class Emotion(str, Enum):
    happy = "happy"
    sad = "sad"
    excited = "excited"
    scared = "scared"
    angry = "angry"
    calm = "calm"
    mysterious = "mysterious"
    curious = "curious"
    neutral = "neutral"


class CharacterRole(str, Enum):
    protagonist = "protagonist"
    antagonist = "antagonist"
    supporting = "supporting"
    narrator = "narrator"


class Character(BaseModel):
    name: str = Field(description="Character's name as it appears in the story")
    description: str = Field(description="One sentence describing the character's personality and physical traits")
    role: CharacterRole
    gender: str = Field(default="unknown", description="The character's gender (e.g. 'male', 'female', 'neutral') used for voice matching")


class StorySegment(BaseModel):
    type: Literal["narration", "dialogue"]
    speaker: str = Field(description="Character name or 'Narrator'")
    text: str = Field(description="Verbatim text from the story")
    emotion: Emotion


class ExtractionResult(BaseModel):
    """Complete structured output from Gemini. Also used as response_schema."""
    characters: list[Character]
    segments: list[StorySegment]


class ExtractRequest(BaseModel):
    text: str = Field(min_length=10, description="Raw story text to analyze")
    filename: str = Field(default="untitled", description="Original filename for reference")


class ExtractResponse(BaseModel):
    story_id: str
    characters: list[Character]
    segments: list[StorySegment]


# ── Voice assignment ──────────────────────────────────────────────────────────

class GeminiVoice(str, Enum):
    Puck    = "Puck"    # young, playful, mischievous
    Charon  = "Charon"  # deep, authoritative, commanding
    Kore    = "Kore"    # warm, maternal, expressive
    Fenrir  = "Fenrir"  # gruff, dramatic, intense
    Aoede   = "Aoede"   # smooth, measured — default narrator voice
    Leda    = "Leda"    # light, cheerful, innocent
    Orus    = "Orus"    # neutral male, reliable
    Zephyr  = "Zephyr"  # bright, energetic, youthful


class CharacterVoiceAssignment(BaseModel):
    character_name: str
    voice_name: GeminiVoice
    speaking_style: str = Field(description="Short phrase e.g. 'slow and menacing'")
    rationale: str = Field(description="One sentence explaining why this voice fits")


class VoiceAssignmentResult(BaseModel):
    """Used as Gemini response_schema for voice casting."""
    assignments: list[CharacterVoiceAssignment]


# ── Story session ─────────────────────────────────────────────────────────────

class StoryMeta(BaseModel):
    story_id: str
    filename: str
    created_at: str  # ISO datetime string
    status: Literal["extracted", "voices_assigned", "audio_generated"]
