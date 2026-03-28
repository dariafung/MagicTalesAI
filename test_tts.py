import asyncio
import json
import traceback
from dotenv import load_dotenv
load_dotenv()

from backend.models import ExtractionResult, VoiceAssignmentResult
from backend.tts_generator import generate_segment_audio

async def main():
    with open("data/7955710c-e1f4-4e77-8a85-f5c3c42955f8/extraction.json") as f:
        ext = ExtractionResult.model_validate(json.load(f))
    with open("data/7955710c-e1f4-4e77-8a85-f5c3c42955f8/voices.json") as f:
        voices = VoiceAssignmentResult.model_validate(json.load(f))
        
    for i in range(16):
        try:
            audio = await generate_segment_audio(ext, voices, i)
            if not audio:
                print(f"Segment {i}: returned empty audio!!")
            else:
                print(f"Segment {i}: OK, length {len(audio)}")
        except Exception as e:
            print(f"Segment {i}: ERROR {e}")
            traceback.print_exc()

asyncio.run(main())
