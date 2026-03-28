# MagicTales-AI

An AI-powered story-to-audio experience that turns written stories into rich, immersive audiobooks — with distinct character voices, emotion-aware narration, and generated background music.

## Vision

Input a story → get back a full audio experience:

- **Character Extraction** — identify every character, assign roles, tag emotions per line
- **Voice Generation** — unique voices per character via ElevenLabs; parent voice cloning so kids hear familiar voices
- **Streaming Narration** — Gemini Multimodal Live API for real-time narration with pause-and-answer for children's questions
- **Background Music** — scene-aware BGM generated with Google's Lyria model
- **Audiobook Output** — narration + dialogue + BGM combined into a playable experience

## Stack

- **Backend**: Python, FastAPI
- **Frontend**: Plain HTML/JS (no build step)
- **AI**: Google Gemini Flash (story structuring), ElevenLabs (TTS + voice cloning), Gemini Live API (interactive Q&A), Google Lyria (BGM)

## Pipeline

| Step | Status | What it does |
|------|--------|-------------|
| Upload book (PDF/EPUB) | ✅ | Extracts text, preserves chapter boundaries |
| Character extraction | ✅ | Gemini Flash → characters, segmented dialogue/narration, emotion tags |
| Story session storage | ✅ | JSON files in `data/{story_id}/`, persisted across requests |
| Voice assignment | ✅ | Gemini casts each character to a TTS voice preset + speaking style |
| Audio generation | 🔜 | Gemini TTS per segment |
| Parent voice cloning | 🔜 | ElevenLabs voice clone API |
| Interactive Q&A | 🔜 | Gemini Multimodal Live API (pause + answer children's questions) |
| BGM generation | 🔜 | Google Lyria based on scene/emotion |
| Audio mixing & playback | 🔜 | Combine narration + BGM into final audiobook |

## API

```
POST /api/upload-text                    # upload PDF/EPUB → returns raw text
POST /api/extract-characters             # text → story_id + characters + segments
POST /api/stories/{story_id}/assign-voices  # Gemini casts voices
GET  /api/stories/{story_id}             # load full session
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add GOOGLE_API_KEY

uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000` — the UI walks through the full pipeline.
