import re

from .file_extractor import EPUB_CHAPTER_SEP

# ~15K input tokens — safe budget per Gemini call
MAX_CHUNK_CHARS = 60_000

# Structural boundary patterns tried in order of preference.
# Each uses a lookahead so the marker stays attached to the chunk it belongs to.
# Listed roughly from most specific to most generic.
_STRUCTURAL_PATTERNS = [
    # EPUB natural chapter boundaries (sentinel inserted by file_extractor)
    re.escape(EPUB_CHAPTER_SEP.strip()),

    # Plays: ACT I · ACT 1 · Act I (any case, optional period)
    r'(?mi)(?=^ACT\s+[IVXivx\d]+\.?\s*$)',

    # Novels: CHAPTER 1 · Chapter One · chapter the first (any case, optional period)
    r'(?mi)(?=^CHAPTER\s+[\w]+)',

    # Parts / Books / Volumes (any case)
    r'(?mi)(?=^(?:PART|BOOK|VOLUME)\s+[\w]+)',

    # Scenes within a play (fallback if ACT split not enough)
    r'(?mi)(?=^SCENE\s+[IVXivx\d]+)',

    # Standalone roman numerals as chapter headers: "IV." or "IV" on its own line
    r'(?m)(?=^[IVXivx]{1,6}\.?\s*$)',

    # Standalone arabic numerals as chapter headers: "12." or "12" on its own line
    r'(?m)(?=^\d{1,3}\.?\s*$)',

    # Markdown headings: # Title or ## Title
    r'(?m)(?=^#{1,3}\s+\S)',

    # Explicit scene/section break lines: *** or * * * or ---
    r'(?m)(?=^\s*(?:\*\s*){3,})',
    r'(?m)(?=^\s*-{3,}\s*$)',
]


def _size_split(text: str) -> list[str]:
    """Split a too-large block at paragraph boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        for sep in ('\n\n', '\n'):
            boundary = text.rfind(sep, start, end)
            if boundary != -1:
                break
        else:
            boundary = end
        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)
        start = boundary
    return [c for c in chunks if c]


def _ensure_size(chunks: list[str]) -> list[str]:
    """Further split any chunk that exceeds the size limit."""
    result = []
    for chunk in chunks:
        if len(chunk) <= MAX_CHUNK_CHARS:
            result.append(chunk)
        else:
            result.extend(_size_split(chunk))
    return result


def split_into_chunks(text: str) -> list[str]:
    """
    Split text into chunks that fit within a single Gemini call.

    Order of preference:
    1. EPUB natural chapter boundaries (sentinel from file_extractor).
    2. Structural text markers: acts, chapters, parts, scenes, headings, etc.
    3. Pure size-based splitting at paragraph boundaries (catch-all).

    Any chunk that is still too large after structural splitting is further
    split by size so no single chunk ever exceeds MAX_CHUNK_CHARS.
    """
    for pattern in _STRUCTURAL_PATTERNS:
        parts = [p.strip() for p in re.split(pattern, text) if p.strip()]
        if len(parts) >= 2:
            return _ensure_size(parts)

    # No structural markers — use size-based split
    if len(text) <= MAX_CHUNK_CHARS:
        return [text.strip()]
    return _size_split(text)


def merge_results(results: list) -> dict:
    """Merge multiple ExtractionResult dicts, deduplicating characters by name."""
    seen_characters: dict[str, dict] = {}
    all_segments: list[dict] = []

    for result in results:
        for char in result["characters"]:
            key = char["name"].lower()
            if key not in seen_characters:
                seen_characters[key] = char
        all_segments.extend(result["segments"])

    return {
        "characters": list(seen_characters.values()),
        "segments": all_segments,
    }
