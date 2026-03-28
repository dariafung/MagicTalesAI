import io

from fastapi import HTTPException

# Sentinel used to mark EPUB chapter boundaries in the combined text,
# chosen to be unlikely to appear in real book content.
EPUB_CHAPTER_SEP = "\n\n<<<CHAPTER_BREAK>>>\n\n"


def extract_text_from_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise HTTPException(status_code=500, detail="pypdf not installed")

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")
    return text


def extract_text_from_epub(data: bytes) -> str:
    """
    Extract text from EPUB, preserving spine-item (chapter) boundaries
    via EPUB_CHAPTER_SEP so the chunker can split on them without needing
    to guess chapter title patterns.
    """
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise HTTPException(status_code=500, detail="ebooklib or beautifulsoup4 not installed")

    book = epub.read_epub(io.BytesIO(data))
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        part = soup.get_text(separator="\n").strip()
        if part:
            parts.append(part)

    if not parts:
        raise HTTPException(status_code=422, detail="Could not extract text from EPUB.")

    return EPUB_CHAPTER_SEP.join(parts)
