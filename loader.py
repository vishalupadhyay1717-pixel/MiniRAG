"""
loader.py
---------
Document ingestion: extract raw text from PDF and TXT files while
preserving per-page metadata (filename + page number).

This module does NOT chunk or embed anything - it only turns a file on
disk into a list of {source, page, text} dictionaries, one entry per
PDF page (or a single entry for a whole TXT file).
"""

import os
import fitz  # PyMuPDF

SUPPORTED_EXTENSIONS = (".pdf", ".txt")


def load_document(filepath: str) -> list[dict]:
    """
    Extract text from a single PDF or TXT file.

    Returns a list of dicts:
        {"source": filename, "page": page_number, "text": page_text}

    Raises ValueError for unsupported types, unreadable files, or
    files that contain no extractable text.
    """
    if not os.path.exists(filepath):
        raise ValueError(f"File not found: {filepath}")

    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return _load_pdf(filepath, filename)
    elif ext == ".txt":
        return _load_txt(filepath, filename)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Only PDF and TXT are supported.")


def _load_pdf(filepath: str, filename: str) -> list[dict]:
    """Extract text page-by-page from a PDF using PyMuPDF."""
    pages = []
    try:
        doc = fitz.open(filepath)
    except Exception as exc:
        raise ValueError(f"Could not open '{filename}' as a PDF: {exc}") from exc

    try:
        for page_number in range(len(doc)):
            page = doc[page_number]
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "source": filename,
                    "page": page_number + 1,   # 1-indexed for human-friendly citations
                    "text": text,
                })
    finally:
        doc.close()

    if not pages:
        raise ValueError(f"No extractable text found in '{filename}'. It may be a scanned/empty PDF.")

    return pages


def _load_txt(filepath: str, filename: str) -> list[dict]:
    """Load a plain text file as a single logical 'page'."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"'{filename}' is empty.")

    return [{"source": filename, "page": 1, "text": text}]


def list_documents(documents_dir: str) -> list[str]:
    """Return a sorted list of supported document filenames in a directory."""
    if not os.path.isdir(documents_dir):
        return []
    return sorted(
        f for f in os.listdir(documents_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )
