"""
chunker.py
----------
Simple word-count based chunking with overlap.

Deliberately NOT semantic chunking - a fixed sliding window over words
is easy to explain in a viva ("we split each page into ~700-word
windows that overlap by ~100 words so we don't cut a sentence's
context in half at a chunk boundary").

Every chunk keeps its metadata (chunk_id, source filename, page
number) so the retriever and citation logic can always trace an
answer back to its origin.
"""

import config


def chunk_pages(pages: list[dict], start_id: int = 0) -> list[dict]:
    """
    Split each page's text into overlapping word-based chunks.

    `pages` is a list of {"source": ..., "page": ..., "text": ...}
    as produced by loader.load_document().

    Returns a list of:
        {"chunk_id": int, "text": str, "source": str, "page": int}
    """
    chunks = []
    chunk_id = start_id

    step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
    if step <= 0:
        # Guard against a misconfigured overlap >= chunk size.
        step = config.CHUNK_SIZE

    for page in pages:
        words = page["text"].split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = start + config.CHUNK_SIZE
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"],
                })
                chunk_id += 1

            if end >= len(words):
                break
            start += step

    return chunks
