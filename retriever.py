"""
retriever.py
------------
Owns the FAISS vector index: building it from chunks, saving/loading it
to/from disk, and running top-K semantic search for a question.

Chunk metadata (source filename, page number, chunk text) is stored
separately as JSON, in the same order as the vectors in the FAISS
index, so a FAISS row index can always be mapped back to a chunk.
"""

import json
import os

import faiss
import numpy as np

import config
import embeddings


def build_index(chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    """
    Embed every chunk and build a FAISS index for cosine-similarity search.

    Uses IndexFlatIP (exact inner-product search). Because embeddings
    are L2-normalized, inner product is mathematically equivalent to
    cosine similarity - this keeps the retrieval math simple and exact,
    which matters for a small college-scale document set.
    """
    if not chunks:
        raise ValueError("Cannot build an index from zero chunks.")

    texts = [c["text"] for c in chunks]
    vectors = embeddings.embed_texts(texts)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    return index, chunks


def save_index(index: faiss.Index, chunks: list[dict]) -> None:
    """Persist the FAISS index and chunk metadata to disk."""
    faiss.write_index(index, config.INDEX_FILE)
    with open(config.METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def load_index() -> tuple[faiss.Index, list[dict]]:
    """Load a previously saved FAISS index and its chunk metadata."""
    if not index_exists():
        raise FileNotFoundError("No saved index found. Please build the index first.")

    index = faiss.read_index(config.INDEX_FILE)
    with open(config.METADATA_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if index.ntotal != len(chunks):
        raise ValueError("Saved FAISS index and metadata are out of sync. Rebuild the index.")
    return index, chunks


def delete_index() -> None:
    """Remove the saved index so it cannot be mistaken for current documents."""
    for path in (config.INDEX_FILE, config.METADATA_FILE):
        if os.path.exists(path):
            os.remove(path)


def index_exists() -> bool:
    """Whether a built index is already saved on disk."""
    return os.path.exists(config.INDEX_FILE) and os.path.exists(config.METADATA_FILE)


def search(index: faiss.Index, chunks: list[dict], query: str, top_k: int | None = None) -> list[dict]:
    """
    Embed the query and return its top_k most similar chunks.

    Each returned chunk dict includes a "similarity" field (0-1 range
    in practice, since both vectors are normalized).
    """
    if top_k is None:
        top_k = config.TOP_K

    query_vector = embeddings.embed_query(query)
    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(chunks):
            continue
        chunk = dict(chunks[int(idx)])
        chunk["similarity"] = float(score)
        results.append(chunk)

    return results
