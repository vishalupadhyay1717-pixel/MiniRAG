"""
embeddings.py
-------------
Turns text into vectors using sentence-transformers.

The model is loaded exactly once (lazy singleton) since loading it is
the slowest part of startup. All embeddings are L2-normalized so that
a FAISS inner-product index behaves exactly like cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load (once) and return the shared embedding model instance."""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(config.EMBEDDING_MODEL)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load embedding model '{config.EMBEDDING_MODEL}'. "
                f"Check your internet connection (first run downloads the model) "
                f"or your sentence-transformers installation. Details: {exc}"
            ) from exc
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return normalized embeddings (float32) for a list of texts."""
    if not texts:
        return np.empty((0, 384), dtype="float32")

    model = get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,   # unit-length vectors -> inner product = cosine similarity
    )
    return vectors.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Return a (1, dim) normalized embedding for a single query string."""
    return embed_texts([query])
