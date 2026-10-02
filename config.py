"""
config.py
---------
Central configuration for MiniRAG.

Every tunable value in the project lives here so a student can change
behaviour (chunk size, retrieval depth, threshold, model names, etc.)
from a single, obvious place instead of hunting through the codebase.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
INDEX_DIR = os.path.join(DATA_DIR, "index")

INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.json")

# Make sure the expected folders exist so a first run doesn't crash.
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 700       # approx. words per chunk
CHUNK_OVERLAP = 100    # approx. words of overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers model name

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 3                      # number of chunks retrieved per question
SIMILARITY_THRESHOLD = 0.35    # minimum cosine similarity to trust a chunk

# ---------------------------------------------------------------------------
# Local LLM (Ollama)
# ---------------------------------------------------------------------------
OLLAMA_MODEL = "llama3.2:3b"                       # change to any installed model
OLLAMA_URL = "http://localhost:11434/api/generate"  # default local Ollama endpoint
OLLAMA_TIMEOUT_SECONDS = 120
