"""
Configuration settings for the RAG Agent.

These settings are used by the various RAG tools.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory for storing RAG data
BASE_DIR = Path(os.path.expanduser("~/.rag_agent"))
CORPORA_DIR = BASE_DIR / "corpora"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# Create directories if they don't exist
BASE_DIR.mkdir(exist_ok=True)
CORPORA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# RAG settings
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 3
DEFAULT_DISTANCE_THRESHOLD = 0.5

# Embedding model settings
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Using sentence-transformers model
EMBEDDING_DIMENSION = 384  # Dimension of the embedding vectors

# File processing settings
SUPPORTED_EXTENSIONS = {
    ".txt": "text",
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
}

# Vertex AI settings
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")

# RAG settings
DEFAULT_EMBEDDING_REQUESTS_PER_MIN = 1000
