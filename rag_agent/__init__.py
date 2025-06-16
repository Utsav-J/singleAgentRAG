"""
Local RAG Agent package.
"""

import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Create base directories
BASE_DIR = Path.home() / ".rag_agent"
CORPORA_DIR = BASE_DIR / "corpora"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

BASE_DIR.mkdir(exist_ok=True)
CORPORA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# Import tools
from .tools import rag_query, list_corpora, get_corpus_info

__all__ = [
    "rag_query",
    "list_corpora",
    "get_corpus_info"
]
