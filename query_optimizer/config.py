"""
Configuration settings for the Query Optimizer Agent.

These settings are used by the various query optimization tools.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory for storing query optimizer data
BASE_DIR = Path(os.path.expanduser("~/.query_optimizer"))
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
BASE_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# NLP Model settings
DEFAULT_NLP_MODEL = "en_core_web_lg"  # Using spaCy's large English model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Using sentence-transformers model

# Query Analysis settings
MAX_QUERY_LENGTH = 1000
MIN_QUERY_LENGTH = 3
MAX_SUB_QUERIES = 5

# Query Enhancement settings
MAX_SYNONYMS_PER_TERM = 5
MAX_RELATED_TERMS = 10
SIMILARITY_THRESHOLD = 0.7

# Cache settings
CACHE_EXPIRY = 3600  # 1 hour in seconds
MAX_CACHE_SIZE = 1000  # Maximum number of cached queries

# Performance settings
BATCH_SIZE = 32
MAX_CONCURRENT_REQUESTS = 10 