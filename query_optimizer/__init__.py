"""
Query Optimizer Agent package.
"""

import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Create base directories
BASE_DIR = Path.home() / ".query_optimizer"
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"

BASE_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Import tools
from .tools import optimize_query, analyze_query, enhance_query

__all__ = [
    "optimize_query",
    "analyze_query",
    "enhance_query"
] 