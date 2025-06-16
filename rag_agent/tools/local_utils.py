"""
Utility functions for local RAG implementation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..config import (
    CORPORA_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    EMBEDDINGS_DIR,
    EMBEDDING_DIMENSION,
)

logger = logging.getLogger(__name__)

# Initialize the embedding model
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """Get or initialize the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    return _embedding_model


def get_corpus_path(corpus_name: str) -> Path:
    """Get the path to a corpus directory."""
    return CORPORA_DIR / corpus_name


def get_embeddings_path(corpus_name: str) -> Path:
    """Get the path to a corpus's embeddings file."""
    return EMBEDDINGS_DIR / f"{corpus_name}.faiss"


def get_metadata_path(corpus_name: str) -> Path:
    """Get the path to a corpus's metadata file."""
    return CORPORA_DIR / corpus_name / "metadata.json"


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end > text_length:
            end = text_length
        chunks.append(text[start:end])
        start = end - chunk_overlap

    return chunks


def create_corpus_directory(corpus_name: str) -> bool:
    """Create a directory for a new corpus."""
    try:
        corpus_path = get_corpus_path(corpus_name)
        corpus_path.mkdir(exist_ok=True)
        
        # Initialize metadata
        metadata = {
            "name": corpus_name,
            "files": {},
            "chunks": [],
            "create_time": str(Path.ctime(corpus_path)),
            "update_time": str(Path.ctime(corpus_path))
        }
        
        with open(get_metadata_path(corpus_name), "w") as f:
            json.dump(metadata, f, indent=2)
            
        return True
    except Exception as e:
        logger.error(f"Error creating corpus directory: {str(e)}")
        return False


def corpus_exists(corpus_name: str) -> bool:
    """Check if a corpus exists."""
    return get_corpus_path(corpus_name).exists()


def get_corpus_metadata(corpus_name: str) -> Optional[Dict]:
    """Get metadata for a corpus."""
    metadata_path = get_metadata_path(corpus_name)
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading corpus metadata: {str(e)}")
        return None


def update_corpus_metadata(corpus_name: str, metadata: Dict) -> bool:
    """Update metadata for a corpus."""
    try:
        metadata_path = get_metadata_path(corpus_name)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error updating corpus metadata: {str(e)}")
        return False


def create_or_load_index(corpus_name: str) -> Tuple[faiss.Index, List[str]]:
    """Create a new FAISS index or load an existing one."""
    embeddings_path = get_embeddings_path(corpus_name)
    metadata = get_corpus_metadata(corpus_name)
    
    if embeddings_path.exists() and metadata:
        # Load existing index
        index = faiss.read_index(str(embeddings_path))
        chunks = metadata.get("chunks", [])
    else:
        # Create new index
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        chunks = []
    
    return index, chunks


def save_index(corpus_name: str, index: faiss.Index, chunks: List[str]) -> bool:
    """Save the FAISS index and update metadata."""
    try:
        # Save index
        embeddings_path = get_embeddings_path(corpus_name)
        faiss.write_index(index, str(embeddings_path))
        
        # Update metadata
        metadata = get_corpus_metadata(corpus_name)
        if metadata:
            metadata["chunks"] = chunks
            metadata["update_time"] = str(Path.ctime(get_corpus_path(corpus_name)))
            update_corpus_metadata(corpus_name, metadata)
        
        return True
    except Exception as e:
        logger.error(f"Error saving index: {str(e)}")
        return False


def compute_embeddings(texts: List[str]) -> np.ndarray:
    """Compute embeddings for a list of texts."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.astype('float32')


def search_similar_chunks(
    corpus_name: str,
    query: str,
    top_k: int = DEFAULT_TOP_K
) -> List[Dict]:
    """Search for similar chunks in a corpus."""
    try:
        # Load index and chunks
        index, chunks = create_or_load_index(corpus_name)
        if not chunks:
            return []
        
        # Compute query embedding
        query_embedding = compute_embeddings([query])[0]
        
        # Search
        distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(chunks):  # Ensure index is valid
                results.append({
                    "text": chunks[idx],
                    "score": float(1.0 / (1.0 + distance)),  # Convert distance to similarity score
                    "rank": i + 1
                })
        
        return results
    except Exception as e:
        logger.error(f"Error searching similar chunks: {str(e)}")
        return []


def list_documents() -> List[Dict]:
    """List all documents in the documents folder."""
    try:
        # Get the path to the rag_agent directory
        current_dir = Path(__file__).parent.parent
        documents_dir = current_dir / "documents"
        
        if not documents_dir.exists():
            logger.warning(f"Documents directory not found at {documents_dir}")
            return []
            
        documents = []
        for file_path in documents_dir.glob("*.pdf"):
            documents.append({
                "name": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "last_modified": file_path.stat().st_mtime
            })
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return [] 