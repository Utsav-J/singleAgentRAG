"""
Tool for querying the local vector store.
"""

import pickle
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K
from .local_utils import get_embedding_model


def rag_query(query: str, max_results: int = DEFAULT_TOP_K) -> dict:
    """
    Query the local vector store with a natural language question.

    Args:
        query (str): The question to ask
        max_results (int): Maximum number of results to return

    Returns:
        dict: A dictionary containing:
            - status: Success or error status
            - message: Description of the result
            - results: List of results with text and metadata
    """
    try:
        # Load the FAISS index
        index = faiss.read_index("vector.faiss")
        
        # Load metadata
        with open("metadata.pkl", "rb") as f:
            chunks_with_metadata = pickle.load(f)
        
        # Get query embedding
        model = get_embedding_model()
        query_embedding = model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize for cosine similarity
        
        # Search
        distances, indices = index.search(query_embedding.reshape(1, -1), max_results)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(chunks_with_metadata):
                chunk = chunks_with_metadata[idx]
                results.append({
                    "text": chunk["content"],
                    "source_file": chunk["source_file"],
                    "page": chunk["page"],
                    "score": float(1.0 / (1.0 + distance)),  # Convert distance to similarity score
                    "rank": i + 1
                })
        
        return {
            "status": "success",
            "message": f"Found {len(results)} relevant results",
            "results": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error querying vector store: {str(e)}",
            "results": []
        }
