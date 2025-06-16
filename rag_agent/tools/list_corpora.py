"""
Tool for listing all available local RAG corpora.
"""

from pathlib import Path
from typing import Dict, List, Union

from ..config import CORPORA_DIR
from .local_utils import get_corpus_metadata


def list_corpora() -> dict:
    """
    List all available local RAG corpora.

    Returns:
        dict: A list of available corpora and status, with each corpus containing:
            - name: The name of the corpus
            - create_time: When the corpus was created
            - update_time: When the corpus was last updated
            - file_count: Number of files in the corpus
    """
    try:
        # Get all corpus directories
        corpus_dirs = [d for d in CORPORA_DIR.iterdir() if d.is_dir()]
        
        # Process corpus information
        corpus_info: List[Dict[str, Union[str, int]]] = []
        for corpus_dir in corpus_dirs:
            corpus_name = corpus_dir.name
            metadata = get_corpus_metadata(corpus_name)
            
            if metadata:
                corpus_data: Dict[str, Union[str, int]] = {
                    "name": corpus_name,
                    "create_time": metadata.get("create_time", ""),
                    "update_time": metadata.get("update_time", ""),
                    "file_count": len(metadata.get("files", {})),
                }
                corpus_info.append(corpus_data)

        return {
            "status": "success",
            "message": f"Found {len(corpus_info)} available corpora",
            "corpora": corpus_info,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error listing corpora: {str(e)}",
            "corpora": [],
        }
