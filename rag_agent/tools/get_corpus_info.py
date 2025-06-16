"""
Tool for retrieving detailed information about a specific RAG corpus.
"""

from pathlib import Path

from google.adk.tools.tool_context import ToolContext

from .local_utils import corpus_exists, get_corpus_metadata


def get_corpus_info(
    corpus_name: str,
    tool_context: ToolContext,
) -> dict:
    """
    Get detailed information about a specific RAG corpus, including its files.

    Args:
        corpus_name (str): The name of the corpus to get information about
        tool_context (ToolContext): The tool context

    Returns:
        dict: Information about the corpus and its files
    """
    try:
        # Check if corpus exists
        if not corpus_exists(corpus_name):
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist",
                "corpus_name": corpus_name,
            }

        # Get corpus metadata
        metadata = get_corpus_metadata(corpus_name)
        if not metadata:
            return {
                "status": "error",
                "message": f"Could not retrieve metadata for corpus '{corpus_name}'",
                "corpus_name": corpus_name,
            }

        # Process file information
        file_details = []
        for file_path, stored_path in metadata.get("files", {}).items():
            stored_file = Path(stored_path)
            if stored_file.exists():
                file_info = {
                    "file_id": stored_file.name,
                    "display_name": stored_file.name,
                    "source_uri": str(stored_file),
                    "create_time": str(Path.ctime(stored_file)),
                    "update_time": str(Path.ctime(stored_file)),
                }
                file_details.append(file_info)

        # Basic corpus info
        return {
            "status": "success",
            "message": f"Successfully retrieved information for corpus '{corpus_name}'",
            "corpus_name": corpus_name,
            "create_time": metadata.get("create_time", ""),
            "update_time": metadata.get("update_time", ""),
            "file_count": len(file_details),
            "chunk_count": len(metadata.get("chunks", [])),
            "files": file_details,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting corpus information: {str(e)}",
            "corpus_name": corpus_name,
        }
