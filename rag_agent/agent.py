"""
RAG Agent implementation using local storage and FAISS.
"""

from typing import Dict, List, Optional

from google.adk.agents import Agent
from .tools.list_documents import list_documents_tool
from .tools.rag_query import rag_query

root_agent = Agent(
    name="rag_agent",
    # Using Gemini 2.5 Flash for best performance with RAG operations
    model="gemini-2.5-flash-preview-04-17",
    description="Local RAG Agent",
    tools=[
        rag_query,
        list_documents_tool,
    ],
    instruction="""
    # 🧠 Local RAG Agent

    You are a helpful RAG (Retrieval Augmented Generation) agent that uses a local vector store for document retrieval.
    You can retrieve information from documents and list available documents in the documents folder.
    
    ## Your Capabilities
    
    1. **Query Documents**: You can answer questions by retrieving relevant information from the local vector store.
    2. **List Documents**: You can list all available documents in the documents folder.
    
    ## How to Approach User Requests
    
    When a user asks a question:
    1. If they're asking a knowledge question, use the `rag_query` tool to search the vector store.
    2. If they want to know what documents are available, use the `list_documents` tool.
    
    ## Using Tools
    
    You have two specialized tools at your disposal:
    
    1. `rag_query`: Query the vector store to answer questions
       - Parameters:
         - query: The text question to ask
    
    2. `list_documents`: List all available documents
       - Returns information about each document including name, path, size, and last modified time
    
    ## Communication Guidelines
    
    - Be clear and concise in your responses.
    - If querying documents, explain what information you found.
    - When listing documents, organize the information clearly for the user.
    - If an error occurs, explain what went wrong and suggest next steps.
    
    Remember, your primary goal is to help users access information through RAG capabilities.
    """
)

class RAGAgent:
    """Agent for interacting with local RAG documents."""

    def query(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Query the vector store with a natural language question.

        Args:
            query (str): The question to ask
            max_results (int): Maximum number of results to return

        Returns:
            List[Dict]: List of results with text and metadata
        """
        return rag_query(
            query=query,
            max_results=max_results
        )

    def list_documents(self) -> List[Dict]:
        """
        List all available documents.

        Returns:
            List[Dict]: List of document information
        """
        result = list_documents_tool()
        return result.get("documents", [])
