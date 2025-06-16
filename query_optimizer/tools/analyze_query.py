"""
Tool for analyzing natural language queries.
"""

import logging
from typing import Dict, List, Optional
import spacy
from sentence_transformers import SentenceTransformer

from ..config import (
    DEFAULT_NLP_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    MAX_QUERY_LENGTH,
    MIN_QUERY_LENGTH,
)

logger = logging.getLogger(__name__)

# Initialize NLP model
_nlp_model = None

def get_nlp_model():
    """Get or initialize the spaCy model."""
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = spacy.load(DEFAULT_NLP_MODEL)
    return _nlp_model

def analyze_query_tool(query: str) -> dict:
    """
    Analyze a natural language query to understand its structure and intent.

    Args:
        query (str): The query to analyze

    Returns:
        dict: A dictionary containing:
            - status: Success or error status
            - message: Description of the result
            - analysis: Dictionary containing:
                - entities: List of identified entities
                - intent: Detected query intent
                - key_terms: List of key terms
                - temporal_aspects: Temporal information
                - query_type: Type of query (factual, analytical, etc.)
                - complexity: Query complexity score
    """
    try:
        # Validate query
        if len(query) < MIN_QUERY_LENGTH:
            return {
                "status": "error",
                "message": f"Query too short (minimum {MIN_QUERY_LENGTH} characters)",
                "analysis": None
            }
        
        if len(query) > MAX_QUERY_LENGTH:
            return {
                "status": "error",
                "message": f"Query too long (maximum {MAX_QUERY_LENGTH} characters)",
                "analysis": None
            }

        # Get NLP model
        nlp = get_nlp_model()
        
        # Process query
        doc = nlp(query)
        
        # Extract entities
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        
        # Extract key terms (nouns and important verbs)
        key_terms = [
            token.text for token in doc
            if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop
        ]
        
        # Detect temporal aspects
        temporal_aspects = [
            token.text for token in doc
            if token.ent_type_ in ["DATE", "TIME"]
        ]
        
        # Determine query type
        query_type = determine_query_type(doc)
        
        # Calculate complexity score
        complexity = calculate_complexity(doc)
        
        # Detect intent
        intent = detect_intent(doc)
        
        return {
            "status": "success",
            "message": "Query analyzed successfully",
            "analysis": {
                "entities": entities,
                "intent": intent,
                "key_terms": key_terms,
                "temporal_aspects": temporal_aspects,
                "query_type": query_type,
                "complexity": complexity
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing query: {str(e)}")
        return {
            "status": "error",
            "message": f"Error analyzing query: {str(e)}",
            "analysis": None
        }

def determine_query_type(doc) -> str:
    """Determine the type of query."""
    # Check for question words
    question_words = {"what", "who", "where", "when", "why", "how"}
    first_token = doc[0].text.lower()
    
    if first_token in question_words:
        return "factual"
    
    # Check for comparative words
    comparative_words = {"compare", "difference", "versus", "vs", "better", "worse"}
    if any(word in doc.text.lower() for word in comparative_words):
        return "comparative"
    
    # Check for analytical words
    analytical_words = {"analyze", "explain", "describe", "discuss"}
    if any(word in doc.text.lower() for word in analytical_words):
        return "analytical"
    
    return "general"

def calculate_complexity(doc) -> float:
    """Calculate query complexity score (0-1)."""
    # Factors that increase complexity:
    # - Number of clauses
    # - Number of entities
    # - Length of query
    # - Presence of complex structures
    
    complexity = 0.0
    
    # Length factor
    length_factor = min(len(doc) / 20, 1.0)  # Normalize to 0-1
    complexity += length_factor * 0.3
    
    # Entity factor
    entity_factor = min(len(doc.ents) / 5, 1.0)
    complexity += entity_factor * 0.3
    
    # Structure factor
    structure_factor = min(len([sent for sent in doc.sents]) / 3, 1.0)
    complexity += structure_factor * 0.4
    
    return min(complexity, 1.0)

def detect_intent(doc) -> str:
    """Detect the intent of the query."""
    # Common intents
    intents = {
        "definition": ["what is", "define", "meaning of"],
        "comparison": ["compare", "difference between", "versus"],
        "explanation": ["explain", "how does", "why does"],
        "list": ["list", "what are", "examples of"],
        "search": ["find", "search for", "look for"]
    }
    
    query_lower = doc.text.lower()
    
    for intent, patterns in intents.items():
        if any(pattern in query_lower for pattern in patterns):
            return intent
    
    return "general" 