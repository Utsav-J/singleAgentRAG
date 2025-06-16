"""
Tool for optimizing natural language queries.
"""

import logging
from typing import Dict, List, Optional
import json

import spacy
from sentence_transformers import SentenceTransformer

from ..config import (
    DEFAULT_NLP_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    MAX_SUB_QUERIES,
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

def optimize_query_tool(query: str, analysis_json: str = "") -> dict:
    """
    Optimize a natural language query based on its analysis.

    Args:
        query (str): The query to optimize
        analysis_json (str): JSON string containing the analysis results from analyze_query_tool

    Returns:
        dict: A dictionary containing:
            - status: Success or error status
            - message: Description of the result
            - optimized_query: Dictionary containing:
                - main_query: The main optimized query
                - sub_queries: List of sub-queries
                - context: Additional context
                - constraints: Query constraints
    """
    try:
        # Parse analysis JSON if provided
        analysis = json.loads(analysis_json) if analysis_json else None
        
        if analysis and analysis.get("status") != "success":
            return {
                "status": "error",
                "message": "Invalid analysis results",
                "optimized_query": None
            }

        # Get NLP model
        nlp = get_nlp_model()
        
        # Process query
        doc = nlp(query)
        
        # If no analysis provided, create basic analysis
        if not analysis:
            analysis = {
                "status": "success",
                "analysis": {
                    "entities": [],
                    "key_terms": [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop],
                    "temporal_aspects": [],
                    "query_type": "general",
                    "complexity": 0.5
                }
            }
        
        # Generate main query
        main_query = generate_main_query(doc, analysis)
        
        # Generate sub-queries
        sub_queries = generate_sub_queries(doc, analysis)
        
        # Extract context
        context = extract_context(doc, analysis)
        
        # Extract constraints
        constraints = extract_constraints(doc, analysis)
        
        return {
            "status": "success",
            "message": "Query optimized successfully",
            "optimized_query": {
                "main_query": main_query,
                "sub_queries": sub_queries[:MAX_SUB_QUERIES],
                "context": context,
                "constraints": constraints
            }
        }
    except Exception as e:
        logger.error(f"Error optimizing query: {str(e)}")
        return {
            "status": "error",
            "message": f"Error optimizing query: {str(e)}",
            "optimized_query": None
        }

def generate_main_query(doc, analysis: Dict) -> str:
    """Generate the main optimized query."""
    # Start with the original query
    main_query = doc.text
    
    # Add context from entities if missing
    entities = [ent["text"] for ent in analysis["analysis"]["entities"]]
    for entity in entities:
        if entity not in main_query:
            main_query = f"{main_query} {entity}"
    
    # Add key terms if missing
    key_terms = analysis["analysis"]["key_terms"]
    for term in key_terms:
        if term not in main_query:
            main_query = f"{main_query} {term}"
    
    return main_query.strip()

def generate_sub_queries(doc, analysis: Dict) -> List[str]:
    """Generate sub-queries based on the main query."""
    sub_queries = []
    
    # Generate sub-queries based on entities
    entities = [ent["text"] for ent in analysis["analysis"]["entities"]]
    for entity in entities:
        sub_query = f"What is {entity}?"
        sub_queries.append(sub_query)
    
    # Generate sub-queries based on key terms
    key_terms = analysis["analysis"]["key_terms"]
    for term in key_terms:
        if term not in entities:  # Avoid duplicate queries
            sub_query = f"How does {term} relate to the main topic?"
            sub_queries.append(sub_query)
    
    # Generate sub-queries based on temporal aspects
    temporal_aspects = analysis["analysis"]["temporal_aspects"]
    for aspect in temporal_aspects:
        sub_query = f"What happened during {aspect}?"
        sub_queries.append(sub_query)
    
    return sub_queries

def extract_context(doc, analysis: Dict) -> Dict:
    """Extract context from the query."""
    context = {
        "domain": determine_domain(doc, analysis),
        "time_period": analysis["analysis"]["temporal_aspects"],
        "entities": [ent["text"] for ent in analysis["analysis"]["entities"]],
        "key_terms": analysis["analysis"]["key_terms"]
    }
    return context

def extract_constraints(doc, analysis: Dict) -> Dict:
    """Extract constraints from the query."""
    constraints = {
        "temporal": analysis["analysis"]["temporal_aspects"],
        "entities": [ent["text"] for ent in analysis["analysis"]["entities"]],
        "query_type": analysis["analysis"]["query_type"],
        "complexity": analysis["analysis"]["complexity"]
    }
    return constraints

def determine_domain(doc, analysis: Dict) -> str:
    """Determine the domain of the query."""
    # Common domains and their indicators
    domains = {
        "technology": ["computer", "software", "hardware", "programming", "code"],
        "science": ["research", "experiment", "study", "scientific"],
        "business": ["company", "market", "industry", "business"],
        "health": ["medical", "health", "disease", "treatment"],
        "education": ["learn", "study", "education", "school"]
    }
    
    query_lower = doc.text.lower()
    
    for domain, indicators in domains.items():
        if any(indicator in query_lower for indicator in indicators):
            return domain
    
    return "general" 