"""
Tool for enhancing natural language queries with additional context and terms.
"""

import logging
from typing import Dict, List, Optional
import json

import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

from ..config import (
    DEFAULT_NLP_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    MAX_SYNONYMS_PER_TERM,
    MAX_RELATED_TERMS,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Initialize models
_nlp_model = None
_embedding_model = None

def get_nlp_model():
    """Get or initialize the spaCy model."""
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = spacy.load(DEFAULT_NLP_MODEL)
    return _nlp_model

def get_embedding_model():
    """Get or initialize the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    return _embedding_model

def enhance_query_tool(query: str, analysis_json: str = "") -> dict:
    """
    Enhance a query with additional context and terms.

    Args:
        query (str): The query to enhance
        analysis_json (str): JSON string containing the analysis results from analyze_query_tool

    Returns:
        dict: A dictionary containing:
            - status: Success or error status
            - message: Description of the result
            - enhanced_query: Dictionary containing:
                - original_query: The original query
                - enhanced_terms: List of enhanced terms
                - synonyms: Dictionary of term synonyms
                - related_terms: List of related terms
                - context_terms: List of context terms
    """
    try:
        # Parse analysis JSON if provided
        analysis = json.loads(analysis_json) if analysis_json else None
        
        if analysis and analysis.get("status") != "success":
            return {
                "status": "error",
                "message": "Invalid analysis results",
                "enhanced_query": None
            }

        # Get models
        nlp = get_nlp_model()
        embedding_model = get_embedding_model()
        
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
                    "complexity": 0.5,
                    "domain": "general"
                }
            }
        
        # Get key terms
        key_terms = analysis["analysis"]["key_terms"]
        
        # Generate synonyms
        synonyms = generate_synonyms(key_terms, nlp)
        
        # Generate related terms
        related_terms = generate_related_terms(key_terms, embedding_model)
        
        # Generate context terms
        context_terms = generate_context_terms(doc, analysis)
        
        return {
            "status": "success",
            "message": "Query enhanced successfully",
            "enhanced_query": {
                "original_query": query,
                "enhanced_terms": key_terms,
                "synonyms": synonyms,
                "related_terms": related_terms,
                "context_terms": context_terms
            }
        }
    except Exception as e:
        logger.error(f"Error enhancing query: {str(e)}")
        return {
            "status": "error",
            "message": f"Error enhancing query: {str(e)}",
            "enhanced_query": None
        }

def generate_synonyms(terms: List[str], nlp) -> Dict[str, List[str]]:
    """Generate synonyms for key terms."""
    synonyms = {}
    
    for term in terms:
        # Get the term's vector
        term_doc = nlp(term)
        if not term_doc.vector.any():
            continue
            
        # Find similar terms
        similar_terms = []
        for token in nlp.vocab:
            if token.is_lower and token.is_alpha:
                similarity = term_doc.similarity(token)
                if similarity > SIMILARITY_THRESHOLD:
                    similar_terms.append((token.text, similarity))
        
        # Sort by similarity and take top N
        similar_terms.sort(key=lambda x: x[1], reverse=True)
        synonyms[term] = [term for term, _ in similar_terms[:MAX_SYNONYMS_PER_TERM]]
    
    return synonyms

def generate_related_terms(terms: List[str], embedding_model) -> List[str]:
    """Generate related terms using sentence transformers."""
    if not terms:
        return []
    
    # Get embeddings for terms
    term_embeddings = embedding_model.encode(terms)
    
    # Get embeddings for vocabulary
    vocab = embedding_model.get_vocab()
    vocab_embeddings = embedding_model.encode(list(vocab))
    
    # Calculate similarities
    similarities = np.dot(term_embeddings, vocab_embeddings.T)
    
    # Get top related terms
    related_terms = set()
    for term_similarities in similarities:
        top_indices = np.argsort(term_similarities)[-MAX_RELATED_TERMS:]
        related_terms.update([vocab[i] for i in top_indices])
    
    return list(related_terms)

def generate_context_terms(doc, analysis: Dict) -> List[str]:
    """Generate context terms based on the query and analysis."""
    context_terms = []
    
    # Add domain-specific terms
    domain = analysis["analysis"].get("domain", "general")
    if domain != "general":
        context_terms.extend(get_domain_terms(domain))
    
    # Add temporal context
    temporal_aspects = analysis["analysis"]["temporal_aspects"]
    if temporal_aspects:
        context_terms.extend(temporal_aspects)
    
    # Add entity context
    entities = [ent["text"] for ent in analysis["analysis"]["entities"]]
    context_terms.extend(entities)
    
    return list(set(context_terms))

def get_domain_terms(domain: str) -> List[str]:
    """Get domain-specific terms."""
    domain_terms = {
        "technology": [
            "software", "hardware", "programming", "algorithm",
            "database", "network", "security", "cloud"
        ],
        "science": [
            "research", "experiment", "theory", "hypothesis",
            "analysis", "data", "methodology", "results"
        ],
        "business": [
            "market", "industry", "company", "strategy",
            "management", "finance", "marketing", "operations"
        ],
        "health": [
            "medical", "treatment", "diagnosis", "patient",
            "healthcare", "medicine", "clinical", "therapy"
        ],
        "education": [
            "learning", "teaching", "student", "curriculum",
            "education", "school", "university", "course"
        ]
    }
    
    return domain_terms.get(domain, []) 