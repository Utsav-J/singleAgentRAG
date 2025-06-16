"""
Query optimization tool that uses Gemini to structure natural language queries.
"""

import logging
from typing import Dict, List, Optional
import json
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

# Initialize Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

def optimize_query_tool(query: str) -> dict:
    """
    Optimize a natural language query using Gemini to create a structured output.
    
    Args:
        query (str): The query to optimize
        
    Returns:
        dict: Structured query output containing main query, sub-queries, key terms, and context
    """
    try:
        # Create prompt for Gemini
        prompt = f"""Analyze this query and provide a structured response in JSON format:
        Query: {query}
        
        Provide a JSON object with the following structure:
        {{
            "main_query": "optimized main query",
            "sub_queries": ["list of relevant sub-queries"],
            "key_terms": ["extracted key terms"],
            "context": {{
                "domain": "determined domain",
                "time_period": "current/past/future",
                "focus": "main focus area"
            }}
        }}
        
        Ensure the response is valid JSON and maintains the exact structure above."""

        # Get response from Gemini
        response = model.generate_content(prompt)
        
        # Parse the response as JSON
        structured_output = json.loads(response.text)
        
        return structured_output
        
    except Exception as e:
        logger.error(f"Error optimizing query: {str(e)}")
        return {
            "error": f"Error optimizing query: {str(e)}",
            "main_query": query,
            "sub_queries": [query],
            "key_terms": [],
            "context": {
                "domain": "unknown",
                "time_period": "unknown",
                "focus": "unknown"
            }
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