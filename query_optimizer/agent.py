"""
Query Optimizer Agent implementation.
"""

from typing import Dict
import json

from google.adk.agents import Agent
from .tools.optimize_query import optimize_query_tool

root_agent = Agent(
    name="query_optimizer",
    # Using Gemini 2.5 Flash for best performance with query optimization
    model="gemini-2.5-flash-preview-04-17",
    description="Query Optimization Agent",
    tools=[optimize_query_tool],
    instruction="""
    # 🧠 Query Optimizer Agent

    You are a Query Optimization agent that structures natural language queries using LLM capabilities.
    
    ## Your Capability
    
    Transform natural language queries into structured JSON format with:
    - Main query
    - Relevant sub-queries
    - Key terms
    - Context information
    
    ## Output Format
    
    Always return a structured JSON response in the following format:
    {
        "main_query": "original query transformed in a proper informative sentence",
        "sub_queries": ["sub query 1", "sub query 2", ...],
        "key_terms": ["term1", "term2", ...],
        "context": {
            "domain": "domain_name",
            "time_period": "time_period",
            "focus": "focus_area"
        }
    }
    
    ## Using the Tool
    
    Use the `optimize_query` tool to structure queries:
    - Parameters:
        - query: The text query to optimize
    
    ## Important Notes
    
    - Always return the structured JSON output directly
    - Do not add any natural language explanations or responses
    - If an error occurs, return the error in the structured format
    - Maintain consistency in the output format
    """
)

class QueryOptimizerAgent:
    """Agent for optimizing natural language queries."""

    def optimize(self, query: str) -> Dict:
        """
        Optimize a natural language query.

        Args:
            query (str): The query to optimize

        Returns:
            Dict: Structured query output
        """
        return optimize_query_tool(query=query) 