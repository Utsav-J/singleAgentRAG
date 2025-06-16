"""
Query Optimizer Agent implementation.
"""

from typing import Dict, List, Optional
import json

from google.adk.agents import Agent
from .tools.optimize_query import optimize_query_tool
from .tools.analyze_query import analyze_query_tool
from .tools.enhance_query import enhance_query_tool

root_agent = Agent(
    name="query_optimizer",
    # Using Gemini 2.5 Flash for best performance with query optimization
    model="gemini-2.5-flash-preview-04-17",
    description="Query Optimization Agent",
    tools=[
        optimize_query_tool,
        analyze_query_tool,
        enhance_query_tool,
    ],
    instruction="""
    # 🧠 Query Optimizer Agent

    You are a helpful Query Optimization agent that enhances and structures natural language queries.
    You can analyze queries, break them down into sub-queries, and enhance them with relevant context.
    
    ## Your Capabilities
    
    1. **Query Analysis**: You can analyze queries to understand their structure and intent.
    2. **Query Optimization**: You can optimize queries for better retrieval results.
    3. **Query Enhancement**: You can enhance queries with relevant context and terms.
    
    ## How to Approach User Requests
    
    When a user provides a query:
    1. First, analyze the query to understand its structure and intent
    2. Then, optimize the query by breaking it down and restructuring it
    3. Finally, enhance the query with relevant context and terms
    
    ## Using Tools
    
    You have three specialized tools at your disposal:
    
    1. `analyze_query`: Analyze the structure and intent of a query
       - Parameters:
         - query: The text query to analyze
    
    2. `optimize_query`: Optimize a query for better retrieval
       - Parameters:
         - query: The text query to optimize
         - analysis_json: (Optional) JSON string containing analysis results
    
    3. `enhance_query`: Enhance a query with context and terms
       - Parameters:
         - query: The text query to enhance
         - analysis_json: (Optional) JSON string containing analysis results
    
    ## Communication Guidelines
    
    - Be clear and concise in your responses.
    - Explain your optimization decisions.
    - Provide structured output for better integration.
    - If an error occurs, explain what went wrong and suggest next steps.
    
    Remember, your primary goal is to improve query effectiveness through optimization.
    """
)

class QueryOptimizerAgent:
    """Agent for optimizing natural language queries."""

    def analyze(self, query: str) -> Dict:
        """
        Analyze a natural language query.

        Args:
            query (str): The query to analyze

        Returns:
            Dict: Analysis results including structure and intent
        """
        return analyze_query_tool(query=query)

    def optimize(self, query: str, analysis: Optional[Dict] = None) -> Dict:
        """
        Optimize a natural language query.

        Args:
            query (str): The query to optimize
            analysis (Dict, optional): Previous analysis results

        Returns:
            Dict: Optimized query structure
        """
        analysis_json = json.dumps(analysis) if analysis else ""
        return optimize_query_tool(query=query, analysis_json=analysis_json)

    def enhance(self, query: str, analysis: Optional[Dict] = None) -> Dict:
        """
        Enhance a query with context and terms.

        Args:
            query (str): The query to enhance
            analysis (Dict, optional): Previous analysis results

        Returns:
            Dict: Enhanced query with context and terms
        """
        analysis_json = json.dumps(analysis) if analysis else ""
        return enhance_query_tool(query=query, analysis_json=analysis_json) 