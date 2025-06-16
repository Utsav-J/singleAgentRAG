# Query Optimizer Agent

A powerful agent for analyzing, optimizing, and enhancing natural language queries.

## Features

- Query Analysis: Understand query structure and intent
- Query Optimization: Break down and restructure queries for better results
- Query Enhancement: Add relevant context and terms to queries

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd query-optimizer
```

2. Install the package and dependencies:
```bash
pip install -e .
```

This will:
- Install all required Python packages
- Download the required spaCy model (en_core_web_lg)

## Usage

```python
from query_optimizer.agent import QueryOptimizerAgent

# Initialize the agent
agent = QueryOptimizerAgent()

# Analyze a query
analysis = agent.analyze("What are the features of a vision transformer?")

# Optimize a query
optimized = agent.optimize("What are the features of a vision transformer?", analysis)

# Enhance a query
enhanced = agent.enhance("What are the features of a vision transformer?", analysis)
```

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for full list of dependencies

## License

[Your License Here]
