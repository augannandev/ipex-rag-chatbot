"""
Query Router Module
Routes queries to traditional RAG or agentic RAG based on complexity.
"""

import logging
import re
from typing import Dict, Literal
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Query type enumeration."""
    SIMPLE = "simple"  # Traditional RAG
    COMPLEX = "complex"  # Agentic RAG (Phase 2)


class QueryRouter:
    """
    Rule-based query router to classify queries as simple or complex.
    """
    
    def __init__(self):
        """Initialize the query router with classification rules."""
        # Patterns for simple queries (traditional RAG)
        self.simple_patterns = [
            r'^what is\b',
            r'^what are\b',
            r'^show me\b',
            r'^tell me\b',
            r'^describe\b',
            r'^list\b',
            r'^find\b',
            r'^search\b',
            r'^lookup\b',
            r'^get\b',
            r'^give me\b',
            # Product code lookup (single code)
            r'^\s*[A-Z]{2,}\d{2,}[A-Z0-9]*\s*$',
            r'^\s*\d{2,}[A-Z]{2,}\d*\s*$',
        ]
        
        # Patterns for complex queries (agentic RAG)
        self.complex_patterns = [
            r'\brecommend\b',
            r'\bwhy\b',
            r'\bhow\b',
            r'\bcompare\b',
            r'\bwhich\b.*\bbetter\b',
            r'\bwhat.*\bdifference\b',
            r'\bshould i\b',
            r'\bwhat.*\bbest\b',
            r'\bexplain.*\bwhy\b',
            r'\bwhen.*\buse\b',
            r'\bmultiple\b',
            r'\bseveral\b',
            r'\bcombination\b',
        ]
        
        # Compile patterns for efficiency
        self.simple_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.simple_patterns]
        self.complex_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.complex_patterns]
    
    def classify(self, query: str) -> Dict[str, any]:
        """
        Classify a query as simple or complex.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with 'type', 'confidence', and 'reason'
        """
        query_lower = query.lower().strip()
        
        # Check for complex patterns first (higher priority)
        complex_matches = sum(
            1 for pattern in self.complex_regex if pattern.search(query_lower)
        )
        
        # Check for simple patterns
        simple_matches = sum(
            1 for pattern in self.simple_regex if pattern.search(query_lower)
        )
        
        # Determine query type
        if complex_matches > 0:
            # Complex query - but for now, route to traditional RAG
            # (Agentic layer not yet implemented)
            confidence = min(0.8, 0.5 + (complex_matches * 0.1))
            return {
                'type': QueryType.SIMPLE,  # Route to traditional RAG for now
                'confidence': confidence,
                'reason': f'Detected {complex_matches} complex pattern(s), but routing to traditional RAG (agentic layer not implemented)',
                'detected_complexity': True
            }
        elif simple_matches > 0:
            confidence = min(0.9, 0.6 + (simple_matches * 0.1))
            return {
                'type': QueryType.SIMPLE,
                'confidence': confidence,
                'reason': f'Detected {simple_matches} simple pattern(s)',
                'detected_complexity': False
            }
        else:
            # Default to simple (traditional RAG)
            return {
                'type': QueryType.SIMPLE,
                'confidence': 0.5,
                'reason': 'No specific patterns detected, defaulting to traditional RAG',
                'detected_complexity': False
            }
    
    def route(self, query: str) -> QueryType:
        """
        Route a query and return the query type.
        
        Args:
            query: User query string
            
        Returns:
            QueryType enum value
        """
        classification = self.classify(query)
        return classification['type']


def get_query_router() -> QueryRouter:
    """
    Get a singleton instance of QueryRouter.
    
    Returns:
        QueryRouter instance
    """
    return QueryRouter()

