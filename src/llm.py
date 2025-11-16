"""
LLM Module
Handles Claude 3 Haiku API calls with streaming support.
"""

import logging
import re
from typing import List, Dict, Any, Iterator, Optional
from anthropic import Anthropic

logger = logging.getLogger(__name__)

# System prompt for the assistant
SYSTEM_PROMPT = """You are a technical assistant for IPEX products. Your role is to help users understand technical documentation, product specifications, installation procedures, and features.

Guidelines:
- Always cite sources using the format [doc_name, page_num] when referencing information from documents
- Be accurate and concise
- If information is not available in the provided context, say so clearly
- Focus on technical accuracy and clarity
- When discussing products, include relevant product codes if mentioned in the context
"""


class ClaudeLLM:
    """
    Wrapper for Claude 3 Haiku API with streaming support.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key
        """
        try:
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-3-haiku-20240307"
            logger.info("Claude client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            raise
    
    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string for the LLM.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk.get('metadata', {})
            doc_name = metadata.get('doc_name', 'Unknown')
            page_num = metadata.get('page_num', 'Unknown')
            content_type = metadata.get('content_type', 'text')
            
            context_parts.append(
                f"[Document {i}: {doc_name}, Page {page_num}, Type: {content_type}]\n"
                f"{chunk.get('document', '')}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def extract_citations(self, text: str) -> List[Dict[str, str]]:
        """
        Extract citations from LLM response text.
        Supports multiple formats:
        - [Document X: doc_name, Page Y]
        - [doc_name, Page Y]
        - [doc_name, page_num]
        - (Reference: doc_name, p.XX)
        - (References: doc_name, p.XX)
        
        Args:
            text: Response text from LLM
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        # Pattern to match citations in bracket format
        citation_pattern_brackets = r'\[(?:Document\s+\d+:\s*)?([^,\]]+?)(?:,\s*Page\s+(\d+)|,\s*(\d+))\]'
        matches = re.finditer(citation_pattern_brackets, text)
        
        for match in matches:
            doc_name = match.group(1).strip()
            # Remove "Document X:" prefix if present in doc_name
            doc_name = re.sub(r'^Document\s+\d+:\s*', '', doc_name)
            page_num = match.group(2) or match.group(3)
            if page_num:
                citations.append({
                    'doc_name': doc_name,
                    'page_num': int(page_num)
                })
        
        # Pattern to match citations in parentheses format
        citation_pattern_parens = r'\(Reference(?:s)?:\s*([^,)]+?)(?:,\s*p\.(\d+)|,\s*(\d+))\)'
        matches = re.finditer(citation_pattern_parens, text)
        
        for match in matches:
            doc_name = match.group(1).strip()
            page_num = match.group(2) or match.group(3)
            if page_num:
                citations.append({
                    'doc_name': doc_name,
                    'page_num': int(page_num)
                })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in citations:
            key = (citation['doc_name'], citation['page_num'])
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        
        return unique_citations
    
    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        stream: bool = True
    ) -> Iterator[str]:
        """
        Generate response using Claude with streaming.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks
            stream: Whether to stream the response
            
        Yields:
            Text chunks as they are generated
        """
        # Format context
        context = self.format_context(retrieved_chunks)
        
        # Construct user message
        user_message = f"""Context from technical documentation:

{context}

User Question: {query}

Please provide a helpful answer based on the context above. Remember to cite sources using [doc_name, page_num] format."""
        
        try:
            if stream:
                # Streaming response
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}]
                ) as stream_obj:
                    for text in stream_obj.text_stream:
                        yield text
            else:
                # Non-streaming response
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}]
                )
                yield message.content[0].text
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"Error: {str(e)}"
    
    def generate_response_sync(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate response synchronously (non-streaming).
        
        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Complete response text
        """
        response = ""
        for chunk in self.generate_response(query, retrieved_chunks, stream=False):
            response += chunk
        return response


def get_claude_llm(api_key: Optional[str] = None) -> ClaudeLLM:
    """
    Create a ClaudeLLM instance.
    
    Args:
        api_key: Anthropic API key (if None, will need to be set via environment)
        
    Returns:
        ClaudeLLM instance
    """
    if api_key is None:
        import os
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    return ClaudeLLM(api_key)

