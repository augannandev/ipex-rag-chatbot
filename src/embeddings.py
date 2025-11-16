"""
Embeddings Module
Handles loading and using sentence-transformers for embeddings.
"""

import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

# Global model instance (will be loaded lazily)
_model = None
_model_name = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> SentenceTransformer:
    """
    Get or load the embedding model (singleton pattern).
    """
    global _model
    if _model is None:
        try:
            logger.info(f"Loading embedding model: {_model_name}")
            _model = SentenceTransformer(_model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    return _model


def embed_text(text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Generate embeddings for text or list of texts.
    
    Args:
        text: Single string or list of strings
        
    Returns:
        numpy array (single text) or list of numpy arrays (multiple texts)
    """
    model = get_embedding_model()
    
    try:
        if isinstance(text, str):
            embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding
        else:
            embeddings = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


def embed_query(query: str) -> np.ndarray:
    """
    Generate embedding for a single query string.
    
    Args:
        query: Query string
        
    Returns:
        numpy array embedding
    """
    return embed_text(query)


def embed_documents(documents: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of documents.
    
    Args:
        documents: List of document strings
        
    Returns:
        numpy array of embeddings (shape: [num_docs, embedding_dim])
    """
    return embed_text(documents)

