"""
Hybrid Retrieval Module
Implements dense vector search + BM25 keyword search + reranking.
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

logger = logging.getLogger(__name__)

# Global cross-encoder model (loaded lazily)
_cross_encoder = None
_cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_cross_encoder() -> CrossEncoder:
    """Get or load the cross-encoder model for reranking."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            logger.info(f"Loading cross-encoder model: {_cross_encoder_name}")
            _cross_encoder = CrossEncoder(_cross_encoder_name)
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise
    return _cross_encoder


class HybridRetriever:
    """
    Hybrid retriever combining dense vector search and BM25 keyword search.
    """
    
    def __init__(
        self,
        chroma_client: chromadb.Client,
        collection_name: str = "ipex_epr_docs",
        top_k_dense: int = 20,
        top_k_bm25: int = 20,
        top_k_final: int = 5,
        rerank_top_k: int = 20
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            chroma_client: ChromaDB client instance
            collection_name: Name of the ChromaDB collection
            top_k_dense: Top K results from dense search
            top_k_bm25: Top K results from BM25 search
            top_k_final: Final number of results to return
            rerank_top_k: Number of results to rerank (top N from fusion)
        """
        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.collection = chroma_client.get_collection(collection_name)
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.top_k_final = top_k_final
        self.rerank_top_k = rerank_top_k
        
        # BM25 index (will be built from collection)
        self.bm25_index = None
        self.bm25_doc_ids = []
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents in the collection."""
        try:
            logger.info("Building BM25 index from ChromaDB collection...")
            
            # Get all documents from collection
            all_results = self.collection.get(include=['documents', 'metadatas'])
            
            if not all_results['documents']:
                logger.warning("No documents found in collection for BM25 index")
                return
            
            documents = all_results['documents']
            self.bm25_doc_ids = all_results['ids']
            
            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in documents]
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            logger.info(f"BM25 index built with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_index = None
    
    def _metadata_prefilter(
        self,
        query: str,
        where: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Apply metadata pre-filtering if product codes are detected in query.
        
        Args:
            query: User query
            where: Existing where clause for ChromaDB
            
        Returns:
            Updated where clause or None
        """
        # Simple product code detection (can be enhanced)
        # Look for patterns that might indicate product codes
        import re
        product_patterns = [
            r'\b[A-Z]{2,}\d{2,}[A-Z0-9]*\b',
            r'\b\d{2,}[A-Z]{2,}\d*\b',
        ]
        
        detected_codes = []
        for pattern in product_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            detected_codes.extend(matches)
        
        if detected_codes and where is None:
            # Filter by product codes in metadata
            where = {
                "$or": [
                    {"product_codes": {"$contains": code}} for code in detected_codes
                ]
            }
        
        return where
    
    def _dense_search(
        self,
        query_embedding: List[float],
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform dense vector search using ChromaDB.
        
        Args:
            query_embedding: Query embedding vector
            where: Metadata filter
            
        Returns:
            List of result dictionaries with scores
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.top_k_dense,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            dense_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    # Convert distance to similarity score (1 - distance for cosine)
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # Cosine distance -> similarity
                    
                    dense_results.append({
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': similarity,
                        'source': 'dense'
                    })
            
            return dense_results
            
        except Exception as e:
            logger.error(f"Error in dense search: {e}")
            return []
    
    def _bm25_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: User query string
            
        Returns:
            List of result dictionaries with scores
        """
        if self.bm25_index is None:
            logger.warning("BM25 index not available, skipping BM25 search")
            return []
        
        try:
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top K indices
            top_indices = np.argsort(scores)[::-1][:self.top_k_bm25]
            
            # Format results
            bm25_results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include positive scores
                    doc_id = self.bm25_doc_ids[idx]
                    
                    # Get document and metadata from collection
                    doc_result = self.collection.get(ids=[doc_id], include=['documents', 'metadatas'])
                    
                    if doc_result['documents']:
                        bm25_results.append({
                            'id': doc_id,
                            'document': doc_result['documents'][0],
                            'metadata': doc_result['metadatas'][0],
                            'score': float(scores[idx]),
                            'source': 'bm25'
                        })
            
            return bm25_results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        Args:
            dense_results: Results from dense search
            bm25_results: Results from BM25 search
            k: RRF parameter (typically 60)
            
        Returns:
            Combined and ranked results
        """
        # Create score dictionaries by document ID
        rrf_scores = {}
        
        # Add dense results
        for rank, result in enumerate(dense_results, start=1):
            doc_id = result['id']
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    'id': doc_id,
                    'document': result['document'],
                    'metadata': result['metadata'],
                    'dense_score': result['score'],
                    'bm25_score': 0.0,
                    'rrf_score': 0.0
                }
            rrf_scores[doc_id]['rrf_score'] += 1 / (k + rank)
            rrf_scores[doc_id]['dense_score'] = result['score']
        
        # Add BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result['id']
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    'id': doc_id,
                    'document': result['document'],
                    'metadata': result['metadata'],
                    'dense_score': 0.0,
                    'bm25_score': result['score'],
                    'rrf_score': 0.0
                }
            rrf_scores[doc_id]['rrf_score'] += 1 / (k + rank)
            rrf_scores[doc_id]['bm25_score'] = result['score']
        
        # Sort by RRF score
        fused_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        
        return fused_results[:self.rerank_top_k]
    
    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: User query
            candidates: Candidate documents to rerank
            
        Returns:
            Reranked results
        """
        if not candidates:
            return []
        
        try:
            cross_encoder = get_cross_encoder()
            
            # Prepare pairs for cross-encoder
            pairs = [[query, candidate['document']] for candidate in candidates]
            
            # Get reranking scores
            rerank_scores = cross_encoder.predict(pairs)
            
            # Add scores to candidates and sort
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(rerank_scores[i])
            
            reranked = sorted(
                candidates,
                key=lambda x: x['rerank_score'],
                reverse=True
            )
            
            return reranked[:self.top_k_final]
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Return original candidates if reranking fails
            return candidates[:self.top_k_final]
    
    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        where: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Perform hybrid retrieval with reranking.
        
        Args:
            query: User query string
            query_embedding: Query embedding vector
            where: Optional metadata filter
            
        Returns:
            Tuple of (retrieved chunks, latency metrics)
        """
        start_time = time.time()
        latency_metrics = {}
        
        # 1. Metadata pre-filtering
        prefilter_start = time.time()
        where = self._metadata_prefilter(query, where)
        latency_metrics['prefilter'] = time.time() - prefilter_start
        
        # 2. Parallel dense and BM25 search
        search_start = time.time()
        dense_results = self._dense_search(query_embedding, where)
        bm25_results = self._bm25_search(query)
        latency_metrics['search'] = time.time() - search_start
        
        # 3. Reciprocal Rank Fusion
        fusion_start = time.time()
        fused_results = self._reciprocal_rank_fusion(dense_results, bm25_results)
        latency_metrics['fusion'] = time.time() - fusion_start
        
        # 4. Reranking
        rerank_start = time.time()
        final_results = self._rerank(query, fused_results)
        latency_metrics['rerank'] = time.time() - rerank_start
        
        # Total latency
        latency_metrics['total'] = time.time() - start_time
        
        return final_results, latency_metrics

