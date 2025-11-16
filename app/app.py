"""
Streamlit RAG Chatbot App
Main application for querying technical PDF documents.
"""

import sys
import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import chromadb

# Add parent directory to path to import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings import embed_query, get_embedding_model, embed_documents
from src.retrieval import HybridRetriever
from src.query_router import QueryRouter, QueryType
from src.llm import get_claude_llm, ClaudeLLM
from src.ingestion import process_pdf_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="IPEX RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "query_router" not in st.session_state:
    st.session_state.query_router = None


@st.cache_resource
def load_embedding_model():
    """Load embedding model (cached)."""
    try:
        return get_embedding_model()
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None


def build_index_if_needed(client, collection_name: str = "ipex_epr_docs", pdf_dir: str = "./data/pdfs"):
    """Build ChromaDB index if collection doesn't exist or is empty."""
    try:
        # Check if collection exists and has documents
        try:
            collection = client.get_collection(collection_name)
            if collection.count() > 0:
                return True  # Index already exists
        except Exception:
            pass  # Collection doesn't exist, need to create it
        
        # Check if PDFs exist
        pdf_path = Path(pdf_dir)
        if not pdf_path.exists():
            return False
        
        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            return False
        
        # Build the index
        logger.info(f"Building index from {len(pdf_files)} PDF files...")
        
        # Process PDFs
        chunks = process_pdf_directory(pdf_dir)
        if not chunks:
            logger.warning("No chunks created from PDFs")
            return False
        
        # Create collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 16,
                "description": "IPEX EPR technical documentation"
            }
        )
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for idx, chunk in enumerate(chunks):
            documents.append(chunk['content'])
            metadatas.append({
                'doc_name': chunk['doc_name'],
                'page_num': chunk['page_num'],
                'content_type': chunk['content_type'],
                'product_codes': ','.join(chunk['product_codes']) if chunk['product_codes'] else '',
            })
            ids.append(f"{chunk['doc_name']}_page{chunk['page_num']}_chunk{idx}")
        
        # Generate embeddings in batches
        batch_size = 100
        embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embed_documents(batch_docs)
            embeddings.extend(batch_embeddings.tolist())
        
        # Add to ChromaDB collection
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully built index with {len(chunks)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        return False


@st.cache_resource
def load_chroma_client(chroma_db_path: str = "./data/chroma_db"):
    """Load ChromaDB client (cached)."""
    try:
        client = chromadb.PersistentClient(path=chroma_db_path)
        # Check if collection exists
        try:
            collection = client.get_collection("ipex_epr_docs")
            if collection.count() == 0:
                # Try to build index automatically
                if build_index_if_needed(client):
                    # Re-get collection after building
                    collection = client.get_collection("ipex_epr_docs")
                    if collection.count() > 0:
                        return client
                st.warning("ChromaDB collection is empty. Please add PDFs to data/pdfs/ directory.")
                return None
            return client
        except Exception:
            # Collection doesn't exist, try to build it automatically
            if build_index_if_needed(client):
                # Re-get collection after building
                try:
                    collection = client.get_collection("ipex_epr_docs")
                    if collection.count() > 0:
                        return client
                except Exception:
                    pass
            st.error("ChromaDB collection 'ipex_epr_docs' not found. Please add PDFs to data/pdfs/ directory and rebuild.")
            return None
    except Exception as e:
        st.error(f"Failed to load ChromaDB client: {e}")
        return None


@st.cache_resource
def load_retriever(_chroma_client, collection_name: str = "ipex_epr_docs"):
    """Load hybrid retriever (cached)."""
    try:
        return HybridRetriever(_chroma_client, collection_name=collection_name)
    except Exception as e:
        st.error(f"Failed to load retriever: {e}")
        return None


def initialize_components():
    """Initialize all components."""
    # Load embedding model
    embedding_model = load_embedding_model()
    if embedding_model is None:
        return False
    
    # Load ChromaDB client
    chroma_client = load_chroma_client()
    if chroma_client is None:
        return False
    
    # Load retriever
    if st.session_state.retriever is None:
        st.session_state.retriever = load_retriever(chroma_client)
        if st.session_state.retriever is None:
            return False
    
    # Initialize query router
    if st.session_state.query_router is None:
        st.session_state.query_router = QueryRouter()
    
    return True


def get_api_key():
    """Get API key from Streamlit secrets or sidebar input."""
    # Try secrets first
    try:
        if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
            return st.secrets['ANTHROPIC_API_KEY']
    except Exception:
        pass
    
    # Fallback to sidebar input
    api_key = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        help="Enter your Anthropic API key or set it in Streamlit secrets"
    )
    return api_key


def main():
    """Main application."""
    st.title("📚 IPEX RAG Chatbot")
    st.markdown("Query technical PDF documents with AI-powered search")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API key input
        api_key = get_api_key()
        
        if api_key:
            try:
                if st.session_state.llm is None:
                    st.session_state.llm = get_claude_llm(api_key)
            except Exception as e:
                st.error(f"Failed to initialize LLM: {e}")
                st.session_state.llm = None
        
        st.divider()
        
        # Initialize components
        if st.button("Initialize Components", type="primary"):
            with st.spinner("Loading models and database..."):
                if initialize_components():
                    st.success("Components initialized successfully!")
                else:
                    st.error("Failed to initialize components. Check logs for details.")
        
        st.divider()
        
        # Info
        st.info("""
        **Instructions:**
        1. Add PDFs to `data/pdfs/` directory
        2. Run `python scripts/build_index.py` to build the index
        3. Initialize components using the button above
        4. Start chatting!
        """)
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Check if components are initialized
    if st.session_state.retriever is None or st.session_state.llm is None:
        st.warning("⚠️ Please initialize components in the sidebar first.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display citations if available
            if "citations" in message and message["citations"]:
                with st.expander("📎 Citations"):
                    for citation in message["citations"]:
                        st.write(f"- **{citation['doc_name']}**, Page {citation['page_num']}")
            
            # Display retrieval sources if available
            if "sources" in message and message["sources"]:
                with st.expander("🔍 Retrieval Sources"):
                    for i, source in enumerate(message["sources"], start=1):
                        metadata = source.get("metadata", {})
                        st.write(f"**Source {i}:** {metadata.get('doc_name', 'Unknown')}, Page {metadata.get('page_num', 'Unknown')}")
                        st.text_area(
                            f"Content {i}",
                            source.get("document", "")[:500] + "..." if len(source.get("document", "")) > 500 else source.get("document", ""),
                            height=100,
                            key=f"source_{message.get('id', 0)}_{i}",
                            disabled=True
                        )
            
            # Display latency metrics if available
            if "latency" in message and message["latency"]:
                with st.expander("⏱️ Latency Metrics"):
                    metrics = message["latency"]
                    st.metric("Total", f"{metrics.get('total', 0):.3f}s")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Retrieval", f"{metrics.get('search', 0):.3f}s")
                    with col2:
                        st.metric("Reranking", f"{metrics.get('rerank', 0):.3f}s")
                    with col3:
                        st.metric("LLM", f"{metrics.get('llm', 0):.3f}s")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the technical documentation..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            try:
                # Route query
                route_result = st.session_state.query_router.classify(prompt)
                query_type = route_result["type"]
                
                # Generate query embedding
                query_embedding = embed_query(prompt).tolist()
                
                # Retrieve relevant chunks
                retrieval_start = time.time()
                retrieved_chunks, retrieval_metrics = st.session_state.retriever.retrieve(
                    prompt,
                    query_embedding
                )
                retrieval_time = time.time() - retrieval_start
                
                if not retrieved_chunks:
                    st.warning("No relevant documents found. Please try rephrasing your query.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I couldn't find any relevant information in the documents. Please try rephrasing your query.",
                        "citations": [],
                        "sources": [],
                        "latency": retrieval_metrics
                    })
                    return
                
                # Generate response with streaming
                llm_start = time.time()
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in st.session_state.llm.generate_response(prompt, retrieved_chunks, stream=True):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                llm_time = time.time() - llm_start
                
                # Extract citations
                citations = st.session_state.llm.extract_citations(full_response)
                
                # Combine latency metrics
                total_metrics = {
                    **retrieval_metrics,
                    "llm": llm_time,
                    "total": retrieval_metrics.get("total", 0) + llm_time
                }
                
                # Add assistant message to history
                message_id = len(st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "citations": citations,
                    "sources": retrieved_chunks,
                    "latency": total_metrics,
                    "id": message_id
                })
                
                # Display citations
                if citations:
                    with st.expander("📎 Citations"):
                        for citation in citations:
                            st.write(f"- **{citation['doc_name']}**, Page {citation['page_num']}")
                
                # Display retrieval sources
                with st.expander("🔍 Retrieval Sources"):
                    for i, source in enumerate(retrieved_chunks, start=1):
                        metadata = source.get("metadata", {})
                        st.write(f"**Source {i}:** {metadata.get('doc_name', 'Unknown')}, Page {metadata.get('page_num', 'Unknown')}")
                        st.text_area(
                            f"Content {i}",
                            source.get("document", "")[:500] + "..." if len(source.get("document", "")) > 500 else source.get("document", ""),
                            height=100,
                            key=f"source_{message_id}_{i}",
                            disabled=True
                        )
                
                # Display latency metrics
                with st.expander("⏱️ Latency Metrics"):
                    st.metric("Total", f"{total_metrics.get('total', 0):.3f}s")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Retrieval", f"{total_metrics.get('search', 0):.3f}s")
                    with col2:
                        st.metric("Reranking", f"{total_metrics.get('rerank', 0):.3f}s")
                    with col3:
                        st.metric("LLM", f"{total_metrics.get('llm', 0):.3f}s")
                
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "citations": [],
                    "sources": [],
                    "latency": {}
                })


if __name__ == "__main__":
    main()

