"""
Streamlit RAG Chatbot App
Main application for querying technical PDF documents.
"""

import sys
import os
import logging
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import chromadb
import pandas as pd

# Add parent directory to path to import src modules
project_root = Path(__file__).parent.parent
project_root_str = str(project_root.resolve())
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# Verify src module can be imported
try:
    from src.embeddings import embed_query, get_embedding_model, embed_documents
    from src.retrieval import HybridRetriever
    from src.query_router import QueryRouter, QueryType
    from src.llm import get_claude_llm, ClaudeLLM
    from src.ingestion import process_pdf_directory
except ImportError as e:
    # If imports fail, try adding current working directory to path
    import os
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    # Retry imports
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
    page_title="IPEX Technical Documentation Chatbot",
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
if "show_latency_metrics" not in st.session_state:
    st.session_state.show_latency_metrics = False


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
            count = collection.count()
            if count == 0:
                # Collection exists but is empty, try to build index automatically
                logger.info("Collection exists but is empty, building index...")
                if build_index_if_needed(client):
                    # Re-get collection after building
                    collection = client.get_collection("ipex_epr_docs")
                    if collection.count() > 0:
                        logger.info(f"Index built successfully with {collection.count()} documents")
                        return client
                st.warning("ChromaDB collection is empty. Please add PDFs to data/pdfs/ directory.")
                return None
            else:
                logger.info(f"Using existing index with {count} documents")
                return client
        except Exception as e:
            # Collection doesn't exist, try to build it automatically
            logger.info("Collection not found, building index...")
            if build_index_if_needed(client):
                # Re-get collection after building
                try:
                    collection = client.get_collection("ipex_epr_docs")
                    count = collection.count()
                    if count > 0:
                        logger.info(f"Index built successfully with {count} documents")
                        return client
                except Exception as ex:
                    logger.error(f"Failed to verify collection after building: {ex}")
            st.error("ChromaDB collection 'ipex_epr_docs' not found. Please add PDFs to data/pdfs/ directory.")
            return None
    except Exception as e:
        st.error(f"Failed to load ChromaDB client: {e}")
        logger.error(f"ChromaDB client error: {e}", exc_info=True)
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


def detect_and_format_tables(text: str) -> Tuple[str, List[pd.DataFrame]]:
    """
    Detect table-like patterns in text and format them as tables.
    
    Args:
        text: Response text that may contain table data
        
    Returns:
        Tuple of (formatted_text, list_of_dataframes)
    """
    tables_found = []
    formatted_text = text
    
    # Pattern 1: Detect lines with "Size:", "Part Number:", "Product Code:" pattern
    # Example: "Size: 1-1/4, Part Number: EPR25, Product Code: 077982"
    # Also handles variations like "Size 1-1/4, Part Number EPR25, Product Code 077982"
    # Match pattern that can span multiple lines
    table_pattern1 = r'(?:Size|size)[:\s]+([^,\n]+?),\s*(?:Part\s+Number|Part Number|Part#|Part\s*#)[:\s]*([^,\n]+?),\s*(?:Product\s+Code|Product Code|Code)[:\s]*([^\n]+?)(?=\n|$)'
    
    matches = list(re.finditer(table_pattern1, text, re.IGNORECASE | re.MULTILINE))
    if matches and len(matches) >= 2:  # Need at least 2 rows to be considered a table
        rows = []
        start_pos = matches[0].start()
        end_pos = matches[-1].end()
        
        for match in matches:
            size = match.group(1).strip()
            part_number = match.group(2).strip()
            product_code = match.group(3).strip()
            # Clean up any trailing punctuation or whitespace
            product_code = product_code.rstrip('.,;')
            rows.append({
                'Size (in)': size,
                'Part Number': part_number,
                'Product Code': product_code
            })
        
        if rows:
            df = pd.DataFrame(rows)
            tables_found.append(df)
            # Replace the entire matched section with a single placeholder
            formatted_text = formatted_text[:start_pos] + '[TABLE_PLACEHOLDER]' + formatted_text[end_pos:]
    
    # Pattern 2: Detect markdown-style tables or pipe-separated tables
    # Pattern for lines with multiple columns separated by | or consistent spacing
    lines = text.split('\n')
    table_lines = []
    in_table = False
    current_table_lines = []
    
    for i, line in enumerate(lines):
        # Check if line looks like a table row (has multiple | or consistent spacing)
        if '|' in line and line.count('|') >= 2:
            if not in_table:
                in_table = True
                current_table_lines = []
            current_table_lines.append(line)
        elif in_table and (line.strip() == '' or not ('|' in line and line.count('|') >= 2)):
            # End of table
            if len(current_table_lines) >= 2:  # At least header + 1 row
                table_lines.append((i - len(current_table_lines), current_table_lines))
            in_table = False
            current_table_lines = []
        elif in_table:
            current_table_lines.append(line)
    
    # Process detected table lines
    for start_idx, table_data in table_lines:
        try:
            # Try to parse as markdown table
            table_text = '\n'.join(table_data)
            # Remove markdown table formatting and parse
            clean_lines = [line.strip('|').strip() for line in table_data if line.strip()]
            if len(clean_lines) >= 2:
                headers = [h.strip() for h in clean_lines[0].split('|')]
                rows_data = []
                for row_line in clean_lines[2:]:  # Skip header and separator
                    row_values = [v.strip() for v in row_line.split('|')]
                    if len(row_values) == len(headers):
                        rows_data.append(dict(zip(headers, row_values)))
                
                if rows_data:
                    df = pd.DataFrame(rows_data)
                    tables_found.append(df)
                    # Replace table in text
                    placeholder = f'[TABLE_PLACEHOLDER_{len(tables_found)}]'
                    formatted_text = formatted_text.replace(table_text, placeholder)
        except Exception as e:
            logger.warning(f"Failed to parse table: {e}")
            continue
    
    return formatted_text, tables_found


def format_response_with_tables(text: str) -> Tuple[str, List[pd.DataFrame]]:
    """
    Format response text, detecting and extracting tables.
    
    Args:
        text: Response text
        
    Returns:
        Tuple of (formatted_text_with_placeholders, list_of_dataframes)
    """
    return detect_and_format_tables(text)


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
    st.title("📚 IPEX Technical Documentation Chatbot")
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
                # Check if this is first-time setup
                try:
                    client = chromadb.PersistentClient(path="./data/chroma_db")
                    try:
                        collection = client.get_collection("ipex_epr_docs")
                        if collection.count() == 0:
                            st.info("🔄 Building index from PDFs (first time setup - this may take a minute)...")
                    except Exception:
                        st.info("🔄 Building index from PDFs (first time setup - this may take a minute)...")
                except Exception:
                    pass
                
                if initialize_components():
                    st.success("Components initialized successfully!")
                else:
                    st.error("Failed to initialize components. Check logs for details.")
        
        st.divider()
        
        # Dev mode toggle
        st.subheader("Developer Options")
        st.session_state.show_latency_metrics = st.checkbox(
            "Show Latency Metrics",
            value=st.session_state.show_latency_metrics,
            help="Enable to display performance metrics for debugging"
        )
        
        st.divider()
        
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
            # Style citations in the content
            content = message["content"]
            # Match citations in multiple formats:
            # [Document X: doc_name, Page Y] or [doc_name, page_num] or [doc_name, Page Y]
            # (Reference: doc_name, p.XX) or (References: doc_name, p.XX)
            citation_pattern_brackets = r'\[(?:Document\s+\d+:\s*)?([^,\]]+?)(?:,\s*Page\s+(\d+)|,\s*(\d+))\]'
            citation_pattern_parens = r'\(Reference(?:s)?:\s*([^,)]+?)(?:,\s*p\.(\d+)|,\s*(\d+))\)'
            
            def style_citation_brackets(match):
                full_match = match.group(0)
                return f'<span style="color: #0066cc; font-style: italic; background-color: #e6f2ff; padding: 3px 6px; border-radius: 4px; font-weight: 500; border: 1px solid #b3d9ff;">{full_match}</span>'
            
            def style_citation_parens(match):
                full_match = match.group(0)
                return f'<span style="color: #0066cc; font-style: italic; background-color: #e6f2ff; padding: 3px 6px; border-radius: 4px; font-weight: 500; border: 1px solid #b3d9ff;">{full_match}</span>'
            
            # Detect and format tables in message history
            if message["role"] == "assistant" and "tables" in message and message["tables"]:
                # Format text with tables
                formatted_text, _ = format_response_with_tables(content)
                styled_content = re.sub(citation_pattern_brackets, style_citation_brackets, formatted_text)
                styled_content = re.sub(citation_pattern_parens, style_citation_parens, styled_content)
                
                # Display text with tables interspersed
                parts = re.split(r'\[TABLE_PLACEHOLDER(?:_\d+)?\]', styled_content)
                table_idx = 0
                for i, part in enumerate(parts):
                    if part.strip():
                        st.markdown(part, unsafe_allow_html=True)
                    if i < len(parts) - 1 and table_idx < len(message["tables"]):
                        # Display table
                        st.dataframe(message["tables"][table_idx], use_container_width=True, hide_index=True)
                        table_idx += 1
            else:
                # Replace citations with styled HTML (both formats)
                styled_content = re.sub(citation_pattern_brackets, style_citation_brackets, content)
                styled_content = re.sub(citation_pattern_parens, style_citation_parens, styled_content)
                st.markdown(styled_content, unsafe_allow_html=True)
            
            # Display citations if available
            if "citations" in message and message["citations"]:
                with st.expander("📎 Citations"):
                    for citation in message["citations"]:
                        st.markdown(
                            f'<p style="color: #1f77b4; font-style: italic; margin: 0.5em 0;">'
                            f'<strong>{citation["doc_name"]}</strong>, Page {citation["page_num"]}'
                            f'</p>',
                            unsafe_allow_html=True
                        )
            
            # Display retrieval sources if available and dev mode is enabled
            if st.session_state.show_latency_metrics and "sources" in message and message["sources"]:
                with st.expander("🔍 Retrieval Sources (Dev Mode)"):
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
            
            # Display latency metrics if available and dev mode is enabled
            if st.session_state.show_latency_metrics and "latency" in message and message["latency"]:
                with st.expander("⏱️ Latency Metrics (Dev Mode)"):
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
                # Match citations in multiple formats:
                # [Document X: doc_name, Page Y] or [doc_name, page_num] or [doc_name, Page Y]
                # (Reference: doc_name, p.XX) or (References: doc_name, p.XX)
                citation_pattern_brackets = r'\[(?:Document\s+\d+:\s*)?([^,\]]+?)(?:,\s*Page\s+(\d+)|,\s*(\d+))\]'
                citation_pattern_parens = r'\(Reference(?:s)?:\s*([^,)]+?)(?:,\s*p\.(\d+)|,\s*(\d+))\)'
                
                def style_citation_brackets(match):
                    full_match = match.group(0)
                    return f'<span style="color: #0066cc; font-style: italic; background-color: #e6f2ff; padding: 3px 6px; border-radius: 4px; font-weight: 500; border: 1px solid #b3d9ff;">{full_match}</span>'
                
                def style_citation_parens(match):
                    full_match = match.group(0)
                    return f'<span style="color: #0066cc; font-style: italic; background-color: #e6f2ff; padding: 3px 6px; border-radius: 4px; font-weight: 500; border: 1px solid #b3d9ff;">{full_match}</span>'
                
                # Batch chunks for smoother streaming (reduce update frequency)
                chunk_buffer = ""
                last_update_time = time.time()
                update_interval = 0.05  # Update every 50ms for smoother experience
                
                for chunk in st.session_state.llm.generate_response(prompt, retrieved_chunks, stream=True):
                    full_response += chunk
                    chunk_buffer += chunk
                    
                    # Throttle updates to reduce re-rendering and improve scrolling
                    current_time = time.time()
                    if current_time - last_update_time >= update_interval or len(chunk_buffer) >= 20:
                        # Style citations in real-time during streaming (both formats)
                        styled_response = re.sub(citation_pattern_brackets, style_citation_brackets, full_response)
                        styled_response = re.sub(citation_pattern_parens, style_citation_parens, styled_response)
                        response_placeholder.markdown(styled_response + "▌", unsafe_allow_html=True)
                        chunk_buffer = ""
                        last_update_time = current_time
                
                # Final display without cursor - detect and format tables
                formatted_text, detected_tables = format_response_with_tables(full_response)
                styled_response = re.sub(citation_pattern_brackets, style_citation_brackets, formatted_text)
                styled_response = re.sub(citation_pattern_parens, style_citation_parens, styled_response)
                
                # Display response with tables
                if detected_tables:
                    # Split text by table placeholders and display with tables
                    parts = re.split(r'\[TABLE_PLACEHOLDER(?:_\d+)?\]', styled_response)
                    table_idx = 0
                    # Clear the placeholder and display content with tables
                    response_placeholder.empty()
                    for i, part in enumerate(parts):
                        if part.strip():
                            st.markdown(part, unsafe_allow_html=True)
                        if i < len(parts) - 1 and table_idx < len(detected_tables):
                            # Display table
                            st.dataframe(detected_tables[table_idx], use_container_width=True, hide_index=True)
                            table_idx += 1
                else:
                    response_placeholder.markdown(styled_response, unsafe_allow_html=True)
                
                llm_time = time.time() - llm_start
                
                # Extract citations
                citations = st.session_state.llm.extract_citations(full_response)
                
                # Combine latency metrics
                total_metrics = {
                    **retrieval_metrics,
                    "llm": llm_time,
                    "total": retrieval_metrics.get("total", 0) + llm_time
                }
                
                # Add assistant message to history (store tables for later display)
                message_id = len(st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "citations": citations,
                    "sources": retrieved_chunks,
                    "latency": total_metrics,
                    "tables": detected_tables,  # Store tables for message history
                    "id": message_id
                })
                
                # Display citations
                if citations:
                    with st.expander("📎 Citations"):
                        for citation in citations:
                            st.markdown(
                                f'<p style="color: #1f77b4; font-style: italic; margin: 0.5em 0;">'
                                f'<strong>{citation["doc_name"]}</strong>, Page {citation["page_num"]}'
                                f'</p>',
                                unsafe_allow_html=True
                            )
                
                # Display retrieval sources (only if dev mode is enabled)
                if st.session_state.show_latency_metrics:
                    with st.expander("🔍 Retrieval Sources (Dev Mode)"):
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
                
                # Display latency metrics (only if dev mode is enabled)
                if st.session_state.show_latency_metrics:
                    with st.expander("⏱️ Latency Metrics (Dev Mode)"):
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

