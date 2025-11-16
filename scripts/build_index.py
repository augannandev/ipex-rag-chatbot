"""
ChromaDB Index Builder
Builds and populates the persistent ChromaDB vector store from PDFs.
"""

import sys
import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import process_pdf_directory
from src.embeddings import embed_documents, get_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_index(
    pdf_dir: str = "./data/pdfs",
    chroma_db_path: str = "./data/chroma_db",
    collection_name: str = "ipex_epr_docs",
    batch_size: int = 100
):
    """
    Build ChromaDB index from PDFs.
    
    Args:
        pdf_dir: Directory containing PDF files
        chroma_db_path: Path to ChromaDB persistent storage
        collection_name: Name of the ChromaDB collection
        batch_size: Number of chunks to process in each batch
    """
    # Initialize ChromaDB client
    logger.info(f"Initializing ChromaDB client at {chroma_db_path}")
    client = chromadb.PersistentClient(path=chroma_db_path)
    
    # Get or create collection with HNSW settings
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 16,
                "description": "IPEX EPR technical documentation"
            }
        )
        logger.info(f"Collection '{collection_name}' ready")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise
    
    # Process PDFs
    logger.info(f"Processing PDFs from {pdf_dir}")
    chunks = process_pdf_directory(pdf_dir)
    
    if not chunks:
        logger.warning("No chunks to index. Please add PDF files to the pdfs directory.")
        return
    
    logger.info(f"Found {len(chunks)} chunks to index")
    
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
    logger.info("Generating embeddings...")
    embeddings = []
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_embeddings = embed_documents(batch_docs)
        embeddings.extend(batch_embeddings.tolist())
        logger.info(f"Processed {min(i + batch_size, len(documents))}/{len(documents)} chunks")
    
    # Add to ChromaDB collection
    logger.info("Adding chunks to ChromaDB collection...")
    try:
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully indexed {len(chunks)} chunks")
        
        # Print collection stats
        count = collection.count()
        logger.info(f"Collection now contains {count} documents")
        
    except Exception as e:
        logger.error(f"Failed to add documents to collection: {e}")
        raise


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ChromaDB index from PDFs")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="./data/pdfs",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--chroma-db-path",
        type=str,
        default="./data/chroma_db",
        help="Path to ChromaDB persistent storage"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="ipex_epr_docs",
        help="Name of the ChromaDB collection"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing chunks"
    )
    
    args = parser.parse_args()
    
    try:
        build_index(
            pdf_dir=args.pdf_dir,
            chroma_db_path=args.chroma_db_path,
            collection_name=args.collection_name,
            batch_size=args.batch_size
        )
        logger.info("Index building completed successfully")
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

