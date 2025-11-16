"""
PDF Ingestion Module
Handles PDF parsing, table extraction, and chunking with metadata.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import camelot
import tiktoken

logger = logging.getLogger(__name__)

# Initialize tokenizer for chunking
try:
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.warning(f"Could not load tiktoken encoding: {e}. Using fallback.")
    encoding = None


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    if encoding is None:
        # Fallback: approximate 1 token = 4 characters
        return len(text) // 4
    return len(encoding.encode(text))


def extract_product_codes(text: str) -> List[str]:
    """
    Extract product codes from text using regex patterns.
    Common patterns: alphanumeric codes, model numbers, etc.
    """
    # Pattern for common product code formats (adjust based on actual format)
    patterns = [
        r'\b[A-Z]{2,}\d{2,}[A-Z0-9]*\b',  # e.g., IPEX123, ABC1234
        r'\b\d{2,}[A-Z]{2,}\d*\b',  # e.g., 123ABC, 456XYZ789
        r'Model\s+[A-Z0-9-]+',  # e.g., Model ABC-123
        r'Part\s+[#:]?\s*[A-Z0-9-]+',  # e.g., Part #ABC123
    ]
    
    product_codes = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        product_codes.update(matches)
    
    return list(product_codes)


def extract_tables_from_pdf(pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
    """
    Extract tables from a specific page using camelot-py.
    Returns list of table dictionaries.
    """
    tables = []
    try:
        # Extract tables from the page
        camelot_tables = camelot.read_pdf(
            str(pdf_path),
            pages=str(page_num + 1),  # camelot uses 1-indexed pages
            flavor='lattice',  # or 'stream' depending on table structure
        )
        
        for table in camelot_tables:
            # Convert table to string representation
            table_df = table.df
            table_text = table_df.to_string(index=False, header=True)
            
            tables.append({
                'content': table_text,
                'content_type': 'table',
                'page_num': page_num,
                'accuracy': table.accuracy,
            })
    except Exception as e:
        logger.warning(f"Error extracting tables from page {page_num} of {pdf_path}: {e}")
        # Try stream flavor as fallback
        try:
            camelot_tables = camelot.read_pdf(
                str(pdf_path),
                pages=str(page_num + 1),
                flavor='stream',
            )
            for table in camelot_tables:
                table_df = table.df
                table_text = table_df.to_string(index=False, header=True)
                tables.append({
                    'content': table_text,
                    'content_type': 'table',
                    'page_num': page_num,
                    'accuracy': table.accuracy,
                })
        except Exception as e2:
            logger.warning(f"Fallback table extraction also failed: {e2}")
    
    return tables


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of specified token size with overlap.
    """
    if not text.strip():
        return []
    
    if encoding is None:
        # Fallback: use character-based chunking (approximate 1 token = 4 chars)
        char_chunk_size = chunk_size * 4
        char_overlap = overlap * 4
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + char_chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= text_len:
                break
            
            start += char_chunk_size - char_overlap
        
        return chunks
    
    # Token-based chunking with tiktoken
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        if end >= len(tokens):
            break
        
        # Move start forward by (chunk_size - overlap)
        start += chunk_size - overlap
    
    return chunks


def process_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Process a PDF file and return chunks with metadata.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of chunk dictionaries with metadata
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc_name = pdf_path.stem
    chunks = []
    
    try:
        # Open PDF with PyMuPDF
        doc = fitz.open(str(pdf_path))
        
        # First pass: Extract tables from each page
        page_tables = {}
        for page_num in range(len(doc)):
            tables = extract_tables_from_pdf(str(pdf_path), page_num)
            if tables:
                page_tables[page_num] = tables
        
        # Second pass: Extract text and create chunks
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Check if this page has tables
            if page_num in page_tables:
                # Add tables as separate chunks
                for table in page_tables[page_num]:
                    product_codes = extract_product_codes(table['content'])
                    chunks.append({
                        'content': table['content'],
                        'doc_name': doc_name,
                        'page_num': page_num,
                        'content_type': 'table',
                        'product_codes': product_codes,
                    })
            
            # Process text chunks (skip if page is mostly table)
            if text.strip():
                # Remove text that might be duplicated in tables
                text_chunks = chunk_text(text, chunk_size=512, overlap=50)
                
                for text_chunk in text_chunks:
                    if text_chunk.strip():
                        product_codes = extract_product_codes(text_chunk)
                        chunks.append({
                            'content': text_chunk,
                            'doc_name': doc_name,
                            'page_num': page_num,
                            'content_type': 'text',
                            'product_codes': product_codes,
                        })
        
        doc.close()
        logger.info(f"Processed {pdf_path.name}: {len(chunks)} chunks created")
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        raise
    
    return chunks


def process_pdf_directory(pdf_dir: str) -> List[Dict[str, Any]]:
    """
    Process all PDFs in a directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        List of all chunks from all PDFs
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    
    all_chunks = []
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return all_chunks
    
    for pdf_path in pdf_files:
        try:
            chunks = process_pdf(str(pdf_path))
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            continue
    
    logger.info(f"Processed {len(pdf_files)} PDFs, created {len(all_chunks)} total chunks")
    return all_chunks

