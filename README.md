# IPEX RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot built with Streamlit for querying technical PDF documents. Features hybrid retrieval (dense vectors + BM25 keyword search), cross-encoder reranking, and streaming responses using Claude 3 Haiku.

## Features

- **Hybrid Retrieval**: Combines dense vector search (ChromaDB) with BM25 keyword search
- **Reranking**: Uses cross-encoder model for improved relevance
- **Streaming Responses**: Real-time streaming from Claude 3 Haiku
- **Citation Support**: Automatic citation extraction and display
- **Table Extraction**: Preserves tables from PDFs using camelot-py
- **Query Routing**: Intelligent query classification (traditional RAG ready, agentic RAG structure prepared)
- **Performance Optimized**: Targets 1-2 second response times

## Architecture

- **Vector Store**: ChromaDB with HNSW indexing
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Reranking**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **LLM**: Claude 3 Haiku (Anthropic)
- **PDF Processing**: PyMuPDF + camelot-py

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ipex-rag-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or create a `.streamlit/secrets.toml` file:
```toml
ANTHROPIC_API_KEY = "your-api-key-here"
```

## Usage

### 1. Add PDF Documents

Place your PDF files in the `data/pdfs/` directory:
```bash
mkdir -p data/pdfs
cp your-documents.pdf data/pdfs/
```

### 2. Build the Index

Run the index builder script to process PDFs and create the vector store:
```bash
python scripts/build_index.py
```

Options:
```bash
python scripts/build_index.py --pdf-dir ./data/pdfs --chroma-db-path ./data/chroma_db --collection-name ipex_epr_docs
```

### 3. Run the Streamlit App

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`.

## Project Structure

```
app/
├── app.py                    # Main Streamlit app
├── requirements.txt
├── .gitignore
├── data/
│   ├── pdfs/                # Source PDFs
│   └── chroma_db/           # Vector store (persistent)
├── src/
│   ├── ingestion.py         # PDF parsing + chunking
│   ├── embeddings.py        # Embedding functions
│   ├── retrieval.py         # Hybrid retrieval + reranking
│   ├── query_router.py      # Route simple vs complex queries
│   └── llm.py              # Claude API with streaming
└── scripts/
    └── build_index.py       # Build ChromaDB index
```

## Configuration

### ChromaDB Settings

The collection is created with HNSW settings optimized for cosine similarity:
- `hnsw:space`: "cosine"
- `hnsw:M`: 16

### Chunking Strategy

- **Tables**: Entire table kept as single chunk
- **Text**: 512 tokens with 50 token overlap
- **Metadata**: Includes doc_name, page_num, content_type, product_codes

### Retrieval Pipeline

1. Metadata pre-filtering (if product codes detected)
2. Parallel dense search (ChromaDB) + BM25 search
3. Reciprocal Rank Fusion (combine results)
4. Cross-encoder reranking (top 20 → top 5)
5. Return top 5 chunks with metadata

## Performance Targets

- Median latency: 1.2s
- p95 latency: 2.5s
- Retrieval accuracy: >90%
- Answer accuracy: >90%

## Troubleshooting

### ChromaDB Collection Not Found

If you see "ChromaDB collection not found", make sure you've run `build_index.py` first.

### API Key Issues

Ensure your Anthropic API key is set either:
- As environment variable: `ANTHROPIC_API_KEY`
- In Streamlit secrets: `.streamlit/secrets.toml`
- Via sidebar input in the app

### Table Extraction Fails

If camelot-py fails to extract tables, it will fall back to stream flavor. Ensure you have the required dependencies:
```bash
pip install camelot-py[cv]
```

### Model Loading Issues

Models are downloaded automatically on first use. Ensure you have internet connectivity and sufficient disk space (~500MB for all models).

## Development

### Adding New Features

- **Agentic RAG**: The query router is prepared for agentic RAG. Implement tools in `src/query_router.py` and add agent logic.
- **Custom Embeddings**: Modify `src/embeddings.py` to use different models.
- **Additional Retrievers**: Extend `src/retrieval.py` with new retrieval strategies.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

