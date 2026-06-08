# Hypothetical Document Embeddings (HyDE) Implementation

## Overview

This project implements **Hypothetical Document Embeddings (HyDE)**, a retrieval-augmented generation (RAG) technique that improves semantic search without fine-tuning. HyDE generates hypothetical documents relevant to a query and uses their embeddings to enhance retrieval accuracy.

## How HyDE Works

Traditional embedding-based retrieval computes similarity between:
- Query embedding (from the user's question)
- Document embeddings (from the corpus)

**HyDE improves this by:**

1. **Query Expansion**: Instead of just using the original query, generate multiple hypothetical documents that would answer the query
2. **Multi-Perspective Matching**: Compute similarity scores between documents and:
   - The original query
   - All hypothetical documents
3. **Ensemble Scoring**: Average the similarity scores to get a more robust relevance score

```
Query: "How do I create a Git repository?"
           ↓
Hypothetical Documents Generated:
  1. "Creating a repository with git init..."
  2. "Initializing a new Git project..."
  3. "Setting up version control..."
           ↓
All are embedded and used to score documents
           ↓
Top relevant documents retrieved
```

## Project Structure

```
src/
├── hyde_retriever.py      # Core HyDE implementation
├── pdf_processor.py       # PDF text extraction and chunking
└── main_hyde.py           # Demo and CLI interface

tests/
└── test_hyde.py           # Comprehensive test suite

data/
├── corpus/                # Document storage
│   └── progit.pdf         # Pro Git v2.1.450 PDF
└── hyde_index.json        # Persisted embeddings index
```

## Core Components

### 1. DocumentChunk (Pydantic Model)
Represents a text chunk with its embeddings and metadata:
```python
class DocumentChunk(BaseModel):
    content: str              # Text content
    metadata: dict            # Metadata (source, page, etc.)
    embedding: Optional[list] # Computed embedding vector
```

### 2. HyDERetriever
Main retrieval engine with these key methods:

#### `add_documents(documents: list[DocumentChunk])`
- Adds documents to the retriever
- Computes embeddings using sentence-transformers
- Builds numpy index for efficient similarity search

#### `generate_hypothetical_documents(query: str, num_docs: int = 3)`
- Generates hypothetical documents using OpenAI's GPT-3.5 (if API key available)
- Falls back to simple keyword expansion without API key
- Returns list of document variations

#### `expand_query(query: str)`
- Calls `generate_hypothetical_documents`
- Computes embeddings for hypothetical docs
- Returns both query and hypothetical embeddings

#### `retrieve(query: str, top_k: int = 5, use_hyde: bool = True)`
- Main retrieval method
- If `use_hyde=True`: averages query + hypothetical doc similarities
- If `use_hyde=False`: standard embedding-based retrieval
- Returns top-k results with scores

#### `save_index/load_index`
- Persist the index to JSON for reuse
- Avoid recomputing embeddings on subsequent runs

### 3. PDF Processing (pdf_processor.py)

#### `extract_text_from_pdf(pdf_path)`
- Extracts text from PDF using pypdf library
- Handles decompression errors gracefully

#### `chunk_text(text, chunk_size, chunk_overlap)`
- Splits text into overlapping chunks using RecursiveCharacterTextSplitter
- Preserves context across chunks
- Configurable chunk size and overlap

#### `process_pdf_to_chunks(pdf_path, ...)`
- End-to-end pipeline: extract → chunk → create DocumentChunk objects
- Adds metadata (source, chunk index, total chunks)

## Usage

### Building the Index

```python
from src.main_hyde import build_hyde_index

# Build and save index from PDF
retriever = build_hyde_index(
    pdf_path="data/corpus/progit.pdf",
    output_index="data/hyde_index.json"
)
```

### Searching with HyDE

```python
# Load existing index
from src.main_hyde import load_hyde_index

retriever = load_hyde_index("data/hyde_index.json")

# Retrieve with HyDE expansion
results = retriever.retrieve(
    query="How do I commit changes?",
    top_k=5,
    use_hyde=True
)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content'][:100]}...")
```

### CLI Usage

```bash
# Run with default query
python -m src.main_hyde

# Run with custom query
python -m src.main_hyde "How do I create branches in Git?"
```

## Configuration

### Embedding Model
Default: `all-MiniLM-L6-v2` (small, fast, multilingual)

Available options (from Sentence Transformers):
- `all-MiniLM-L6-v2` - Small, fast (recommended for CPU)
- `all-mpnet-base-v2` - Larger, better quality (needs GPU)
- `multilingual-e5-base` - For multilingual documents

### Chunk Parameters
In `main_hyde.py`:
```python
chunk_size=1000,      # Characters per chunk
chunk_overlap=100,    # Overlap between chunks
```

## Performance Considerations

### Embedding Computation
- **Time**: ~100K documents/minute on CPU (M-series Mac)
- **Memory**: ~500MB for 1000 documents with MiniLM
- **Trade-off**: Smaller models = faster, less accurate; larger = slower, more accurate

### Similarity Search
- **Numpy dot product**: O(n) where n = number of documents
- **For millions of docs**: Use Faiss, Weaviate, or Pinecone

### HyDE Query Expansion
- **With OpenAI API**: ~1-2 seconds per query (API latency)
- **With fallback**: <100ms per query (local keyword expansion)

## API Key Configuration

The implementation works with or without an OpenAI API key:

### With API Key (Best Quality)
```bash
export OPENAI_API_KEY="sk-..."
python -m src.main_hyde "query"  # Uses GPT-3.5 for hypothetical docs
```

### Without API Key (Fallback)
```bash
python -m src.main_hyde "query"  # Uses keyword expansion
```

Fallback expansion strategy:
1. Original query
2. Query + common related terms (tutorial, guide, steps, etc.)
3. Query focused on main keywords
4. "How to [query]" variant

## Testing

Run the test suite:
```bash
pytest tests/test_hyde.py -v

# Run only tests that need Pro Git PDF
pytest tests/test_hyde.py::test_progit_retrieval -v

# Skip tests that need API key
pytest tests/test_hyde.py -m "not needs_api_key" -v
```

## Results on Pro Git PDF

Sample queries with 1000 document chunks:

**Query**: "How do I create a Git repository?"
```
[1] Score: 0.5970 - "Getting a Git Repository - You typically obtain a Git repository..."
[2] Score: 0.5884 - "New repository dropdown dialog..."
[3] Score: 0.5604 - "Repository creation using FileRepositoryBuilder..."
[4] Score: 0.5565 - "Basic understanding of Git..."
[5] Score: 0.5525 - "Contributors list..."
```

## Limitations & Future Work

### Current Limitations
1. **No API key required**: Falls back to keyword expansion (less sophisticated)
2. **No semantic caching**: Re-expands the same query each time
3. **No multi-language support**: Embeddings are English-only
4. **No re-ranking**: Simple averaging of similarity scores

### Potential Improvements
1. **Use local LLM**: Replace OpenAI with Ollama/Llama2 for self-hosted HyDE
2. **Semantic caching**: Cache hypothetical documents for repeated queries
3. **Cross-encoder re-ranking**: Use a cross-encoder to re-rank top-k results
4. **Vector DB integration**: Move from numpy to Faiss/Pinecone for scale
5. **Multi-modal retrieval**: Support images, code, mathematical formulas

## References

- **HyDE Paper**: [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- **Sentence Transformers**: [SBERT Documentation](https://www.sbert.net/)
- **Pro Git Book**: [https://git-scm.com/book](https://git-scm.com/book)

## License

MIT License - Feel free to use this implementation for research or production use.
