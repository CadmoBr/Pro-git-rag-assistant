"""Tests for HyDE retriever implementation."""

import pytest
from pathlib import Path

from src.hyde_retriever import HyDERetriever, DocumentChunk
from src.pdf_processor import process_pdf_to_chunks


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        DocumentChunk(
            content="Git is a version control system that tracks changes in files.",
            metadata={"source": "test.pdf", "chunk_index": 0},
        ),
        DocumentChunk(
            content="To create a repository, use 'git init' in an empty directory.",
            metadata={"source": "test.pdf", "chunk_index": 1},
        ),
        DocumentChunk(
            content="Branching allows you to work on features independently.",
            metadata={"source": "test.pdf", "chunk_index": 2},
        ),
        DocumentChunk(
            content="Commits save snapshots of your changes to the repository.",
            metadata={"source": "test.pdf", "chunk_index": 3},
        ),
        DocumentChunk(
            content="Merging combines changes from different branches together.",
            metadata={"source": "test.pdf", "chunk_index": 4},
        ),
    ]


def test_hyde_retriever_initialization():
    """Test HyDE retriever can be initialized."""
    retriever = HyDERetriever()
    assert retriever.embedding_model is not None
    assert len(retriever.documents) == 0
    assert retriever.index is None


def test_add_documents(sample_documents):
    """Test adding documents to retriever."""
    retriever = HyDERetriever()
    retriever.add_documents(sample_documents)

    assert len(retriever.documents) == len(sample_documents)
    assert retriever.index is not None
    assert retriever.index.shape[0] == len(sample_documents)

    # Check embeddings were computed
    for doc in retriever.documents:
        assert doc.embedding is not None
        assert len(doc.embedding) > 0


def test_fallback_query_expansion():
    """Test fallback query expansion without LLM."""
    retriever = HyDERetriever()
    query = "How to use Git branches"

    expanded = retriever._fallback_query_expansion(query, num_docs=3)

    assert len(expanded) == 3
    assert query in expanded[0]
    assert all(isinstance(doc, str) for doc in expanded)


def test_retrieve_without_hyde(sample_documents):
    """Test basic retrieval without HyDE expansion."""
    retriever = HyDERetriever()
    retriever.add_documents(sample_documents)

    results = retriever.retrieve(
        "git repository creation",
        top_k=2,
        use_hyde=False,
    )

    assert len(results) <= 2
    assert all("content" in r for r in results)
    assert all("metadata" in r for r in results)
    assert all("score" in r for r in results)


def test_retrieve_with_hyde(sample_documents):
    """Test retrieval with HyDE expansion."""
    retriever = HyDERetriever()
    retriever.add_documents(sample_documents)

    results = retriever.retrieve(
        "How do I create a new repository",
        top_k=3,
        use_hyde=True,
    )

    assert len(results) <= 3
    assert all("content" in r for r in results)
    assert all(r["score"] > 0 for r in results)


def test_save_and_load_index(sample_documents, tmp_path):
    """Test saving and loading the index."""
    # Create and save index
    retriever1 = HyDERetriever()
    retriever1.add_documents(sample_documents)

    index_path = tmp_path / "test_index.json"
    retriever1.save_index(index_path)

    assert index_path.exists()

    # Load index
    retriever2 = HyDERetriever()
    retriever2.load_index(index_path)

    assert len(retriever2.documents) == len(sample_documents)
    assert retriever2.index is not None

    # Verify loaded documents match
    for doc1, doc2 in zip(retriever1.documents, retriever2.documents):
        assert doc1.content == doc2.content
        assert doc1.metadata == doc2.metadata


@pytest.mark.skipif(
    not Path("data/corpus/progit.pdf").exists(),
    reason="Pro Git PDF not found",
)
def test_process_progit_pdf():
    """Test processing the Pro Git PDF."""
    documents = process_pdf_to_chunks(
        "data/corpus/progit.pdf",
        chunk_size=500,
        chunk_overlap=50,
    )

    assert len(documents) > 0
    assert all(isinstance(doc, DocumentChunk) for doc in documents)
    assert all(doc.metadata["source"] == "progit.pdf" for doc in documents)
    assert all(len(doc.content) > 0 for doc in documents)


@pytest.mark.skipif(
    not Path("data/corpus/progit.pdf").exists(),
    reason="Pro Git PDF not found",
)
def test_progit_retrieval():
    """Test retrieval on actual Pro Git PDF."""
    documents = process_pdf_to_chunks(
        "data/corpus/progit.pdf",
        chunk_size=1000,
        chunk_overlap=100,
    )

    retriever = HyDERetriever()
    retriever.add_documents(documents)

    # Test various queries
    queries = [
        "How do I initialize a Git repository?",
        "What are Git branches?",
        "How do I commit changes?",
    ]

    for query in queries:
        results = retriever.retrieve(query, top_k=3, use_hyde=True)
        assert len(results) > 0
        assert all(r["score"] >= 0 for r in results)
