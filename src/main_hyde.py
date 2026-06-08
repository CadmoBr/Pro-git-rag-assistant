"""Main script to build and query the HyDE-based retrieval system."""

import json
from pathlib import Path

from src.hyde_retriever import HyDERetriever
from src.pdf_processor import process_pdf_to_chunks


def build_hyde_index(
    pdf_path: str = "data/corpus/progit.pdf",
    output_index: str = "data/hyde_index.json",
) -> HyDERetriever:
    """Build a HyDE index from a PDF file.

    Args:
        pdf_path: Path to the PDF file
        output_index: Path to save the index

    Returns:
        Initialized HyDERetriever with documents
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Processing PDF: {pdf_path}")
    documents = process_pdf_to_chunks(
        pdf_path,
        chunk_size=1000,
        chunk_overlap=100,
    )
    print(f"Created {len(documents)} document chunks")

    print("Initializing HyDE retriever...")
    retriever = HyDERetriever()

    print("Computing embeddings for all documents...")
    retriever.add_documents(documents)

    print(f"Saving index to {output_index}")
    Path(output_index).parent.mkdir(parents=True, exist_ok=True)
    retriever.save_index(output_index)

    return retriever


def load_hyde_index(index_path: str = "data/hyde_index.json") -> HyDERetriever:
    """Load a previously built HyDE index.

    Args:
        index_path: Path to the saved index

    Returns:
        Initialized HyDERetriever with loaded documents
    """
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    print(f"Loading index from {index_path}")
    retriever = HyDERetriever()
    retriever.load_index(index_path)
    print(f"Loaded {len(retriever.documents)} documents")

    return retriever


def demo_search(
    query: str = "How do I commit changes in Git?",
    use_hyde: bool = True,
) -> None:
    """Run a demo search using the HyDE retriever.

    Args:
        query: The search query
        use_hyde: Whether to use HyDE expansion
    """
    # Try to load existing index, build if not found
    index_path = Path("data/hyde_index.json")
    if index_path.exists():
        retriever = load_hyde_index(index_path)
    else:
        retriever = build_hyde_index()

    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"Using HyDE: {use_hyde}")
    print(f"{'='*80}\n")

    # Perform retrieval
    results = retriever.retrieve(query, top_k=5, use_hyde=use_hyde)

    print(f"Top 5 results:\n")
    for i, result in enumerate(results, 1):
        print(f"[{i}] Score: {result['score']:.4f}")
        print(f"    Source: {result['metadata']['source']}")
        print(f"    Content: {result['content'][:200]}...")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        demo_search(query)
    else:
        # Run demo with default query
        demo_search()
