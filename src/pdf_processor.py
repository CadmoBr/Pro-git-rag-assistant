"""PDF document processing and chunking utilities."""

from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.hyde_retriever import DocumentChunk


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(pdf_path)

    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += f"\n--- Page {page_num + 1} ---\n{page_text}"

    return text


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The text to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def process_pdf_to_chunks(
    pdf_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    source_name: Optional[str] = None,
) -> list[DocumentChunk]:
    """Process a PDF file into document chunks.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        source_name: Name of the source document

    Returns:
        List of DocumentChunk objects
    """
    pdf_path = Path(pdf_path)
    source_name = source_name or pdf_path.name

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Split into chunks
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    # Create DocumentChunk objects
    documents = [
        DocumentChunk(
            content=chunk,
            metadata={
                "source": source_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    return documents
