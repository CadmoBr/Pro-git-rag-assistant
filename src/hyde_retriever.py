"""Hypothetical Document Embeddings (HyDE) retriever for semantic search."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """A chunk of text from a document."""

    content: str
    metadata: dict
    embedding: Optional[list] = None


class HyDERetriever:
    """Implements Hypothetical Document Embeddings (HyDE) retrieval.

    HyDE generates hypothetical documents relevant to a query and uses
    their embeddings to improve retrieval without fine-tuning.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        llm_client: Optional[OpenAI] = None,
    ):
        """Initialize the HyDE retriever.

        Args:
            model_name: Name of the sentence transformer model
            llm_client: OpenAI client for generating hypothetical docs
        """
        self.embedding_model = SentenceTransformer(model_name)

        # Try to initialize LLM client, but don't fail if API key is missing
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            try:
                self.llm_client = OpenAI()
            except Exception:
                print("Warning: OpenAI API key not found. Using fallback query expansion.")
                self.llm_client = None

        self.documents: list[DocumentChunk] = []
        self.index: np.ndarray | None = None

    def add_documents(self, documents: list[DocumentChunk]) -> None:
        """Add documents and compute their embeddings.

        Args:
            documents: List of document chunks to add
        """
        self.documents.extend(documents)

        # Compute embeddings for all documents
        texts = [doc.content for doc in self.documents]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)

        # Store embeddings
        for doc, embedding in zip(self.documents, embeddings):
            doc.embedding = embedding.tolist()

        # Create numpy index for fast similarity search
        self.index = np.array([doc.embedding for doc in self.documents])

    def generate_hypothetical_documents(
        self, query: str, num_docs: int = 3
    ) -> list[str]:
        """Generate hypothetical documents relevant to the query using LLM.

        Args:
            query: The search query
            num_docs: Number of hypothetical documents to generate

        Returns:
            List of hypothetical document texts
        """
        if not self.llm_client:
            # Fallback: simple keyword expansion without LLM
            return self._fallback_query_expansion(query, num_docs)

        try:
            prompt = f"""Generate {num_docs} plausible and diverse hypothetical documents that would be 
relevant to the following search query. Each document should be 2-3 sentences and realistically 
correspond to content that might appear in a technical book or documentation.

Query: "{query}"

Return only the hypothetical documents, one per line, numbered 1-{num_docs}:"""

            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )

            # Parse the response to extract documents
            text = response.choices[0].message.content
            docs = []
            for line in text.split("\n"):
                line = line.strip()
                if line and not line[0].isdigit():
                    docs.append(line)
            return docs[:num_docs]
        except Exception as e:
            print(f"Warning: LLM generation failed ({e}), using fallback")
            return self._fallback_query_expansion(query, num_docs)

    def _fallback_query_expansion(self, query: str, num_docs: int = 3) -> list[str]:
        """Fallback query expansion without LLM.

        Uses simple heuristics to expand the query.

        Args:
            query: The search query
            num_docs: Number of documents to generate

        Returns:
            List of expanded query variations
        """
        words = query.lower().split()
        variations = [query]

        # Variation 1: Add common related terms
        expanded = query + " tutorial steps procedure method"
        variations.append(expanded)

        # Variation 2: Focus on main keywords
        if len(words) > 1:
            main_terms = " ".join(words[:min(3, len(words))])
            variations.append(f"{main_terms} guide documentation example")

        # Variation 3: Add common patterns
        variations.append(f"How to {query}")

        return variations[:num_docs]

    def expand_query(self, query: str) -> tuple[list[np.ndarray], np.ndarray]:
        """Expand query using HyDE approach.

        Generates hypothetical documents and computes their embeddings.

        Args:
            query: The search query

        Returns:
            Tuple of (hypothetical doc embeddings, original query embedding)
        """
        # Get original query embedding
        query_embedding = self.embedding_model.encode(
            query, convert_to_numpy=True
        )

        # Generate hypothetical documents
        hyp_docs = self.generate_hypothetical_documents(query)

        # Compute embeddings for hypothetical documents
        hyp_embeddings = self.embedding_model.encode(hyp_docs, convert_to_numpy=True)

        return hyp_embeddings, query_embedding

    def retrieve(
        self, query: str, top_k: int = 5, use_hyde: bool = True
    ) -> list[dict]:
        """Retrieve relevant documents using optional HyDE expansion.

        Args:
            query: The search query
            top_k: Number of top documents to retrieve
            use_hyde: Whether to use HyDE query expansion

        Returns:
            List of retrieved documents with scores
        """
        if self.index is None:
            return []

        if use_hyde:
            hyp_embeddings, query_embedding = self.expand_query(query)

            # Compute similarities for query and hypothetical docs
            query_sims = np.dot(self.index, query_embedding)
            hyp_sims = np.dot(self.index, hyp_embeddings.T)

            # Average the similarities (giving more weight to hypothetical docs)
            avg_sims = (query_sims + np.mean(hyp_sims, axis=1)) / 2
        else:
            # Standard embedding-based retrieval
            query_embedding = self.embedding_model.encode(
                query, convert_to_numpy=True
            )
            avg_sims = np.dot(self.index, query_embedding)

        # Get top-k results
        top_indices = np.argsort(avg_sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "content": doc.content,
                "metadata": doc.metadata,
                "score": float(avg_sims[idx]),
            })

        return results

    def save_index(self, filepath: str | Path) -> None:
        """Save the document index to disk.

        Args:
            filepath: Path to save the index
        """
        data = {
            "documents": [
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "embedding": doc.embedding,
                }
                for doc in self.documents
            ]
        }
        Path(filepath).write_text(json.dumps(data))

    def load_index(self, filepath: str | Path) -> None:
        """Load the document index from disk.

        Args:
            filepath: Path to load the index from
        """
        data = json.loads(Path(filepath).read_text())
        self.documents = [
            DocumentChunk(
                content=doc["content"],
                metadata=doc["metadata"],
                embedding=doc["embedding"],
            )
            for doc in data["documents"]
        ]
        self.index = np.array([doc.embedding for doc in self.documents])
