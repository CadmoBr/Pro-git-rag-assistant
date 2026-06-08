"""RAG pipeline — chunk, embed, index, retrieve, generate.

Reaproveita as funcoes do notebook 02. Voce vai preencher 3 TODOs aqui.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import uuid

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Usamos o cliente nativo do Groq para banir o erro de autenticação (401)
from groq import Groq


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local e inferência via Groq."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        # Recupera a chave do Groq direto do .env carregado
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError("Configure GROQ_API_KEY no seu arquivo .env")
            
        # Inicializa o cliente oficial do Groq de forma direta e limpa
        self.client = Groq(api_key=groq_key)
        
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
        self.embed_model = embed_model or os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        # Garante o uso do modelo local rodando em CPU (Custo $0 tokens)
        self.embed_fn = SentenceTransformerEmbeddingFunction(model_name=self.embed_model)

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    # ------------------------------------------------------------------ TODO 1
    def ingest_and_index(self) -> int:
        """Le PDFs de `corpus_dir`, faz chunking e indexa em Chroma.

        Retorna numero de chunks indexados.
        """
        docs: list[dict] = []
        pdf_files = list(self.corpus_dir.glob("*.pdf"))
        
        for pdf_path in pdf_files:
            try:
                reader = PdfReader(pdf_path)
                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        docs.append({
                            "text": text,
                            "source": pdf_path.name,
                            "page": page_idx + 1
                        })
            except Exception as e:
                print(f"Erro ao ler o arquivo {pdf_path.name}: {e}")

        if not docs:
            return 0

        # Chunking Recursivo Técnico (800 caracteres / 100 overlap)
        chunks: list[dict] = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )

        for doc in docs:
            split_texts = text_splitter.split_text(doc["text"])
            for text_chunk in split_texts:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": text_chunk,
                    "source": doc["source"],
                    "page": doc["page"]
                })

        # Adicionar os chunks gerados no banco de dados Chroma
        if chunks:
            ids = [c["id"] for c in chunks]
            documents = [c["text"] for c in chunks]
            metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
            
            # Divide em lotes de 400 para evitar estourar o limite de batch do Chroma
            batch_size = 400
            for i in range(0, len(chunks), batch_size):
                self.collection.add(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )

        return self.collection.count()

    # ------------------------------------------------------------------ TODO 2
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Busca top-k chunks similares a query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        hits = []
        if results and results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)
            
            for doc, meta, dist in zip(documents, metadatas, distances):
                hits.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 0),
                    "distance": dist
                })
        
        return hits

    # ------------------------------------------------------------------ TODO 3
    def answer(self, question: str, k: int = 5) -> dict:
        """Pipeline completo: retrieve + augment + generate. Retorna {answer, sources}."""
        hits = self.retrieve(question, k=k)

        # 1. Montar o contexto injetando as referências estruturadas exigidas pela rubrica
        context_blocks = []
        for h in hits:
            block = f"[{h['source']}:{h['page']}]\n{h['text']}"
            context_blocks.append(block)
        
        context_str = "\n\n---\n\n".join(context_blocks)

        # 2. Construir o prompt a partir do template padrão
        prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)

        # 3. Chamar a API do Groq configurada no cliente nativo
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        
        answer_text = response.choices[0].message.content or "Nao encontrado no corpus."

        # 4. Retornar a resposta e a lista de fontes mapeadas
        return {
            "answer": answer_text,
            "sources": [(h["source"], h["page"]) for h in hits]
        }


PROMPT_TEMPLATE = """Você é um engenheiro especialista em Git e assistente técnico do livro Pro Git.
Responda à pergunta do usuário de forma clara e detalhada baseando-se nos trechos do livro fornecidos no contexto abaixo.

Nota de tradução: O livro está em inglês. Se o usuário perguntar por "regra de ouro do rebase", procure pelo conceito de "Do not rebase commits that exist outside your repository" (Golden Rule of Rebasing).

Se a informação não puder ser deduzida ou encontrada de nenhuma forma nos trechos abaixo, responda estritamente: "Nao encontrado no corpus."
Sempre cite o arquivo e a página de onde extraiu a informação usando o formato [arquivo:pagina].

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Factory: cria pipeline e indexa corpus se ainda nao indexado."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()
    return pipeline