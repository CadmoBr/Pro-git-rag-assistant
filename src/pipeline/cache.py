import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

class SemanticCache:
    """
    Cache semântico local em memória usando HuggingFace Embeddings.
    Garante taxa zero de chamadas repetidas ao Groq se a pergunta for similar.
    """
    def __init__(self, threshold: float = 0.85):
        self.storage = {}  # Mapeia vetor_da_pergunta -> resposta
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.threshold = threshold  # Limiar de similaridade ajustado para o modelo open-source

    def _cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
        return dot_product / (norm_vec1 * norm_vec2)

    def lookup(self, query: str) -> str | None:
        """Verifica se existe alguma pergunta parecida no cache."""
        if not self.storage:
            return None
            
        query_vector = self.embeddings.embed_query(query)
        
        for cached_vector, cached_response in self.storage.items():
            similarity = self._cosine_similarity(query_vector, np.array(cached_vector))
            if similarity >= self.threshold:
                return cached_response
                
        return None

    def update(self, query: str, response: str):
        """Salva a nova pergunta e resposta no cache."""
        query_vector = self.embeddings.embed_query(query)
        self.storage[tuple(query_vector)] = response