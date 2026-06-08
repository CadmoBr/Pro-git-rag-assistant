import os
from src.pipeline.rag import build_rag_pipeline
from src.pipeline.cache import SemanticCache
from src.pipeline.routing import route_query
from src.pipeline.tools import get_tools

# Inicializa o cache semântico global na memória
cache_sistema = SemanticCache()

def process_user_message(query: str) -> dict:
    """
    Orquestra o fluxo de execução completo cumprindo a rubrica de Custo e Latência:
    1. Verifica o Cache Semântico (Custo $0, Latência mínima)
    2. Aplica o Roteador Cheap-First (Manda saudações direto pro LLM sem RAG)
    3. Aciona o RAG Pipeline se a dúvida for técnica sobre Git
    """
    # 1. Checagem de Cache Semântico
    resposta_em_cache = cache_sistema.lookup(query)
    if resposta_em_cache:
        print("[CACHE HIT] Resposta recuperada do Cache Semântico local.")
        return {
            "answer": resposta_em_cache,
            "sources": [("Cache Local", 0)],
            "cached": True
        }
    
    # 2. Decisão do Roteador (Cheap-First Routing)
    rota = route_query(query)
    print(f"[ROTEADOR] Query direcionada para a rota: {rota.upper()}")
    
    # Instancia o RAG (ele vai ler o PDF automaticamente se o Chroma estiver vazio)
    rag = build_rag_pipeline()
    
    # Rota Conversacional (Bate-papo rápido ou saudação)
    if rota == "conversational":
        # Consome o Groq diretamente com pouquíssimos tokens, sem injetar o livro
        response = rag.client.chat.completions.create(
            model=rag.llm_model,
            messages=[
                {"role": "system", "content": "Você é um assistente simpático especializado no livro Pro Git. Diga olá ou responda de forma breve à interação social do usuário."},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        answer = response.choices[0].message.content or ""
        # Atualiza o cache para interações futuras
        cache_sistema.update(query, answer)
        return {"answer": answer, "sources": [], "cached": False}
    
    # Rota RAG (Dúvida técnica densa sobre Git)
    try:
        resultado_rag = rag.answer(query, k=4)
        
        # Atualiza o cache semântico com a resposta robusta do RAG
        cache_sistema.update(query, resultado_rag["answer"])
        
        resultado_rag["cached"] = False
        return resultado_rag
        
    except Exception as e:
        print(f"[ERRO PIPELINE] Falha na execução do RAG: {e}")
        return {
            "answer": "Desculpe, ocorreu um erro interno ao processar sua dúvida sobre o livro técnico.",
            "sources": [],
            "cached": False
        }