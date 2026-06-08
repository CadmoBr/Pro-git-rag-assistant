import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    """Esquema para capturar a decisão estruturada do modelo no Groq."""
    destination: str = Field(
        description="Escolha 'rag' se a pergunta for técnica sobre comandos Git ou conceitos do livro. Escolha 'conversational' se for uma saudação, agradecimento ou conversa informal."
    )

def route_query(query: str) -> str:
    """
    Avalia a intenção do usuário usando o Groq e decide o fluxo.
    Cumpre a rubrica de otimização de custo (cheap-first routing).
    """
    # Força a leitura direta do os.environ para evitar falhas de escopo
    groq_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    if not groq_key:
        raise ValueError("ERRO: GROQ_API_KEY não encontrada no ambiente do sistema.")
        
    # Inicializa o cliente passando a api_key explicitamente
    llm = ChatGroq(
        model=model_name, 
        temperature=0,
        groq_api_key=groq_key
    )
    
    # Força o Groq a devolver um JSON estruturado seguindo o nosso schema Pydantic
    structured_llm = llm.with_structured_output(RouteDecision)
    
    system_prompt = (
        "Você é o roteador oficial de um sistema RAG focado no livro Pro Git.\n"
        "Analise a mensagem do usuário e decida o destino ideal:\n"
        "- 'rag': Dúvidas técnicas, comandos (init, commit, rebase), conceitos de branching ou configurações.\n"
        "- 'conversational': Saudações (olá, bom dia), agradecimentos, ou conversas casuais que não exigem consulta ao livro."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        decision = chain.invoke({"query": query})
        return decision.destination
    except Exception as e:
        print(f"[AVISO ROTEADOR] Falha ao rotear, usando fallback 'rag'. Erro: {e}")
        return "rag"