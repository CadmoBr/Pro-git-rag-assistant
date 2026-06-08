"""Streamlit UI — Interface Premium para o Portfólio de Engenharia de IA.

Conectada ao pipeline híbrido avançado com suporte a HyDE e Cache Semântico.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Configuração de caminhos do sistema
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv()

from src.pipeline.main_pipeline import process_user_message, cache_sistema
from src.pipeline.rag import build_rag_pipeline

# ---------------------------------------------------------------- Page Configuration
st.set_page_config(
    page_title="Pro Git AI Expert", 
    page_icon="⚡", 
    layout="wide", # Layout expandido para melhor aproveitamento de tela
    initial_sidebar_state="expanded"
)

# Injeção de CSS customizado para polimento de fontes e espaçamento (UX sênior)
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stChatMessage { border-radius: 8px; margin-bottom: 10px; }
        div[data-testid="stMetricBackground"] { background-color: #f0f2f6; padding: 10px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# Inicialização cacheada do núcleo do RAG
@st.cache_resource
def init_rag_core():
    return build_rag_pipeline()

with st.spinner("Carregando inteligência local e base vetorial..."):
    rag_core = init_rag_core()

# ---------------------------------------------------------------- SIDEBAR (Painel Clínico/Métricas)
with st.sidebar:
    st.image("https://git-scm.com/images/logos/downloads/Git-Logo-2Color.png", width=120)
    st.title("Painel de Controle")
    st.caption("Monitoramento de Recursos e Custos da Aplicação em tempo real.")
    
    st.markdown("---")
    st.subheader("📊 Volumetria & Cache")
    
    # Organização das métricas em cards elegantes
    st.metric(label="Camadas de Texto Indexadas", value=f"{rag_core.collection.count()} chunks")
    st.metric(label="Memória Semântica Ativa", value=f"{len(cache_sistema.storage)} respostas")
    
    st.markdown("---")
    st.subheader("⚙️ Stack Tecnológica")
    st.markdown("""
    - **LLM Core:** Llama 3.3-70b via **Groq Cloud**
    - **Embeddings:** MiniLM-L6-v2 via **HuggingFace**
    - **Orquestração:** Advanced RAG com técnica **HyDE**
    - **Vector DB:** ChromaDB Local Pro
    """)
    
    st.markdown("---")
    if st.button("🧹 Limpar Memória de Cache", use_container_width=True):
        cache_sistema.storage.clear()
        st.toast("O cache do sistema foi esvaziado com sucesso!")
        st.rerun()

# ---------------------------------------------------------------- MAIN AREA (Layout por Abas)
st.title("⚡ Pro Git Expert Assistant")
st.markdown(
    "Interface inteligente para auditoria e consulta do livro oficial **Pro Git**. "
    "Equipado com expansão de consultas hipotéticas (HyDE) para alta resiliência semântica."
)

tab_chat, tab_docs, tab_arch = st.tabs(["💬 Chat Especialista", "📘 Base de Documentos", "📐 Engenharia do Projeto"])

# --- ABA 1: INTERFACE DE CHAT ---
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Olá! Sou o seu auditor do ecossistema Git. Como posso acelerar o seu workflow técnico hoje?"}
        ]

    # Renderiza o histórico com design polido
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown("---")
                for src, pg in msg["sources"]:
                    st.caption(f"📖 **Referência:** Livro `{src}` | Seção/Página: `{pg}`")

    # Entrada do chat
    if user_query := st.chat_input("Ex: Qual o perigo de fazer rebase em branches públicos?"):
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Executando alinhamento semântico via HyDE..."):
                try:
                    result = process_user_message(user_query)
                    
                    # Notificação visual discreta e elegante de otimização de custo
                    if result.get("cached"):
                        st.info("⚡ *Insights recuperados instantaneamente via Cache Semântico Local ($0 tokens consumidos)*")
                    
                    st.markdown(result["answer"])
                    
                    sources_list = result.get("sources", [])
                    if sources_list:
                        st.markdown("---")
                        for src, pg in sources_list:
                            st.caption(f"📖 **Referência Mapeada:** Arquivo `{src}` | Página `{pg}`")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": sources_list
                    })
                    
                except Exception as e:
                    st.error(f"Falha na esteira de processamento: {e}")

# --- ABA 2: DETALHES DA BASE DE CONHECIMENTO ---
with tab_docs:
    st.subheader("Livros e Papers Ativos na Base Vetorial")
    st.markdown("""
    O assistente está treinado e indexado estritamente sobre a seguinte documentação oficial:
    """)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://git-scm.com/assets/progit_v2.lca552.png", width=140)
    with col2:
        st.markdown("### **Pro Git (Second Edition)**")
        st.markdown("""
        * **Autores:** Scott Chacon e Ben Straub
        * **Licença:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0
        * **Escopo Técnico:** Fundamentos do Git, Ramificação (Branching), Git no Servidor, Customização e Engenharia Interna (Git Internals).
        * **Status do Ingest:** Completamente mapeado, quebrado em parágrafos estruturados e vetorizado em CPU local.
        """)

# --- ABA 3: ENGENHARIA E DESIGN DE ARQUITETURA ---
with tab_arch:
    st.subheader("Arquitetura Avançada do RAG Híbrido")
    st.markdown("""
    Esta aplicação demonstra um pipeline de Engenharia de IA focado em **máxima acurácia** com **mínimo custo operacional**:
    
    1. **Camada Cheap-First (Routing):** Consultas casuais ou saudações são interceptadas antes de ativar o pipeline RAG, economizando computação vetorial.
    2. **Expansão de Query via HyDE:** O sistema gera uma resposta hipotética rica em jargões para garantir que a busca semântica no ChromaDB encontre o documento mesmo se você digitar termos com erros ou em português.
    3. **Embeddings Locais:** Toda a matemática vetorial roda localmente na CPU da máquina usando `all-MiniLM-L6-v2`, garantindo dependência zero de APIs pagas para indexação.
    4. **Cache Semântico:** Respostas equivalentes são bloqueadas e resolvidas localmente, permitindo escala infinita com custo zero de API na nuvem.
    """)