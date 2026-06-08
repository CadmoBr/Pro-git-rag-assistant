import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# 1. Carrega as variáveis de ambiente
load_dotenv()

# 2. LIMPEZA SEGURA (Apaga o banco antigo para forçar o RAG a ler o PDF do zero)
CHROMA_DIR = Path("data/chroma")
if CHROMA_DIR.exists():
    print("[RESET] Apagando banco de dados antigo para forçar nova indexação do PDF...")
    shutil.rmtree(CHROMA_DIR)

# 3. Importa o pipeline de execução
from src.pipeline.main_pipeline import process_user_message

print("="*50)
print("🤖 INICIANDO TESTE DO PIPELINE DE PORTFÓLIO 🤖")
print("="*50)

# Teste 1: Roteamento Conversacional
print("\n👉 Teste 1: Enviando saudação casual...")
r1 = process_user_message("Olá, tudo bem? Quem é você?")
print(f"Resposta do modelo:\n{r1['answer']}\n")

# Teste 2: Pergunta RAG Técnica
print("\n👉 Teste 2: Fazendo pergunta técnica sobre Git (Vai indexar o PDF agora)...")
pergunta_tecnica = "Segundo o livro, qual é a regra de ouro do rebase (Golden Rule of Rebasing)?"
r2 = process_user_message(pergunta_tecnica)
print(f"Resposta do modelo:\n{r2['answer']}")
print(f"Fontes utilizadas: {r2['sources']}\n")

# Teste 3: Cache Semântico
print("\n👉 Teste 3: Repetindo a pergunta técnica...")
r3 = process_user_message(pergunta_tecnica)
print(f"Resposta vinda do Cache? {r3.get('cached')}\n")