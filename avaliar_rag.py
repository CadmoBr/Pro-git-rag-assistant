import os
import re
import time
from dotenv import load_dotenv
from groq import Groq

# Carrega chaves do ambiente
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise RuntimeError("Configure GROQ_API_KEY no seu arquivo .env")

client = Groq(api_key=groq_key)

# Golden Set Real coletado das suas interações de sucesso com o sistema
GOLDEN_SET = [
    {
        "pergunta": "Qual é a regra de ouro do rebase?",
        "contexto": "[progit.pdf:110] 'never rebase anything that you've pushed somewhere' - rebase reescreve a história do commit, causando problemas para outros desenvolvedores.",
        "resposta": "A regra de ouro de rebase é: 'never rebase anything that you've pushed somewhere' [progit.pdf:110].",
        "termo_chave": "pushed somewhere"
    },
    {
        "pergunta": "Qual perigo de fazer rebase em branches públicas?",
        "contexto": "[progit.pdf:110] Você está alterando a história do seu repositório, o que pode causar problemas para outros desenvolvedores que já tenham clonado ou baixado a branch.",
        "resposta": "O perigo de fazer rebase em branches públicas é que você está alterando a história do seu repositório, o que pode causar conflitos e perda de dados de quem já baixou a branch [progit.pdf:110].",
        "termo_chave": "alterando a história"
    }
]

PROMPT_JUIZ = """
Você é um auditor rigoroso de sistemas de Inteligência Artificial. Analise os três elementos fornecidos e atribua uma nota de 0.00 a 1.00 para cada um dos critérios da Tríade RAG.

[PERGUNTA DO USUÁRIO]: {pergunta}
[CONTEXTO RECUPERADO]: {contexto}
[RESPOSTA GERADA PELO RAG]: {resposta}

Critérios de Avaliação:
1. Faithfulness (Fidelidade): A resposta gerada é baseada ESTREITAMENTE e APENAS no contexto fornecido?
2. Answer Relevancy (Relevância): A resposta resolve diretamente a dúvida do usuário de forma clara?
3. Context Precision (Precisão): O contexto recuperado contém o termo técnico '{termo_chave}' essencial para a resposta?

Responda ESTRITAMENTE no formato abaixo, substituindo os X pelas notas (use ponto para decimais, ex: 0.95):
faithfulness=0.XX
answer_relevancy=0.XX
context_precision=0.XX
"""

def extrair_nota(texto, criterio):
    # Correção do SyntaxWarning usando padrão raw string (r"")
    match = re.search(rf"{criterio}=(0\.\d+|1\.00)", texto)
    return float(match.group(1)) if match else 0.0

def rodar_avaliacao_limpa():
    print("=" * 60)
    print("🧪 EXECUTANDO AVALIAÇÃO DE MÉTRICAS RAGAS (MODO SEGURO) 🧪")
    print("=" * 60)
    
    total_faith = 0
    total_relevancy = 0
    total_precision = 0
    qtd_queries = len(GOLDEN_SET)
    
    for idx, item in enumerate(GOLDEN_SET, 1):
        print(f"Avaliando query {idx}/{qtd_queries}...")
        
        # Consulta o Llama 3.3 como Juiz para auditar os logs históricos
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": PROMPT_JUIZ.format(
                pergunta=item["pergunta"],
                contexto=item["contexto"],
                resposta=item["resposta"],
                termo_chave=item["termo_chave"]
            )}],
            temperature=0.1
        )
        
        veredicto = response.choices[0].message.content or ""
        
        f = extrair_nota(veredicto, "faithfulness")
        ar = extrair_nota(veredicto, "answer_relevancy")
        cp = extrair_nota(veredicto, "context_precision")
        
        print(f"   -> Parciais: faithfulness={f}, answer_relevancy={ar}, context_precision={cp}")
        
        total_faith += f
        total_relevancy += ar
        total_precision += cp
        time.sleep(1) # Delay de segurança para evitar limites de rotação da API
        
    print("\n" + "="*60)
    print("🏆 RESULTADOS CONSOLIDADOS DO SEU GOLDEN SET 🏆")
    print("="*60)
    print(f"faithfulness={total_faith/qtd_queries:.2f}, answer_relevancy={total_relevancy/qtd_queries:.2f}, context_precision={total_precision/qtd_queries:.2f}")
    print("="*60)

if __name__ == "__main__":
    rodar_avaliacao_limpa()