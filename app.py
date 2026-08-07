"""Orquestracao simples do loop Agentic RAG para a PoC."""

import json
import os
import re
from pathlib import Path

import streamlit as st
from openai import OpenAI

from credit_tools import buscar_informacoes_relevantes, validar_elegibilidade


SYSTEM_PROMPT = """
Voce e um assistente de credito B2B que atua de forma informativa.
Use o contexto recuperado para identificar produtos de credito aderentes.
Quando a pergunta envolver faturamento, receita, tempo de empresa ou idade da empresa,
voce OBRIGATORIAMENTE deve chamar a ferramenta validar_elegibilidade antes de responder.
Extraia numeros da pergunta original do usuario, nao da query limpa.
O campo idade_cliente deve ser informado em meses.
O campo produto deve ser a chave de um produto recuperado no contexto.
Se faltar faturamento, tempo de empresa ou produto aderente, peca a informacao ao cliente.
Se usar um dado do contexto para simulacao, diga explicitamente que e uma simulacao.
Nunca invente aprovacao, reprovacao ou regra numerica sem retorno da ferramenta.
""".strip()

VALIDAR_ELEGIBILIDADE_TOOL = {
    "type": "function",
    "name": "validar_elegibilidade",
    "description": (
        "Valida elegibilidade de uma empresa para um produto de credito usando "
        "faturamento e tempo de empresa."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "faturamento_cliente": {
                "type": "number",
                "description": "Faturamento do cliente em reais.",
            },
            "idade_cliente": {
                "type": "integer",
                "description": "Tempo de empresa do cliente em meses.",
            },
            "produto": {
                "type": "string",
                "description": "Chave do produto recuperado no contexto.",
            },
        },
        "required": ["faturamento_cliente", "idade_cliente", "produto"],
        "additionalProperties": False,
    },
}


def limpar_query(texto):
    """Remove ruido numerico da pergunta antes da busca vetorial."""
    sem_moeda = re.sub(r"R\$\s*", " ", texto, flags=re.IGNORECASE)
    sem_numeros = re.sub(r"\b\d+([.,]\d+)?\b", " ", sem_moeda)
    sem_sinais = re.sub(r"\s+", " ", sem_numeros)
    return sem_sinais.strip()


def montar_contexto_llm(resultados):
    """Formata resultados recuperados para enviar ao modelo."""
    linhas = []
    for resultado in resultados:
        linha = (
            f"- tipo: {resultado['tipo']}\n"
            f"  id: {resultado['id']}\n"
            f"  titulo: {resultado['titulo']}\n"
            f"  texto: {resultado['texto']}"
        )
        produto = resultado.get("produto")
        if produto:
            produto_id = resultado["id"].replace("produto:", "")
            linha += (
                f"\n  produto_chave: {produto_id}"
                f"\n  faturamento_minimo: {produto['faturamento_minimo']}"
                f"\n  idade_minima_meses: {produto['idade_minima_meses']}"
            )
        linhas.append(linha)
    return "\n\n".join(linhas)


def responder_pergunta(pergunta_usuario):
    """Executa o loop: query original, limpeza, retrieval, LLM e ferramenta."""
    _carregar_env_local()
    query_limpa = limpar_query(pergunta_usuario)
    resultados = buscar_informacoes_relevantes(query_limpa)
    contexto = montar_contexto_llm(resultados)
    logs = [
        {"etapa": "query_original", "valor": pergunta_usuario},
        {"etapa": "query_limpa", "valor": query_limpa},
        {"etapa": "retrieval", "valor": resultados},
    ]

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "resposta": (
                "OPENAI_API_KEY nao esta configurada. Configure a chave no .env "
                "para executar a etapa de LLM e function calling."
            ),
            "query_limpa": query_limpa,
            "contexto": resultados,
            "logs": logs,
        }

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    prompt_usuario = (
        "Pergunta original do usuario:\n"
        f"{pergunta_usuario}\n\n"
        "Contexto recuperado via busca vetorial:\n"
        f"{contexto}"
    )

    primeira_resposta = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=prompt_usuario,
        tools=[VALIDAR_ELEGIBILIDADE_TOOL],
    )
    logs.append({"etapa": "llm_primeira_resposta", "valor": _resumo_resposta(primeira_resposta)})

    tool_calls = [
        item
        for item in primeira_resposta.output
        if getattr(item, "type", None) == "function_call"
        and getattr(item, "name", None) == "validar_elegibilidade"
    ]

    if not tool_calls:
        return {
            "resposta": primeira_resposta.output_text,
            "query_limpa": query_limpa,
            "contexto": resultados,
            "logs": logs,
        }

    function_outputs = []
    for tool_call in tool_calls:
        argumentos = json.loads(tool_call.arguments)
        resultado_ferramenta = validar_elegibilidade(
            argumentos["faturamento_cliente"],
            argumentos["idade_cliente"],
            argumentos["produto"],
        )
        logs.append(
            {
                "etapa": "tool_call",
                "nome": "validar_elegibilidade",
                "argumentos": argumentos,
                "retorno": resultado_ferramenta,
            }
        )
        function_outputs.append(
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": json.dumps(resultado_ferramenta, ensure_ascii=True),
            }
        )

    resposta_final = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        previous_response_id=primeira_resposta.id,
        input=function_outputs,
        tools=[VALIDAR_ELEGIBILIDADE_TOOL],
    )
    logs.append({"etapa": "llm_resposta_final", "valor": _resumo_resposta(resposta_final)})

    return {
        "resposta": resposta_final.output_text,
        "query_limpa": query_limpa,
        "contexto": resultados,
        "logs": logs,
    }


def _carregar_env_local(caminho=None):
    if caminho is None:
        caminho = Path(__file__).with_name(".env")

    if not os.path.exists(caminho):
        return

    with open(caminho, encoding="utf-8") as arquivo_env:
        for linha in arquivo_env:
            linha = linha.strip()
            if not linha or linha.startswith("#") or "=" not in linha:
                continue
            chave, valor = linha.split("=", 1)
            chave = chave.strip()
            valor = valor.strip().strip('"').strip("'")
            if not os.getenv(chave):
                os.environ[chave] = valor


def _resumo_resposta(resposta):
    return {
        "id": resposta.id,
        "output_text": resposta.output_text,
    }


def main():
    st.set_page_config(page_title="CreditLLM", page_icon="CreditLLM")
    st.title("CreditLLM")

    pergunta = st.text_input(
        "Pergunta do cliente",
        placeholder="Quero credito de giro, minha empresa tem 2 anos e fatura 1 milhao",
    )
    if not pergunta:
        return

    with st.spinner("Processando..."):
        resultado = responder_pergunta(pergunta)

    st.write(resultado["resposta"])


if __name__ == "__main__":
    main()
