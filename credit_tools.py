import hashlib
import math
import re
import unicodedata

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

PRODUTOS_CREDITO = {
    "capital_giro": {
        "nome": "Capital de Giro Flex",
        "descricao_para_busca": (
            "Credito para reforcar caixa, pagar fornecedores, cobrir despesas "
            "operacionais e financiar necessidade de capital de giro."
        ),
        "faturamento_minimo": 500_000,
        "idade_minima_meses": 12,
    },
    "bndes_automatico": {
        "nome": "BNDES Automatico Empresarial",
        "descricao_para_busca": (
            "Financiamento de longo prazo para expansao, modernizacao, projetos "
            "produtivos e investimento empresarial com perfil BNDES."
        ),
        "faturamento_minimo": 2_400_000,
        "idade_minima_meses": 24,
    },
    "antecipacao_recebiveis": {
        "nome": "Antecipacao de Recebiveis",
        "descricao_para_busca": (
            "Linha para antecipar duplicatas, boletos, contratos e recebiveis "
            "comerciais de clientes corporativos."
        ),
        "faturamento_minimo": 300_000,
        "idade_minima_meses": 6,
    },
    "financiamento_maquinas": {
        "nome": "Financiamento de Maquinas e Equipamentos",
        "descricao_para_busca": (
            "Credito para compra de maquinas, equipamentos, veiculos produtivos "
            "e ativos usados na operacao da empresa."
        ),
        "faturamento_minimo": 1_200_000,
        "idade_minima_meses": 18,
    },
    "conta_garantida": {
        "nome": "Conta Garantida PJ",
        "descricao_para_busca": (
            "Limite rotativo para empresas que precisam de liquidez imediata, "
            "cobertura de fluxo de caixa e credito emergencial."
        ),
        "faturamento_minimo": 800_000,
        "idade_minima_meses": 12,
    },
}

CONTEXTO_FINANCEIRO_BRASIL = {
    "capital_giro": {
        "titulo": "Capital de giro",
        "texto": (
            "Capital de giro e usado para financiar caixa, estoque, fornecedores, "
            "folha de pagamento e despesas operacionais de curto prazo."
        ),
    },
    "bndes": {
        "titulo": "BNDES",
        "texto": (
            "Linhas BNDES normalmente financiam investimento produtivo, expansao, "
            "modernizacao e compra de bens com prazos mais longos."
        ),
    },
    "recebiveis": {
        "titulo": "Antecipacao de recebiveis",
        "texto": (
            "Antecipacao de recebiveis transforma vendas a prazo, boletos, duplicatas "
            "ou contratos em caixa antes do vencimento."
        ),
    },
    "conta_garantida": {
        "titulo": "Conta garantida",
        "texto": (
            "Conta garantida e um limite rotativo para cobrir descasamentos temporarios "
            "de fluxo de caixa empresarial."
        ),
    },
    "faturamento": {
        "titulo": "Faturamento empresarial",
        "texto": (
            "Faturamento anual ou mensal ajuda a medir porte, capacidade de pagamento "
            "e aderencia minima a politicas de credito."
        ),
    },
    "tempo_empresa": {
        "titulo": "Tempo de empresa",
        "texto": (
            "Tempo de empresa indica historico operacional e maturidade do negocio, "
            "sendo criterio comum para elegibilidade em credito PJ."
        ),
    },
    "garantias": {
        "titulo": "Garantias",
        "texto": (
            "Garantias podem reduzir risco da operacao e incluir recebiveis, veiculos, "
            "maquinas, aval ou outros ativos empresariais."
        ),
    },
    "liquidez": {
        "titulo": "Liquidez",
        "texto": (
            "Liquidez representa a capacidade da empresa cumprir obrigacoes de curto "
            "prazo sem comprometer a operacao."
        ),
    },
    "investimento": {
        "titulo": "Investimento produtivo",
        "texto": (
            "Investimento produtivo envolve compra de maquinas, equipamentos, tecnologia "
            "ou expansao que aumenta capacidade operacional."
        ),
    },
    "risco_credito": {
        "titulo": "Risco de credito",
        "texto": (
            "Risco de credito avalia a chance de inadimplencia usando dados cadastrais, "
            "financeiros, historico e caracteristicas da operacao."
        ),
    },
}

_VECTOR_CLIENT = chromadb.Client()
_EMBEDDING_FUNCTION = None
_COLLECTION = None
_TOKENS_GENERICOS_RANKING = {
    "credito",
    "empresa",
    "empresas",
    "minha",
    "quero",
    "preciso",
    "linha",
    "anos",
    "fatura",
    "faturamento",
}


class HashingEmbeddingFunction(EmbeddingFunction):
    """Embedding local simples para a PoC, sem downloads ou chamadas externas."""

    def __call__(self, input: Documents) -> Embeddings:
        return [_embedding_hashing(documento) for documento in input]


def buscar_informacoes_relevantes(query_limpa, limite=4):
    """Busca produtos e contexto financeiro relevantes para uma query ja limpa."""
    collection = _obter_collection()
    total_documentos = len(PRODUTOS_CREDITO) + len(CONTEXTO_FINANCEIRO_BRASIL)
    resultados = collection.query(
        query_texts=[query_limpa],
        n_results=total_documentos,
        include=["documents", "metadatas"],
    )

    itens = []
    ids = resultados.get("ids", [[]])[0]
    documentos = resultados.get("documents", [[]])[0]
    metadados = resultados.get("metadatas", [[]])[0]

    for item_id, documento, metadata in zip(ids, documentos, metadados):
        item = {
            "tipo": metadata["tipo"],
            "id": item_id,
            "titulo": metadata["titulo"],
            "texto": documento,
        }
        produto_id = metadata.get("produto_id")
        if produto_id:
            item["produto"] = PRODUTOS_CREDITO[produto_id]
        itens.append(item)

    itens.sort(key=lambda item: _pontuar_relevancia(query_limpa, item), reverse=True)
    return itens[:limite]


def _obter_collection():
    global _COLLECTION, _EMBEDDING_FUNCTION

    if _COLLECTION is not None:
        return _COLLECTION

    _EMBEDDING_FUNCTION = HashingEmbeddingFunction()
    _COLLECTION = _VECTOR_CLIENT.get_or_create_collection(
        name="creditllm_contexto",
        embedding_function=_EMBEDDING_FUNCTION,
    )
    _popular_collection(_COLLECTION)
    return _COLLECTION


def _popular_collection(collection):
    ids = []
    documents = []
    metadatas = []

    for produto_id, produto in PRODUTOS_CREDITO.items():
        ids.append(f"produto:{produto_id}")
        documents.append(produto["descricao_para_busca"])
        metadatas.append(
            {
                "tipo": "produto",
                "titulo": produto["nome"],
                "produto_id": produto_id,
            }
        )

    for contexto_id, contexto in CONTEXTO_FINANCEIRO_BRASIL.items():
        ids.append(f"contexto:{contexto_id}")
        documents.append(contexto["texto"])
        metadatas.append(
            {
                "tipo": "contexto",
                "titulo": contexto["titulo"],
                "produto_id": "",
            }
        )

    collection.add(ids=ids, documents=documents, metadatas=metadatas)


def validar_elegibilidade(faturamento_cliente, idade_cliente, produto):
    """Valida faturamento e idade minima da empresa para um produto."""
    produto_info = _resolver_produto(produto)
    if produto_info is None:
        return {"status": "reprovado", "motivo": "Produto invalido"}

    motivos = []
    if faturamento_cliente < produto_info["faturamento_minimo"]:
        motivos.append("Faturamento insuficiente")
    if idade_cliente < produto_info["idade_minima_meses"]:
        motivos.append("Tempo de empresa insuficiente")

    if motivos:
        return {"status": "reprovado", "motivo": "; ".join(motivos)}

    return {"status": "aprovado"}


def _resolver_produto(produto):
    if isinstance(produto, str):
        return PRODUTOS_CREDITO.get(produto)

    if isinstance(produto, dict):
        campos_obrigatorios = {
            "nome",
            "descricao_para_busca",
            "faturamento_minimo",
            "idade_minima_meses",
        }
        if campos_obrigatorios.issubset(produto):
            return produto

    return None


def _embedding_hashing(texto, dimensoes=64):
    vetor = [0.0] * dimensoes
    for token in _tokens(texto):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        indice = int(digest[:8], 16) % dimensoes
        vetor[indice] += 1.0

    norma = math.sqrt(sum(valor * valor for valor in vetor))
    if norma == 0:
        return vetor

    return [valor / norma for valor in vetor]


def _tokens(texto):
    texto_ascii = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.findall(r"[a-zA-Z]{3,}", texto_ascii.lower())


def _pontuar_relevancia(query, item):
    tokens_query = set(_tokens(query)) - _TOKENS_GENERICOS_RANKING
    texto_item = f"{item['id']} {item['titulo']} {item['texto']}"
    tokens_item = set(_tokens(texto_item)) - _TOKENS_GENERICOS_RANKING
    return len(tokens_query & tokens_item)
