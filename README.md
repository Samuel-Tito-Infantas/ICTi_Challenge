# CreditLLM

CreditLLM is a proof of concept for a B2B credit assistant using Agentic RAG.

The main idea is to separate two responsibilities:

- the LLM interprets the customer's question and organizes the response;
- numerical eligibility rules are validated by deterministic Python code.

This prevents the model from "inventing" calculations or approving products outside the defined policy.

## How It Works

The PoC flow follows five simple steps:

1. The customer asks a question about credit.
2. The question is cleaned to remove numbers before semantic search.
3. The system retrieves credit products and financial context from a local vector database.
4. The LLM receives the original question and the retrieved context.
5. When revenue and company age are provided, the LLM calls a Python function to validate eligibility.

## What Is Already Built

- Mock corporate credit products.
- Local vector search with in-memory ChromaDB.
- Deterministic eligibility validation function.
- Orchestration with the OpenAI Responses API and function calling.
- Simple Streamlit interface.
- Notebook with documented test cases.

## Setup

Install the dependencies:

```bash
uv sync --group dev
```

Configure the OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5-nano
CHROMA_PERSIST_DIR=.chroma
```

## Running the App

```bash
uv run streamlit run app.py
```

## Validating the Loop

Run the validation notebook:

```bash
uv run jupyter nbconvert --to notebook --execute sandbox_jupyters/task5_loop_tests.ipynb --output task5_loop_tests_executed.ipynb --output-dir sandbox_jupyters
```

The notebook covers three scenarios:

- customer approved for working capital;
- customer rejected due to insufficient revenue;
- question with incomplete data.

## Next Steps

- Display system logs in the UI to show when the tool was called.
- Add automated tests.
- Refine value extraction for revenue and company age.
- Improve vector search ranking and quality.
- Replace mock data with real policies in a controlled environment.
