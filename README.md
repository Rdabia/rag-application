# Offline RAG Assistant

A fully offline, local Retrieval-Augmented Generation (RAG) Q&A assistant built with [Microsoft Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/what-is-foundry-local). It answers questions about a document collection by retrieving relevant passages locally and feeding them to an on-device LLM — no internet connection or cloud API required.

This project was built as part of a one-month RAG development program, following the "retrieve, augment, generate" pattern.

## How it works

```
User question
     │
     ▼
Embed the question (qwen3-embedding-0.6b)
     │
     ▼
Compare against stored chunk embeddings (cosine similarity, SQLite)
     │
     ▼
Take the top-k most relevant chunks as context
     │
     ▼
Build a prompt: system rules + context + question
     │
     ▼
Local chat model generates an answer (phi-3.5-mini)
     │
     ▼
Answer + source citation printed to the user
```

The core idea: the model is instructed to answer **only** using the retrieved context, and to explicitly say it doesn't know if the answer isn't in the context. This reduces hallucination and makes every answer traceable back to a source file.

## Features

- **Fully offline inference** via Foundry Local — no API keys, no network calls.
- **Local vector search** using SQLite + cosine similarity (no external vector DB needed at this scale).
- **Source-grounded answers** — every response cites the source file it was based on.
- **Relevance filtering** — a similarity-score threshold prevents the assistant from answering questions that fall outside the knowledge base, instead of guessing.
- **Streaming responses** in the CLI for a more responsive feel.

## Project structure

```
rag-application/
├── data/                  # Put your .txt source documents here
├── ingest.py              # Chunks documents, generates embeddings, stores them in SQLite
├── db_manager.py          # SQLite setup, bulk insert, cosine similarity search
├── main.py                # CLI chat loop: retrieval + generation
├── knowledge_base.db       # Generated after running ingest.py
└── README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install foundry-local-sdk numpy
   ```
2. Create a `data/` folder (or let `ingest.py` create it for you) and add your `.txt` source documents.
3. Run ingestion to build the knowledge base:
   ```bash
   python3 ingest.py
   ```
   This reads every `.txt` file in `data/`, splits it into overlapping chunks, generates an embedding for each chunk, and stores everything in `knowledge_base.db`.
4. Run the assistant:
   ```bash
   python3 main.py
   ```
   Type a question and press Enter. Type `quit` to exit.

## Configuration

| Setting | File | Default | Purpose |
|---|---|---|---|
| `CHUNK_SIZE` | `ingest.py` | 2000 chars | Size of each text chunk |
| `CHUNK_OVERLAP` | `ingest.py` | 400 chars | Overlap between consecutive chunks |
| `SIMILARITY_THRESHOLD` | `main.py` | 0.4 | Minimum cosine similarity for a chunk to be considered relevant |
| Embedding model | `main.py` / `ingest.py` | `qwen3-embedding-0.6b` | Converts text to vectors |
| Chat model | `main.py` | `phi-3.5-mini` | Generates the final answer |

## Testing

A set of manual test queries was used to verify the assistant behaves correctly both for in-scope and out-of-scope questions:

| Query | Expected behavior | Result |
|---|---|---|
| "Who is considered the father of medicine?" | Correct answer with source citation | ✅ Correct (Hippocrates, cited) |
| "How did the Greeks contribute to the history of medicine?" | Correct, context-grounded answer | ✅ Correct |
| "What is the capital of France?" | "I don't have enough information..." | ✅ Correct — initially hallucinated ("Paris") with `qwen2.5-0.5b`; fixed by switching to `phi-3.5-mini` and adding a similarity threshold, then re-verified |
| "Who won the World Cup in 2026?" | "I don't have enough information..." | ✅ Correct — initially hallucinated ("Argentina"); fixed and re-verified the same way |
| Empty input | Re-prompt, don't exit | ✅ Correct |
| `quit` | Clean exit, models unloaded | ✅ Correct |

## Lessons learned

- **Small models don't reliably follow instructions.** The initial chat model (`qwen2.5-0.5b`, 0.5B parameters) ignored the system prompt's "only use the provided context" rule for general-knowledge questions it already "knew" the answer to (e.g. the capital of France). Switching to a slightly larger model (`phi-3.5-mini`) and adding a **similarity-score threshold** — rejecting retrieval results below a minimum relevance score before they even reach the model — solved this reliably.
- **Don't assume an SDK supports batching.** An attempt to speed up embedding generation by sending a list of texts in one call failed, because `foundry_local_sdk`'s `generate_embedding()` only accepts a single string. Always check the actual API surface rather than assuming feature parity with other embedding APIs.
- **Model loading/unloading needs to be crash-safe.** Wrapping the main chat loop in `try/finally` ensures models are always unloaded, even if the user exits with `Ctrl+C` or an unexpected error occurs mid-query.

## Limitations & future work

- Retrieval is brute-force (loads all embeddings into memory and compares one by one) — fine for a small document set, but wouldn't scale to a large knowledge base without a proper vector index.
- Only `.txt` files are supported for ingestion.
- The similarity threshold (`0.4`) was tuned for this specific dataset and embedding model; it may need adjustment for other document collections.
- Currently CLI-only; a Streamlit/Gradio interface would be a natural next step for easier interaction.