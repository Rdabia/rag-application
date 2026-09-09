import db_manager
from foundry_local_sdk import Configuration, FoundryLocalManager


SYSTEM_PROMPT_TEMPLATE = (
    "You are a strict technical Q&A assistant without personal identity.\n"
    "Your job is to answer using ONLY the provided context below.\n\n"
    "RULES:\n"
    "1. Rely ONLY on facts directly stated in the Context. Do NOT use any "
    "knowledge you already have, even if you are confident it is correct.\n"
    "2. If the Context does not contain the answer, or only mentions the "
    "topic in passing without actually answering the question, respond "
    "EXACTLY with: 'I don't have enough information in my knowledge base "
    "to answer this question.' Do not guess, and do not fill in gaps with "
    "your own knowledge.\n"
    "3. Always cite the source file name when giving an answer.\n"
    "4. Never answer general-knowledge questions (geography, sports, "
    "current events, etc.) unless the exact answer is explicitly present "
    "in the Context below.\n\n"
    "--- CONTEXT START ---\n{context}\n--- CONTEXT END ---"
)

NO_INFO_MESSAGE = (
    "I don't have enough information in my knowledge base to answer this "
    "question."
)

# Minimum cosine similarity for a chunk to be considered relevant.
# Tune this by testing a few in-scope and out-of-scope questions and
# checking the printed scores (see DEBUG note below).
SIMILARITY_THRESHOLD = 0.4


def build_context(relevant_chunks: list[dict]) -> str:
    """Formats retrieved chunks with source attribution."""
    context_blocks = [
        f"[{i + 1}] (Source: {item['source']}):\n{item['content']}"
        for i, item in enumerate(relevant_chunks)
    ]
    return "\n\n".join(context_blocks)


def ensure_model_ready(model, alias: str) -> None:
    """Downloads the model if it isn't cached yet, then loads it into memory."""
    if not model.is_cached:
        print(f"'{alias}' is not cached yet, downloading...")

        def _on_progress(progress):
            # The SDK may report progress as a 0-1 fraction, a 0-100
            # percentage, or an object with a .percentage attribute.
            # Normalize whatever we get into a clean 0-100 percentage.
            if hasattr(progress, "percentage"):
                pct = progress.percentage
            else:
                pct = progress

            pct = float(pct)
            if pct <= 1.0:
                pct *= 100  # was a 0-1 fraction

            pct = max(0.0, min(100.0, pct))  # clamp to a sane range
            print(f"\rDownloading '{alias}': {pct:5.1f}%", end="", flush=True)

        model.download(_on_progress)
        print()  # move to a new line after the progress bar

    print(f"Loading '{alias}' into memory...")
    model.load()


def main():
    db_manager.init_db()

    if db_manager.is_db_empty():
        print("Database is empty — running ingestion first...\n")
        import ingest

        ingest.process_files()

        if db_manager.is_db_empty():
            # ingest.py ran but found no .txt files in data/, so there's
            # still nothing to answer questions from.
            print(
                "\nStill no data after ingestion. Add .txt files to "
                "'data/' and try again."
            )
            return

        print()  # spacing before the model-loading logs below

    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Models
    CHAT_MODEL_ALIAS = "phi-3.5-mini"

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    ensure_model_ready(embedding_model, "qwen3-embedding-0.6b")
    embedding_client = embedding_model.get_embedding_client()

    chat_model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
    ensure_model_ready(chat_model, CHAT_MODEL_ALIAS)
    chat_client = chat_model.get_chat_client()

    print("\nModels loaded. Enter queries below:\n")

    try:
        while True:
            query = input("Question (type 'quit' to exit): ").strip()

            if query.lower() == "quit":
                break
            if not query:
                # Empty input -> just ask again, don't exit
                continue

            # --- Retrieval ---
            try:
                q_emb = (
                    embedding_client.generate_embedding(query)
                    .data[0]
                    .embedding
                )
                relevant_chunks = db_manager.get_top_k_relevant(q_emb, top_k=3)
            except Exception as e:
                print(f"Answer: [Error during retrieval: {e}]\n")
                continue

            # Filter out chunks that aren't actually similar enough.
            # This stops the model from being fed irrelevant context for
            # out-of-scope questions (e.g. general knowledge questions
            # the tiny chat model would otherwise try to "help" with).
            relevant_chunks = [
                c for c in relevant_chunks if c["score"] >= SIMILARITY_THRESHOLD
            ]

            if not relevant_chunks:
                print(f"Answer: {NO_INFO_MESSAGE}\n")
                continue

            context = build_context(relevant_chunks)

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_TEMPLATE.format(context=context),
                },
                {"role": "user", "content": query},
            ]

            # --- Generation ---
            print("Answer: ", end="", flush=True)
            try:
                for chunk in chat_client.complete_streaming_chat(messages):
                    if chunk.choices and chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
            except Exception as e:
                print(f"\n[Error during generation: {e}]", end="")
            print("\n")

    finally:
        # Guaranteed to run even on Ctrl+C or an unexpected crash
        print("\nUnloading models...")
        embedding_model.unload()
        chat_model.unload()


if __name__ == "__main__":
    main()