from pathlib import Path
import db_manager
from foundry_local_sdk import Configuration, FoundryLocalManager

DATA_DIR = Path("data")
CHUNK_SIZE = 2000  # Cannot be <= overlap
CHUNK_OVERLAP = 400
BATCH_SIZE = 32


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Cuts the text into chunks."""
    if chunk_size <= overlap:
        print(
            "Chunk size cannot be equal to/smaller than overlap, "
            "resetting to defaults (2000/400)."
        )
        chunk_size = 2000
        overlap = 400

    chunks = []
    start = 0
    text = text.strip().replace("\r", "")

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def process_files():
    db_manager.init_db()

    if not DATA_DIR.exists():
        DATA_DIR.mkdir()
        print(f"'{DATA_DIR}' folder created, please add your .txt files.")
        return

    txt_files = list(DATA_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in '{DATA_DIR}'.")
        return

    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    print(f"Found {len(txt_files)} file(s). Processing...\n")

    # 1) Read all files, split into chunks, collect them
    all_chunks = []
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_text(content)
        print(f"Processing: '{file_path.name}' -> {len(chunks)} chunk(s) created")

        for chunk in chunks:
            all_chunks.append((file_path.name, chunk))

    total_chunks = len(all_chunks)

    if not all_chunks:
        print("No chunks were created, exiting.")
        embedding_model.unload()
        return

    # 2) Generate embeddings one chunk at a time
    # NOTE: this SDK's generate_embedding() only accepts a single string,
    # not a list, so batching isn't possible here.
    bulk_records = []

    for idx, (source, chunk) in enumerate(all_chunks, 1):
        print(f"\rGenerating embeddings: {idx}/{total_chunks}...", end="", flush=True)

        response = embedding_client.generate_embedding(chunk)
        vec = response.data[0].embedding

        bulk_records.append((source, chunk, vec))

    print()  # move to a new line after the progress bar

    # 3) Write everything to the database in one go
    if bulk_records:
        db_manager.insert_chunks_bulk(bulk_records)

    embedding_model.unload()

    print(f"\nSuccessfully loaded {total_chunks} chunk(s) into the SQLite database!")


if __name__ == "__main__":
    process_files()