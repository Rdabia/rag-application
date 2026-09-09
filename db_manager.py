import json
import sqlite3
import numpy as np

DB_NAME = "knowledge_base.db"


def init_db():
    """Initializes SQLite database with source metadata support."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_file TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()


def is_db_empty() -> bool:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM documents LIMIT 1")
    result = cursor.fetchone()
    conn.close()
    return result is None


def insert_chunks_bulk(records: list[tuple[str, str, list[float]]]):
    """Stores multiple chunks in SQLite in a single transaction."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    formatted_data = [
        (source_file, content, json.dumps(vec))
        for source_file, content, vec in records
    ]

    cursor.executemany(
        "INSERT INTO documents (source_file, content, embedding) VALUES (?, ?,"
        " ?)",
        formatted_data,
    )
    conn.commit()
    conn.close()


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity."""
    norm_1 = np.linalg.norm(v1)
    norm_2 = np.linalg.norm(v2)
    if norm_1 == 0 or norm_2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm_1 * norm_2))


def get_top_k_relevant(
    query_embedding: list[float], top_k: int = 3
) -> list[dict]:
    """Retrieves top-k relevant chunks with source metadata."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT source_file, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return []

    q_vec = np.array(query_embedding, dtype=np.float32)
    scored_docs = []

    for source_file, content, emb_json in rows:
        doc_vec = np.array(json.loads(emb_json), dtype=np.float32)
        score = cosine_similarity(q_vec, doc_vec)
        scored_docs.append(
            {"source": source_file, "content": content, "score": score}
        )

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs[:top_k]