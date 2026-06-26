import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_STORE_DIR = "vector_stores"


def retrieve_chunks(query: str, session_id: str, top_k: int = 5) -> list[dict]:
    path = os.path.join(VECTOR_STORE_DIR, session_id)
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    k = min(top_k, len(chunks))
    distances, indices = index.search(query_embedding, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "score": float(dist)
        })

    results.sort(key=lambda x: x["score"])
    return results
