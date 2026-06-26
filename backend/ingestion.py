import fitz  # PyMuPDF for PDF
import os
import pickle
import numpy as np
import faiss
import json
import csv
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_STORE_DIR = "vector_stores"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


# ── Extractors ──

def extract_pdf(path: str) -> list[dict]:
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def extract_txt(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    # Split into chunks of ~2000 chars treating each as a "page"
    size = 2000
    parts = [text[i:i+size] for i in range(0, len(text), size)]
    return [{"page": i + 1, "text": p} for i, p in enumerate(parts) if p.strip()]


def extract_docx(path: str) -> list[dict]:
    try:
        import docx
        doc = docx.Document(path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        size = 2000
        parts = [text[i:i+size] for i in range(0, len(text), size)]
        return [{"page": i + 1, "text": p} for i, p in enumerate(parts) if p.strip()]
    except Exception as e:
        return [{"page": 1, "text": f"Could not extract DOCX content: {str(e)}"}]


def extract_image(path: str) -> list[dict]:
    # Use PyMuPDF to extract text from image via OCR if available
    # Otherwise return description placeholder for LLM to handle
    try:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        if text.strip():
            return [{"page": 1, "text": text.strip()}]
    except:
        pass
    # Read raw image and return as base64 description
    return [{"page": 1, "text": f"[Image file: {os.path.basename(path)}. The image has been uploaded. Please analyze based on available context.]"}]


def extract_csv(path: str) -> list[dict]:
    pages = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # Group rows into chunks of 50
    chunk_size = 50
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i+chunk_size]
        text = "\n".join([", ".join(row) for row in chunk])
        pages.append({"page": i // chunk_size + 1, "text": text})
    return pages


def extract_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    text = json.dumps(data, indent=2)
    size = 2000
    parts = [text[i:i+size] for i in range(0, len(text), size)]
    return [{"page": i + 1, "text": p} for i, p in enumerate(parts) if p.strip()]


# ── Chunking ──

def chunk_pages(pages: list[dict], size: int = 400, overlap: int = 80) -> list[dict]:
    chunks = []
    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            chunks.append({"text": " ".join(words[start:end]), "page": page["page"]})
            if end == len(words):
                break
            start += size - overlap
    return chunks


# ── Embed & Store ──

def embed_and_store(chunks: list[dict], session_id: str) -> int:
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    path = os.path.join(VECTOR_STORE_DIR, session_id)
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    return len(chunks)


# ── Main ingestion ──

def ingest_file(file_path: str, session_id: str, ext: str) -> tuple[int, str]:
    ext = ext.lower()

    if ext == ".pdf":
        pages = extract_pdf(file_path)
        file_type = "PDF"
    elif ext in [".txt", ".md"]:
        pages = extract_txt(file_path)
        file_type = "Text"
    elif ext in [".docx", ".doc"]:
        pages = extract_docx(file_path)
        file_type = "Word Document"
    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
        pages = extract_image(file_path)
        file_type = "Image"
    elif ext == ".csv":
        pages = extract_csv(file_path)
        file_type = "CSV"
    elif ext == ".json":
        pages = extract_json(file_path)
        file_type = "JSON"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not pages:
        raise ValueError("No content could be extracted from this file.")

    chunks = chunk_pages(pages)
    num_chunks = embed_and_store(chunks, session_id)

    # Clean up uploaded file
    if os.path.exists(file_path):
        os.remove(file_path)

    return num_chunks, file_type
