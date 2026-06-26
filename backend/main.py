from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ingestion import ingest_file
from retrieval import retrieve_chunks
from llm import get_answer
import os
import uuid

app = FastAPI(title="DocMind V2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

sessions = {}

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".csv", ".json"
}

class QuestionRequest(BaseModel):
    session_id: str
    question: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported. Supported: PDF, DOCX, TXT, MD, PNG, JPG, JPEG, CSV, JSON")

    content = await file.read()
    session_id = str(uuid.uuid4())

    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{session_id}{ext}"
    with open(file_path, "wb") as f:
        f.write(content)

    num_chunks, file_type = ingest_file(file_path, session_id, ext)

    sessions[session_id] = {
        "filename": file.filename,
        "file_type": file_type,
        "chat_history": []
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "file_type": file_type,
        "chunks_indexed": num_chunks,
        "message": "File uploaded and indexed successfully."
    }

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a file first.")

    session = sessions[req.session_id]
    chunks = retrieve_chunks(req.question, req.session_id, top_k=5)

    if not chunks:
        raise HTTPException(status_code=500, detail="No relevant content found.")

    answer, sources = get_answer(req.question, chunks, session["chat_history"])

    session["chat_history"].append({"role": "user", "content": req.question})
    session["chat_history"].append({"role": "assistant", "content": answer})
    if len(session["chat_history"]) > 12:
        session["chat_history"] = session["chat_history"][-12:]

    return {
        "answer": answer,
        "sources": sources,
        "document": session["filename"]
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
