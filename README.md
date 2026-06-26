# DocMind — AI-Powered Document Q&A System

A full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload any file and ask natural language questions, receiving precise answers grounded strictly in the document content with page-level citations.

Built using FastAPI, FAISS, Sentence Transformers, and Llama 3.3 via Groq as a placement preparation project demonstrating AI/ML pipelines, vector search, and full-stack development.

---

## Live Features

- Upload any file — PDF, Word document, plain text, markdown, image, CSV, or JSON
- Ask natural language questions and receive answers grounded in the document
- Page-level source citations with relevant excerpts for every answer
- Conversational memory — follow-up questions understand prior context
- Multi-format text extraction with intelligent chunking and overlap
- Semantic vector search using FAISS for fast and accurate retrieval
- Powered by Llama 3.3 70B via Groq — completely free, no credit card required

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | HTML, CSS, Vanilla JavaScript | UI and file upload |
| Backend | FastAPI + Uvicorn | REST API server |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) | Text vectorization |
| Vector Store | FAISS (CPU) | Similarity search |
| LLM | Llama 3.3 70B via Groq API | Answer generation |
| PDF Parsing | PyMuPDF (fitz) | Text extraction from PDFs |
| Word Parsing | python-docx | Text extraction from DOCX files |

---

## Architecture

```
User Uploads File
        │
        ▼
  File Type Detection
        │
        ├── PDF ──────────► PyMuPDF text extraction
        ├── DOCX ─────────► python-docx extraction
        ├── TXT / MD ─────► Direct text read
        ├── CSV ──────────► Row-based chunking
        ├── JSON ─────────► Structured text extraction
        └── Image ────────► PyMuPDF OCR extraction
        │
        ▼
  Text Chunking (400 tokens, 80 token overlap)
        │
        ▼
  Sentence Transformer Embeddings (384-dim vectors)
        │
        ▼
  FAISS IndexFlatL2 (stored per session on disk)


User Asks Question
        │
        ▼
  Query Embedding (same Sentence Transformer model)
        │
        ▼
  FAISS Similarity Search → Top 5 most relevant chunks
        │
        ▼
  Llama 3.3 70B (Groq) with chunks as context
        │
        ▼
  Answer + Page Citations returned to user
```

---

## Supported File Types

| Format | Extension | Extraction Method |
|---|---|---|
| PDF | .pdf | PyMuPDF page-by-page extraction |
| Word Document | .docx, .doc | python-docx paragraph extraction |
| Plain Text | .txt, .md | Direct file read with chunking |
| Image | .png, .jpg, .jpeg, .webp, .gif | PyMuPDF OCR |
| Spreadsheet | .csv | Row-based chunking (50 rows per chunk) |
| Data File | .json | Structured JSON to text conversion |

---

## Project Structure

```
docmind-v2/
├── backend/
│   ├── main.py           # FastAPI app — upload, ask, session management
│   ├── ingestion.py      # File extraction, chunking, embedding, FAISS indexing
│   ├── retrieval.py      # Query embedding and FAISS similarity search
│   ├── llm.py            # Groq API call with RAG context and chat history
│   └── requirements.txt  # Python dependencies
├── frontend/
│   └── index.html        # Complete UI — upload, chat, source citations
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload file → extract → chunk → embed → index |
| POST | `/ask` | Ask question → retrieve → generate answer |
| GET | `/health` | Health check |

### Upload Request
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

### Upload Response
```json
{
  "session_id": "abc-123",
  "filename": "document.pdf",
  "file_type": "PDF",
  "chunks_indexed": 142,
  "message": "File uploaded and indexed successfully."
}
```

### Ask Request
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "question": "What are the main conclusions?"}'
```

---

## Getting Started

### Prerequisites

- Python 3.10 or above
- Free Groq API key from console.groq.com

### Installation

```bash
# Clone the repository
git clone https://github.com/nagabhargavi12/docmind-rag-qa.git
cd docmind-rag-qa

# Install backend dependencies
cd backend
pip install -r requirements.txt
```

### Running the Application

```bash
# Terminal 1 — Start backend
cd backend
set GROQ_API_KEY=gsk_your_key_here        # Windows
# export GROQ_API_KEY=gsk_your_key_here   # Mac/Linux
uvicorn main:app --reload --port 8000

# Terminal 2 — Serve frontend
cd frontend
python -m http.server 3001
```

Open browser at `http://localhost:3001`

---

## Key Design Decisions

**Why RAG over fine-tuning?**
RAG works on any document without retraining the model. It avoids hallucination by constraining the LLM to retrieved context, is instantly updatable by re-ingesting new files, and is significantly more cost-effective than fine-tuning for document-specific Q&A.

**Why overlapping chunks?**
Chunks overlap by 80 tokens to preserve context at boundaries. Without overlap, a sentence split across two chunks would lose its meaning in retrieval, reducing answer quality.

**Why FAISS over a cloud vector database?**
FAISS runs locally with zero latency and zero cost. For this project scope, it is sufficient. In production, Pinecone or Weaviate would provide persistence, scalability, and multi-user support across server restarts.

**Why Sentence Transformers over OpenAI embeddings?**
all-MiniLM-L6-v2 is fast, lightweight (80MB), runs entirely locally with no API cost, and produces high-quality 384-dimensional embeddings suitable for semantic similarity on English text.

**Why Groq over OpenAI or Anthropic?**
Groq provides free API access to Llama 3.3 70B with extremely fast inference (tokens per second). For development and demonstration purposes, it eliminates cost barriers while delivering production-quality responses.

**Why in-memory session management?**
In-memory storage is fast and requires no database setup. The trade-off is sessions are lost on server restart. In production, sessions and vector stores would be persisted in PostgreSQL with pgvector.

---

## Potential Improvements

- Add persistent storage with PostgreSQL and pgvector for sessions and embeddings
- Implement re-ranking using a cross-encoder model to improve retrieval precision
- Add hybrid search combining BM25 keyword search with FAISS semantic search
- Support multi-document sessions — query across multiple uploaded files
- Integrate evaluation using RAGAS to measure faithfulness and answer relevance
- Add streaming responses for real-time token-by-token answer display
- Deploy backend on Railway or Render, frontend on Vercel for a live demo URL

---

## Concepts Demonstrated

- Retrieval-Augmented Generation (RAG) pipeline design
- Dense vector embeddings and cosine similarity search
- FAISS vector indexing and nearest neighbour retrieval
- Multi-format document parsing and text extraction
- Session-based stateful API design with FastAPI
- LLM prompt engineering with context injection
- Full-stack application development with REST APIs



