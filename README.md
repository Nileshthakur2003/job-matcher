---
title: AI Job Matcher Backend API
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# AI Job Matcher — Backend API 🚀

Production-grade AI Job Matching Backend API built with **FastAPI**, **PostgreSQL (`pgvector`)**, **OpenAI Structured Outputs**, **Hybrid Retrieval (Dense + BM25)**, and **Cross-Encoder Reranking**.

---

## 🛠️ Tech Stack & Architecture

- **Web Framework:** FastAPI, Uvicorn, Pydantic v2
- **Vector Database:** PostgreSQL 16 + `pgvector` (HNSW indexing + GIN full-text index)
- **AI Extraction & Embeddings:** OpenAI `gpt-4o-mini` (Strict JSON schema) & `text-embedding-3-small` (or Gemini 1.5 Flash)
- **Reranker:** Local CPU Cross-Encoder (`flashrank` with `ms-marco-MiniLM-L-12-v2`) or Cohere Rerank API
- **PDF Parser:** PyMuPDF (`fitz`)

---

## 🚀 Quickstart: Local Setup

```bash
# 1. Start local PostgreSQL with pgvector
docker-compose up -d

# 2. Setup virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env

# 4. Ingest sample jobs
python -m app.ingest --source seed --limit 50

# 5. Run retrieval benchmark
python -m app.eval

# 6. Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive Docs: [http://localhost:8000/docs](http://localhost:8000/docs)  
Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 🌐 Deploy to Render ($0 / Free Tier)

1. Create a new GitHub repository named `ai-job-matcher-backend` and push the contents of this folder.
2. In [render.com](https://render.com), click **New +** $\to$ **Blueprint** (or **Web Service** $\to$ Build from Docker).
3. Connect your repo. Render will automatically read `render.yaml`.
4. Add your Environment Variables:
   - `DATABASE_URL`: Your Supabase or Neon PostgreSQL connection string.
   - `OPENAI_API_KEY`: Your OpenAI or Google Gemini API key.
5. Deploy! Your backend will be live at `https://your-service.onrender.com`.

---

## 🌐 Deploy to Hugging Face Spaces ($0 / Free 2 vCPU 16GB RAM)

1. In [huggingface.co/spaces](https://huggingface.co/spaces), create a new Space.
2. Select **Docker (Blank)**.
3. Push this repository to your Hugging Face space.
4. Under **Settings** $\to$ **Variables and secrets**, add `DATABASE_URL` and `OPENAI_API_KEY`.
5. Hugging Face will automatically boot your container on port 8000.
