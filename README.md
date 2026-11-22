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

Production-ready, layered AI Job Matching Backend API built with **FastAPI**, **PostgreSQL (`pgvector`)**, **OpenAI Structured Outputs**, **Hybrid Retrieval (Dense + BM25)**, and **Cross-Encoder Reranking**.

---

## 🏛️ Clean Architecture Breakdown

```
backend/
├── app/
│   ├── main.py                 # FastAPI application factory, middleware & router inclusion
│   │
│   ├── core/                   # Application-wide singletons & lifecycle
│   │   ├── config.py           # Pydantic Settings & Neon/Supabase URL sanitization
│   │   └── database.py         # Async SQLAlchemy engine, SSL support & DDL triggers
│   │
│   ├── models/                 # Pure database entities
│   │   └── job.py              # Job model with Vector(1536) & tsvector columns
│   │
│   ├── schemas/                # Pure Pydantic v2 validation DTOs
│   │   ├── job.py              # JobBase, JobCreate, JobResponse, JobNormalizationOutput
│   │   ├── candidate.py        # CandidateProfile schema
│   │   └── match.py            # JobMatchResult, MatchResponse
│   │
│   ├── api/                    # Presentation / Controller layer
│   │   ├── router.py           # Master /api router
│   │   ├── health.py           # GET /api/health
│   │   └── match.py            # POST /api/match
│   │
│   └── services/               # Pure business & domain logic
│       ├── pdf_service.py      # PyMuPDF resume text extraction
│       ├── llm_service.py      # OpenAI/Gemini structured outputs & heuristic fallbacks
│       ├── embedding_service.py# 1536-dim vector embedding generator
│       ├── rerank_service.py   # FlashRank local ONNX cross-encoder & Cohere API
│       ├── search_service.py   # Hybrid retrieval (dense <=> + sparse BM25 + RRF)
│       └── ingest_service.py   # Multi-source scraper (Greenhouse, RemoteOK, seed) & upsert
│
├── cli.py                      # Operational CLI (ingest, benchmark eval, init-db)
├── Dockerfile                  # Container build with dynamic $PORT support
├── render.yaml                 # 1-Click Render Blueprint specification
├── docker-compose.yml          # Local PostgreSQL 16 + pgvector container
├── requirements.txt            # Pinned dependencies
├── .env.example                # Sample environment variables
├── .gitignore                  # Python ignore rules
└── README.md                   # Backend documentation & cloud deploy guide
```

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

# 4. Ingest sample jobs via CLI
python cli.py ingest --source seed --limit 50

# 5. Run retrieval & latency benchmark
python cli.py eval

# 6. Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Interactive Docs: [http://localhost:8000/docs](http://localhost:8000/docs)  
- Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 🌐 Deploy to Render ($0 / Free Tier)

1. Push this folder to a GitHub repository named `ai-job-matcher-backend`.
2. In [render.com](https://render.com), click **New +** $\to$ **Blueprint** (or **Web Service** $\to$ Docker).
3. Connect your repo. Render automatically reads `render.yaml`.
4. Set Environment Variables:
   - `DATABASE_URL`: Your Supabase or Neon PostgreSQL connection string.
   - `OPENAI_API_KEY`: Your OpenAI or Google Gemini API key.
5. Deploy! Your backend will be live at `https://your-service.onrender.com`.

---

## 🌐 Deploy to Hugging Face Spaces ($0 / Free 2 vCPU 16GB RAM)

1. In [huggingface.co/spaces](https://huggingface.co/spaces), create a new Space with **Docker (Blank)** SDK.
2. Push this repository to your Hugging Face Space.
3. Under **Settings** $\to$ **Variables and secrets**, add `DATABASE_URL` and `OPENAI_API_KEY`.
4. Hugging Face boots the container on port 8000 automatically.
