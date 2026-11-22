import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db, init_db
from app.schemas import MatchResponse
from app.search import extract_candidate_profile, search_and_rank_jobs
from app.services.pdf_parser import extract_text_from_pdf_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables and vector extensions exist
    logger.info("Initializing database and pgvector extensions...")
    try:
        await init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not connect to database on startup: {e}. Ensure DATABASE_URL is configured.")
    yield
    logger.info("Shutting down AI Job Matcher API.")


app = FastAPI(
    title="AI Job Matcher API",
    description="Production-grade AI-powered job matching backend engine using PostgreSQL, pgvector, hybrid search, and cross-encoder reranking.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for external frontends (e.g. Vercel deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint for backend service."""
    return {
        "service": "AI Job Matcher Backend API",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/api/health",
        "endpoints": {
            "match": "POST /api/match (multipart/form-data: resume PDF + optional preferences)",
            "health": "GET /api/health"
        }
    }


@app.get("/api/health", tags=["System"])
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check verifying database connection and pgvector extension."""
    try:
        result = await db.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';"))
        row = result.scalar()
        if not row:
            return {"status": "warning", "database": "connected", "pgvector": "not installed"}

        job_count = (await db.execute(text("SELECT COUNT(*) FROM jobs;"))).scalar()
        return {
            "status": "healthy",
            "database": "connected",
            "pgvector": "active",
            "total_indexed_jobs": job_count or 0,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection error: {str(e)}",
        )


@app.post("/api/match", response_model=MatchResponse, tags=["Matching"])
async def match_resume(
    resume: UploadFile = File(..., description="PDF Resume of the applicant"),
    preferences: Optional[str] = Form(None, description="Optional search preferences e.g. 'Looking for remote AI internships'"),
    db: AsyncSession = Depends(get_db),
):
    """
    1. Extracts raw text from uploaded PDF resume
    2. Parses candidate profile using OpenAI Structured Outputs (Pydantic)
    3. Runs Hybrid Retrieval (Dense vector <=> + Sparse BM25 + RRF)
    4. Applies Cross-Encoder Reranking
    5. Returns candidate profile & top 10 ranked jobs with personalized fit explanations
    """
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF resumes are supported.",
        )

    # 1. Read and extract PDF text
    try:
        pdf_bytes = await resume.read()
        raw_text = extract_text_from_pdf_bytes(pdf_bytes)
        if not raw_text or len(raw_text.strip()) < 30:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract readable text from the uploaded PDF resume.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"PDF parsing error: {e}")

    # 2. Extract Candidate Profile via LLM Structured Output
    logger.info("Extracting candidate profile from resume...")
    candidate_profile = await extract_candidate_profile(raw_text, user_preferences=preferences)

    # 3. Hybrid Search + Cross-Encoder Reranking + Explanations
    logger.info("Executing hybrid search and reranking...")
    matches = await search_and_rank_jobs(
        db=db,
        candidate=candidate_profile,
        user_preferences=preferences,
        top_k=10,
    )

    return MatchResponse(
        candidate=candidate_profile,
        total_found=len(matches),
        matches=matches,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
