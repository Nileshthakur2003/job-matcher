import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.database import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: initializing database extensions and schema...")
    try:
        await init_db()
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not connect to database on startup: {e}. Check DATABASE_URL.")
    yield
    logger.info("Shutting down AI Job Matcher API.")


app = FastAPI(
    title="AI Job Matcher API",
    description="Production-grade AI-powered job matching backend engine using PostgreSQL, pgvector, hybrid search, and cross-encoder reranking.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for decoupled frontends (Vercel, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(api_router)


@app.get("/", tags=["Root"])
async def root():
    """Welcome and status endpoint."""
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
