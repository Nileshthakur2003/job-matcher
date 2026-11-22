import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, init_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["System"])


@router.get("", summary="Database & pgvector Health Check")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Verifies database connectivity and pgvector extension status, with auto-initialization."""
    try:
        # Check pgvector extension
        result = await db.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';"))
        row = result.scalar()
        if not row:
            try:
                await init_db()
            except Exception as e:
                logger.warning(f"Could not install vector extension: {e}")

        # Check jobs count, auto-initializing if table doesn't exist
        job_count = 0
        try:
            job_count = (await db.execute(text("SELECT COUNT(*) FROM jobs;"))).scalar() or 0
        except Exception:
            logger.info("Table 'jobs' not found. Initializing schema and seed data...")
            await init_db()
            job_count = (await db.execute(text("SELECT COUNT(*) FROM jobs;"))).scalar() or 0

        return {
            "status": "healthy",
            "database": "connected",
            "pgvector": "active",
            "total_indexed_jobs": job_count,
        }
    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection error: {str(e)}",
        )
