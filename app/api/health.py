from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

router = APIRouter(prefix="/health", tags=["System"])


@router.get("", summary="Database & pgvector Health Check")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Verifies database connectivity and pgvector extension status."""
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
