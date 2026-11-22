import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, init_db
from app.schemas.match import MatchResponse
from app.services.pdf_service import extract_text_from_pdf_bytes
from app.services.search_service import extract_candidate_profile, search_and_rank_jobs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/match", tags=["Matching"])


@router.post("", response_model=MatchResponse, summary="Match Candidate Resume to Jobs")
async def match_resume(
    resume: UploadFile = File(..., description="PDF Resume of the applicant"),
    preferences: Optional[str] = Form(None, description="Optional search preferences e.g. 'Looking for remote AI internships'"),
    db: AsyncSession = Depends(get_db),
):
    """
    1. Extracts raw text from uploaded PDF resume
    2. Parses candidate profile using OpenAI / Gemini Structured Outputs (Pydantic v2)
    3. Runs Hybrid Retrieval (Dense vector <=> + Sparse BM25 + RRF)
    4. Applies Cross-Encoder Reranking
    5. Returns candidate profile & top 10 ranked jobs with personalized fit explanations
    """
    if not resume.filename or not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a PDF resume.",
        )

    # 1. Read & parse PDF text
    try:
        pdf_bytes = await resume.read()
        raw_text = extract_text_from_pdf_bytes(pdf_bytes)
        if not raw_text or len(raw_text.strip()) < 30:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract readable text from the uploaded PDF resume. Ensure the PDF contains selectable text (not scanned images).",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF parsing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to read PDF file: {str(e)}",
        )

    # 2. Extract Candidate Profile via LLM
    try:
        logger.info("Extracting candidate profile from resume...")
        candidate_profile = await extract_candidate_profile(raw_text, user_preferences=preferences)
    except Exception as e:
        logger.error(f"Candidate profile extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM profile extraction failed: {str(e)}",
        )

    # 3. Hybrid search + Cross-Encoder reranking
    try:
        logger.info("Executing hybrid search and reranking...")
        matches = await search_and_rank_jobs(
            db=db,
            candidate=candidate_profile,
            user_preferences=preferences,
            top_k=10,
        )
    except Exception as e:
        logger.warning(f"Search query error: {e}. Attempting self-healing init_db()...")
        try:
            await init_db()
            matches = await search_and_rank_jobs(
                db=db,
                candidate=candidate_profile,
                user_preferences=preferences,
                top_k=10,
            )
        except Exception as retry_err:
            logger.exception(f"Search failed after retry: {retry_err}")
            matches = []

    return MatchResponse(
        candidate=candidate_profile,
        total_found=len(matches),
        matches=matches,
    )
