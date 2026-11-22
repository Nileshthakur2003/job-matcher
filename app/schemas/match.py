from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.candidate import CandidateProfile
from app.schemas.job import JobResponse


class JobMatchResult(BaseModel):
    """Represents a matched job with relevance scores, skill overlaps, and fit explanation."""
    job: JobResponse
    match_score: float = Field(..., description="Overall relevance percentage (0-100%)")
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    explanation: Optional[str] = Field(None, description="Personalized 1-sentence fit explanation.")


class MatchResponse(BaseModel):
    """API response model for POST /api/match."""
    candidate: CandidateProfile
    total_found: int
    matches: List[JobMatchResult]
