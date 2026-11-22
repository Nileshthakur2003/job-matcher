from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field


# -------------------------------------------------------------
# LLM Extraction / Structured Output Schemas
# -------------------------------------------------------------

class JobNormalizationOutput(BaseModel):
    """Structured output schema for LLM-based job posting normalization."""
    is_remote: bool = Field(
        ...,
        description="Whether the job allows full remote work."
    )
    role_type: Literal["full-time", "internship", "contract", "part-time"] = Field(
        default="full-time",
        description="The employment type of the position."
    )
    skills: List[str] = Field(
        default_factory=list,
        description="Standardized technical skills, tools, and languages (e.g. ['Python', 'FastAPI', 'PostgreSQL', 'Docker'])."
    )
    seniority_level: Literal["intern", "junior", "mid", "senior", "lead", "all"] = Field(
        default="mid",
        description="The estimated seniority level required."
    )
    cleaned_summary: str = Field(
        ...,
        description="A concise 2-sentence summary of the core role and key requirements."
    )


class CandidateProfile(BaseModel):
    """Structured profile extracted from a candidate's resume + preferences."""
    candidate_name: Optional[str] = Field(None, description="Candidate full name if available.")
    target_roles: List[str] = Field(
        default_factory=list,
        description="Target job titles (e.g. ['AI Engineer', 'Backend Engineer'])."
    )
    skills: List[str] = Field(
        default_factory=list,
        description="Core technical skills extracted from experience and projects."
    )
    preferred_locations: List[str] = Field(
        default_factory=list,
        description="Preferred job locations extracted from resume or user preferences."
    )
    is_remote_only: bool = Field(
        default=False,
        description="Whether the candidate strictly requires remote opportunities."
    )
    seniority_level: str = Field(
        default="mid",
        description="Inferred candidate seniority level (e.g. 'intern', 'junior', 'mid', 'senior')."
    )
    candidate_summary: str = Field(
        ...,
        description="Dense, high-signal 2-3 sentence technical summary capturing candidate's stack and achievements."
    )


# -------------------------------------------------------------
# Database & API Schemas
# -------------------------------------------------------------

class JobBase(BaseModel):
    title: str
    company: str
    location: str = "Unknown"
    is_remote: bool = False
    role_type: str = "full-time"
    skills: List[str] = []
    description: str
    apply_url: str


class JobCreate(JobBase):
    embedding: Optional[List[float]] = None


class JobResponse(JobBase):
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class JobMatchResult(BaseModel):
    job: JobResponse
    match_score: float = Field(..., description="Overall relevance score (0-100%)")
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    explanation: Optional[str] = Field(None, description="Personalized 1-sentence fit explanation.")


class MatchResponse(BaseModel):
    candidate: CandidateProfile
    total_found: int
    matches: List[JobMatchResult]
