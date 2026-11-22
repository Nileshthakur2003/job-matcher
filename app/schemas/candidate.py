from typing import List, Optional
from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    """Structured candidate profile extracted from resume PDF + optional search preferences."""
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
        description="Inferred candidate seniority level ('intern', 'junior', 'mid', 'senior')."
    )
    candidate_summary: str = Field(
        ...,
        description="Dense, high-signal 2-3 sentence technical summary capturing candidate's stack and strengths."
    )
