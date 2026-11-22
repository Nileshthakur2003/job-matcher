from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class JobNormalizationOutput(BaseModel):
    """Schema for LLM-based job posting normalization via Pydantic Structured Outputs."""
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
        description="Standardized technical skills (e.g. ['Python', 'FastAPI', 'PostgreSQL', 'Docker'])."
    )
    seniority_level: Literal["intern", "junior", "mid", "senior", "lead", "all"] = Field(
        default="mid",
        description="The estimated seniority level required."
    )
    cleaned_summary: str = Field(
        ...,
        description="A concise 2-sentence summary of the core role and key requirements."
    )


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
