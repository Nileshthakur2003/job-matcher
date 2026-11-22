import logging
import re
from typing import Type, TypeVar, Optional
from pydantic import BaseModel
from openai import AsyncOpenAI

from app.config import settings
from app.schemas import JobNormalizationOutput, CandidateProfile

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> Optional[AsyncOpenAI]:
    global _client
    if _client is None and settings.OPENAI_API_KEY:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


async def extract_structured_data(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    temperature: float = 0.0,
) -> T:
    """
    Extract structured data using OpenAI Structured Outputs (Pydantic).
    Falls back to heuristic extraction if OPENAI_API_KEY is not set.
    """
    client = get_openai_client()

    if client:
        try:
            response = await client.beta.chat.completions.parse(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_model,
                temperature=temperature,
            )
            parsed = response.choices[0].message.parsed
            if parsed:
                return parsed
        except Exception as e:
            logger.warning(f"OpenAI structured output call failed: {e}. Falling back to heuristic.")

    # Fallback heuristic parser if API key is not configured or fails
    if response_model == JobNormalizationOutput:
        return _fallback_normalize_job(user_prompt)  # type: ignore
    elif response_model == CandidateProfile:
        return _fallback_parse_resume(user_prompt)  # type: ignore

    raise ValueError(f"No fallback available for response model {response_model}")


def _fallback_normalize_job(raw_text: str) -> JobNormalizationOutput:
    """Heuristic fallback for job normalization when running offline or without an API key."""
    text_lower = raw_text.lower()

    # Remote detection
    is_remote = any(k in text_lower for k in ["remote", "work from home", "anywhere", "wfh", "distributed"])

    # Role type detection
    role_type = "full-time"
    if any(k in text_lower for k in ["intern", "internship", "co-op"]):
        role_type = "internship"
    elif any(k in text_lower for k in ["contract", "freelance", "contractor"]):
        role_type = "contract"
    elif "part-time" in text_lower:
        role_type = "part-time"

    # Common technical skill keywords
    known_skills = [
        "Python", "FastAPI", "PostgreSQL", "pgvector", "Docker", "Kubernetes", "AWS",
        "GCP", "PyTorch", "TensorFlow", "React", "Next.js", "TypeScript", "JavaScript",
        "OpenAI", "LangChain", "LlamaIndex", "SQLAlchemy", "Git", "Redis", "Kafka",
        "Go", "Rust", "C++", "Java", "Scikit-Learn", "Hugging Face", "RAG", "LLM"
    ]
    extracted_skills = [skill for skill in known_skills if re.search(rf"\b{re.escape(skill.lower())}\b", text_lower)]
    if not extracted_skills:
        extracted_skills = ["Software Engineering", "Python", "Backend"]

    # Seniority level
    seniority = "mid"
    if any(k in text_lower for k in ["senior", "sr.", "lead", "principal", "staff"]):
        seniority = "senior"
    elif any(k in text_lower for k in ["junior", "jr.", "entry", "intern", "associate"]):
        seniority = "junior"

    summary = raw_text[:250].replace("\n", " ").strip() + "..."

    return JobNormalizationOutput(
        is_remote=is_remote,
        role_type=role_type,  # type: ignore
        skills=extracted_skills,
        seniority_level=seniority,  # type: ignore
        cleaned_summary=summary,
    )


def _fallback_parse_resume(raw_text: str) -> CandidateProfile:
    """Heuristic fallback for resume extraction when running offline or without an API key."""
    text_lower = raw_text.lower()

    known_skills = [
        "Python", "FastAPI", "PostgreSQL", "pgvector", "Docker", "PyTorch", "Kubernetes",
        "AWS", "GCP", "React", "TypeScript", "OpenAI", "LangChain", "Redis", "Git", "SQL"
    ]
    skills = [skill for skill in known_skills if re.search(rf"\b{re.escape(skill.lower())}\b", text_lower)]
    if not skills:
        skills = ["Python", "FastAPI", "Backend"]

    is_remote = "remote" in text_lower

    return CandidateProfile(
        candidate_name="Applicant",
        target_roles=["AI Engineer", "Backend Software Engineer"],
        skills=skills,
        preferred_locations=["Remote", "San Francisco, CA", "New York, NY"],
        is_remote_only=is_remote,
        seniority_level="mid",
        candidate_summary=raw_text[:280].replace("\n", " ").strip() + "..."
    )
