import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job import Job
from app.schemas.candidate import CandidateProfile
from app.schemas.job import JobResponse
from app.schemas.match import JobMatchResult
from app.services.embedding_service import get_single_embedding
from app.services.llm_service import extract_structured_data, get_openai_client
from app.services.rerank_service import rerank_documents

logger = logging.getLogger(__name__)

CANDIDATE_EXTRACTION_PROMPT = """You are an expert technical talent evaluator.
Analyze the candidate's resume text and any additional user preferences provided.
Extract a structured candidate profile:
1. candidate_name: Candidate's full name if found
2. target_roles: 1-3 targeted job titles (e.g. ['AI Engineer', 'Backend Engineer'])
3. skills: Comprehensive list of extracted technical skills, frameworks, languages, tools
4. preferred_locations: Locations mentioned or preferred
5. is_remote_only: Boolean (true if candidate or preferences explicitly specify remote-only)
6. seniority_level: One of ['intern', 'junior', 'mid', 'senior', 'lead']
7. candidate_summary: A dense, high-signal 2-3 sentence technical summary capturing candidate's stack, project highlights, and strengths.
"""

EXPLANATION_PROMPT = """You are an expert AI career advisor.
Given a candidate's profile and a target job listing, write EXACTLY ONE clear, compelling sentence explaining why this candidate is a strong fit for this role.
Highlight specific overlapping skills, projects, or background.
"""


async def extract_candidate_profile(
    resume_text: str,
    user_preferences: Optional[str] = None,
) -> CandidateProfile:
    """Extract structured candidate profile from raw resume text + optional preferences."""
    prompt = f"Resume Content:\n{resume_text[:4000]}"
    if user_preferences:
        prompt += f"\n\nUser Additional Preferences / Search Query:\n{user_preferences}"

    return await extract_structured_data(
        system_prompt=CANDIDATE_EXTRACTION_PROMPT,
        user_prompt=prompt,
        response_model=CandidateProfile,
    )


async def generate_fit_explanation(candidate: CandidateProfile, job_title: str, company: str, job_skills: List[str]) -> str:
    """Generate a single-sentence personalized match explanation."""
    client = get_openai_client()
    if client:
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EXPLANATION_PROMPT},
                    {
                        "role": "user",
                        "content": f"Candidate: {candidate.candidate_summary}\nCandidate Skills: {', '.join(candidate.skills)}\n\nJob: {job_title} at {company}\nRequired Skills: {', '.join(job_skills)}",
                    },
                ],
                max_tokens=60,
                temperature=0.3,
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                return content.strip()
        except Exception as e:
            logger.warning(f"Fit explanation generation failed: {e}")

    common = set(s.lower() for s in candidate.skills).intersection(set(s.lower() for s in job_skills))
    if common:
        return f"Strong match due to your hands-on experience with {', '.join(list(common)[:3])}."
    return f"Well-aligned with your target role in {job_title}."


async def hybrid_search_jobs(
    db: AsyncSession,
    candidate: CandidateProfile,
    user_preferences: Optional[str] = None,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """
    Executes Hybrid Search:
    1. SQL metadata filters (is_remote, role_type)
    2. Dense vector cosine similarity (<=>)
    3. Sparse BM25 / full-text search (ts_rank_cd)
    4. Reciprocal Rank Fusion (RRF) to merge and select Top 30
    """
    search_query = f"{candidate.candidate_summary} {' '.join(candidate.skills[:8])} {' '.join(candidate.target_roles)}"
    if user_preferences:
        search_query += f" {user_preferences}"

    query_embedding = await get_single_embedding(search_query)

    # Sanitize keywords for full text search (remove punctuation that could break tsquery)
    raw_keywords = list(candidate.skills[:6]) + candidate.target_roles[:2]
    if user_preferences:
        raw_keywords.extend(user_preferences.split()[:4])

    sanitized_words = [re.sub(r"[^\w]", "", k) for k in raw_keywords]
    keyword_str = " ".join([w for w in sanitized_words if len(w) > 1]) or "Software Engineer"

    filters = [Job.embedding.isnot(None)]
    if candidate.is_remote_only:
        filters.append(Job.is_remote == True)

    if candidate.seniority_level == "intern" or (user_preferences and "intern" in user_preferences.lower()):
        filters.append(Job.role_type == "internship")

    try:
        # Dense Search
        dense_stmt = (
            select(Job, Job.embedding.cosine_distance(query_embedding).label("distance"))
            .where(*filters)
            .order_by(Job.embedding.cosine_distance(query_embedding))
            .limit(50)
        )
        dense_results = (await db.execute(dense_stmt)).all()
    except Exception as e:
        logger.warning(f"Dense search query failed: {e}")
        dense_results = []

    try:
        # Sparse BM25 Search
        sparse_query = func.websearch_to_tsquery("english", keyword_str)
        sparse_stmt = (
            select(Job, func.ts_rank_cd(Job.search_vector, sparse_query).label("rank"))
            .where(*filters)
            .where(Job.search_vector.op("@@")(sparse_query))
            .order_by(desc("rank"))
            .limit(50)
        )
        sparse_results = (await db.execute(sparse_stmt)).all()
    except Exception as e:
        logger.warning(f"Sparse search query failed: {e}")
        sparse_results = []

    # Reciprocal Rank Fusion (RRF)
    K = 60
    job_scores: Dict[str, Dict[str, Any]] = {}

    for rank, (job, dist) in enumerate(dense_results, start=1):
        job_id = str(job.id)
        if job_id not in job_scores:
            job_scores[job_id] = {"job": job, "dense_rank": rank, "sparse_rank": None, "rrf_score": 0.0}
        job_scores[job_id]["dense_rank"] = rank
        job_scores[job_id]["rrf_score"] += 1.0 / (K + rank)

    for rank, (job, bm25_rank) in enumerate(sparse_results, start=1):
        job_id = str(job.id)
        if job_id not in job_scores:
            job_scores[job_id] = {"job": job, "dense_rank": None, "sparse_rank": rank, "rrf_score": 0.0}
        job_scores[job_id]["sparse_rank"] = rank
        job_scores[job_id]["rrf_score"] += 1.0 / (K + rank)

    sorted_candidates = sorted(job_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    top_candidates = sorted_candidates[:limit]

    payload = []
    for item in top_candidates:
        job: Job = item["job"]
        payload.append({
            "id": job.id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "is_remote": job.is_remote,
            "role_type": job.role_type,
            "skills": job.skills or [],
            "description": job.description,
            "apply_url": job.apply_url,
            "created_at": job.created_at,
            "dense_rank": item["dense_rank"],
            "sparse_rank": item["sparse_rank"],
            "rrf_score": item["rrf_score"],
        })

    return payload


async def search_and_rank_jobs(
    db: AsyncSession,
    candidate: CandidateProfile,
    user_preferences: Optional[str] = None,
    top_k: int = 10,
) -> List[JobMatchResult]:
    """
    End-to-end matching pipeline:
    1. Hybrid Search -> Top 30
    2. Cross-Encoder Reranker -> Top 10
    3. Personalized Explanations -> Top 5
    """
    candidates = await hybrid_search_jobs(db, candidate, user_preferences, limit=30)
    if not candidates:
        return []

    rerank_query = f"{candidate.candidate_summary} Required skills: {', '.join(candidate.skills)}"
    reranked = await rerank_documents(query=rerank_query, documents=candidates, top_n=top_k)

    candidate_skills_lower = {s.lower() for s in candidate.skills}
    results: List[JobMatchResult] = []
    explanation_tasks = []

    for idx, item in enumerate(reranked):
        job_skills = item.get("skills", [])
        if idx < 5:
            explanation_tasks.append(
                generate_fit_explanation(candidate, item["title"], item["company"], job_skills)
            )
        else:
            explanation_tasks.append(asyncio.sleep(0, result=None))

    explanations = await asyncio.gather(*explanation_tasks)

    for item, exp in zip(reranked, explanations):
        job_skills = item.get("skills", [])
        matched = [s for s in job_skills if s.lower() in candidate_skills_lower]
        missing = [s for s in job_skills if s.lower() not in candidate_skills_lower]

        job_response = JobResponse(
            id=item["id"],
            title=item["title"],
            company=item["company"],
            location=item["location"],
            is_remote=item["is_remote"],
            role_type=item["role_type"],
            skills=job_skills,
            description=item["description"],
            apply_url=item["apply_url"],
            created_at=item["created_at"],
        )

        results.append(
            JobMatchResult(
                job=job_response,
                match_score=item.get("rerank_score", 85.0),
                dense_rank=item.get("dense_rank"),
                sparse_rank=item.get("sparse_rank"),
                rrf_score=item.get("rrf_score"),
                matched_skills=matched,
                missing_skills=missing[:5],
                explanation=exp,
            )
        )

    return results
