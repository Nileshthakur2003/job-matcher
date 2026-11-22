import argparse
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
import httpx
from sqlalchemy.dialects.postgresql import insert

from app.db import AsyncSessionLocal, Job, init_db
from app.schemas import JobNormalizationOutput
from app.services.embedding import get_embeddings
from app.services.llm import extract_structured_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")

# Known public greenhouse company boards with active listings
GREENHOUSE_COMPANIES = [
    "anthropic",
    "cloudflare",
    "github",
    "stripe",
    "canonical",
    "gitlab",
    "elastic",
    "datadog",
    "reddit",
    "automattic",
]

HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_html(raw_html: str) -> str:
    """Strip HTML tags and unescape common entities."""
    clean_text = HTML_TAG_RE.sub(" ", raw_html)
    clean_text = (
        clean_text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )
    return " ".join(clean_text.split())


async def fetch_greenhouse_jobs(client: httpx.AsyncClient, company: str, limit_per_company: int = 40) -> List[Dict[str, Any]]:
    """Fetch live jobs from Greenhouse public job board API."""
    url = f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true"
    try:
        resp = await client.get(url, timeout=12.0)
        if resp.status_code != 200:
            logger.warning(f"Greenhouse board for '{company}' returned HTTP {resp.status_code}")
            return []
        data = resp.json()
        raw_jobs = data.get("jobs", [])[:limit_per_company]
        parsed = []
        for j in raw_jobs:
            content = clean_html(j.get("content") or "")
            if not content or len(content) < 50:
                continue
            location_name = j.get("location", {}).get("name", "Unknown") if isinstance(j.get("location"), dict) else "Unknown"
            parsed.append({
                "title": j.get("title", "Software Engineer").strip(),
                "company": company.capitalize(),
                "location": location_name,
                "description": content[:5000],
                "apply_url": j.get("absolute_url") or f"https://boards.greenhouse.io/{company}/jobs/{j.get('id')}",
            })
        return parsed
    except Exception as e:
        logger.warning(f"Error fetching Greenhouse jobs for '{company}': {e}")
        return []


async def fetch_remoteok_jobs(client: httpx.AsyncClient, limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch live tech jobs from RemoteOK public API."""
    url = "https://remoteok.com/api"
    try:
        headers = {"User-Agent": "AIJobMatcherBot/1.0 (Academic & Proof of Work Project)"}
        resp = await client.get(url, headers=headers, timeout=15.0)
        if resp.status_code != 200:
            return []
        data = resp.json()
        jobs = []
        # first item in RemoteOK is legal notice
        for item in data[1:limit + 1]:
            if not isinstance(item, dict):
                continue
            desc = clean_html(item.get("description", ""))
            if not desc or len(desc) < 50:
                continue
            jobs.append({
                "title": item.get("position", "Software Engineer").strip(),
                "company": item.get("company", "Tech Company").strip(),
                "location": item.get("location", "Remote"),
                "description": desc[:5000],
                "apply_url": item.get("url") or f"https://remoteok.com/l/{item.get('id')}",
            })
        return jobs
    except Exception as e:
        logger.warning(f"Error fetching RemoteOK jobs: {e}")
        return []


def generate_seed_dataset(count: int = 50) -> List[Dict[str, Any]]:
    """Generate high-signal realistic seed jobs for instant offline testing and verification."""
    roles = [
        ("AI Engineer", "Develop LLM applications, RAG pipelines, pgvector search, and fine-tuning with Python, LangChain, and FastAPI."),
        ("Senior Backend Engineer", "Build distributed microservices with async Python, PostgreSQL, Redis, Docker, and Kafka on AWS."),
        ("Machine Learning Engineer", "Deploy PyTorch and HuggingFace models to production via Triton, Kubernetes, and FastAPI."),
        ("Full Stack AI Engineer", "Build full-stack applications with Next.js, TypeScript, Tailwind, Python FastAPI, and OpenAI APIs."),
        ("AI Software Engineer Intern", "Assist in training evaluation pipelines, dataset curation, and building RAG workflows in Python and PyTorch."),
        ("Data Engineer", "Build high-throughput data processing pipelines using Spark, dbt, Snowflake, and PostgreSQL."),
        ("DevOps / MLOps Engineer", "Manage Kubernetes clusters, CI/CD with GitHub Actions, model observability with Prometheus, and Docker."),
        ("NLP Research Scientist", "Conduct research on model distillation, cross-encoder rerankers, semantic search, and transformer architectures."),
    ]
    companies = ["Scale AI", "OpenAI", "Anthropic", "Mistral", "Cohere", "Datadog", "Cloudflare", "Supabase", "Pinecone", "Vercel"]
    locations = ["Remote", "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX"]

    jobs = []
    for i in range(count):
        role_title, role_desc = roles[i % len(roles)]
        company = companies[(i * 3) % len(companies)]
        loc = locations[i % len(locations)]
        is_remote = "Remote" in loc or (i % 2 == 0)
        jobs.append({
            "title": f"{role_title} #{i+1}",
            "company": company,
            "location": loc if not is_remote else "Remote",
            "description": f"At {company}, we are looking for a talented {role_title}. {role_desc} Requirements: Strong proficiency in Python, PostgreSQL, REST APIs, Git, and automated testing.",
            "apply_url": f"https://careers.{company.lower().replace(' ', '')}.com/jobs/{1000 + i}",
        })
    return jobs


# -------------------------------------------------------------
# Normalization Pipeline
# -------------------------------------------------------------
SYSTEM_NORMALIZATION_PROMPT = """You are an expert AI talent recruiter.
Your job is to analyze the job posting text and return a strictly structured JSON output:
1. is_remote: Boolean (true if fully remote or remote-first)
2. role_type: One of ["full-time", "internship", "contract", "part-time"]
3. skills: Array of 3-10 standardized technical skills/languages/frameworks
4. seniority_level: One of ["intern", "junior", "mid", "senior", "lead", "all"]
5. cleaned_summary: A dense, 2-sentence summary highlighting the core tech stack and requirements.
"""

sem = asyncio.Semaphore(10)


async def normalize_single_job(raw_job: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single job using LLM with Pydantic structured output."""
    async with sem:
        prompt = f"Title: {raw_job['title']}\nCompany: {raw_job['company']}\nLocation: {raw_job['location']}\n\nDescription:\n{raw_job['description'][:2000]}"
        output: JobNormalizationOutput = await extract_structured_data(
            system_prompt=SYSTEM_NORMALIZATION_PROMPT,
            user_prompt=prompt,
            response_model=JobNormalizationOutput,
        )
        return {
            **raw_job,
            "is_remote": output.is_remote,
            "role_type": output.role_type,
            "skills": output.skills,
            "cleaned_summary": output.cleaned_summary,
        }


# -------------------------------------------------------------
# Upsert & Ingestion Orchestrator
# -------------------------------------------------------------
async def upsert_jobs_batch(jobs_with_embeddings: List[Dict[str, Any]]):
    """Upsert normalized jobs into PostgreSQL with pgvector, deduplicating on apply_url."""
    if not jobs_with_embeddings:
        return

    records = []
    for j in jobs_with_embeddings:
        records.append({
            "title": j["title"],
            "company": j["company"],
            "location": j.get("location", "Unknown"),
            "is_remote": j["is_remote"],
            "role_type": j["role_type"],
            "skills": j["skills"],
            "description": j["description"],
            "apply_url": j["apply_url"],
            "embedding": j["embedding"],
        })

    async with AsyncSessionLocal() as session:
        stmt = insert(Job).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Job.apply_url],
            set_={
                "title": stmt.excluded.title,
                "company": stmt.excluded.company,
                "location": stmt.excluded.location,
                "is_remote": stmt.excluded.is_remote,
                "role_type": stmt.excluded.role_type,
                "skills": stmt.excluded.skills,
                "description": stmt.excluded.description,
                "embedding": stmt.excluded.embedding,
            },
        )
        await session.execute(stmt)
        await session.commit()
    logger.info(f"Upserted batch of {len(records)} jobs into PostgreSQL.")


async def run_ingestion(source: str = "all", limit: int = 500, use_seed_fallback: bool = True):
    """Main ingestion coordinator."""
    logger.info("Initializing database schema, pgvector, and search vector triggers...")
    await init_db()

    raw_jobs: List[Dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        if source in ("greenhouse", "all"):
            logger.info("Fetching jobs from Greenhouse boards...")
            for comp in GREENHOUSE_COMPANIES:
                if len(raw_jobs) >= limit:
                    break
                comp_jobs = await fetch_greenhouse_jobs(client, comp, limit_per_company=25)
                raw_jobs.extend(comp_jobs)
                logger.info(f"Fetched {len(comp_jobs)} jobs from {comp}.")

        if (source in ("remoteok", "all")) and len(raw_jobs) < limit:
            logger.info("Fetching jobs from RemoteOK...")
            remote_jobs = await fetch_remoteok_jobs(client, limit=limit - len(raw_jobs))
            raw_jobs.extend(remote_jobs)
            logger.info(f"Fetched {len(remote_jobs)} jobs from RemoteOK.")

    # Deduplicate raw jobs by apply_url before processing
    seen_urls = set()
    deduped_raw: List[Dict[str, Any]] = []
    for j in raw_jobs:
        if j["apply_url"] not in seen_urls:
            seen_urls.add(j["apply_url"])
            deduped_raw.append(j)

    if not deduped_raw and use_seed_fallback:
        logger.info(f"No external jobs retrieved. Generating {limit} realistic seed jobs...")
        deduped_raw = generate_seed_dataset(count=min(limit, 100))

    deduped_raw = deduped_raw[:limit]
    logger.info(f"Total jobs to process & ingest: {len(deduped_raw)}")

    # 1. Structured Normalization via LLM (with concurrent semaphore)
    logger.info("Step 1/3: Normalizing job listings using Pydantic Structured Outputs...")
    normalization_tasks = [normalize_single_job(j) for j in deduped_raw]
    normalized_jobs = await asyncio.gather(*normalization_tasks)

    # 2. Batch Embedding Generation
    logger.info("Step 2/3: Generating 1536-dimensional vector embeddings...")
    embedding_texts = [
        f"Title: {j['title']}\nCompany: {j['company']}\nRole: {j['role_type']}\nRemote: {j['is_remote']}\nSkills: {', '.join(j['skills'])}\nSummary: {j['cleaned_summary']}"
        for j in normalized_jobs
    ]

    # Process in chunks of 50 for embedding API
    batch_size = 50
    all_embeddings: List[List[float]] = []
    for i in range(0, len(embedding_texts), batch_size):
        chunk = embedding_texts[i : i + batch_size]
        chunk_embeddings = await get_embeddings(chunk)
        all_embeddings.extend(chunk_embeddings)
        logger.info(f"Generated embeddings {len(all_embeddings)}/{len(embedding_texts)}")

    for j, emb in zip(normalized_jobs, all_embeddings):
        j["embedding"] = emb

    # 3. Database Upsert
    logger.info("Step 3/3: Upserting jobs into PostgreSQL...")
    for i in range(0, len(normalized_jobs), batch_size):
        batch = normalized_jobs[i : i + batch_size]
        await upsert_jobs_batch(batch)

    logger.info(f"Ingestion complete! Successfully stored {len(normalized_jobs)} jobs in pgvector.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest jobs into PostgreSQL with pgvector.")
    parser.add_argument("--source", type=str, default="all", choices=["all", "greenhouse", "remoteok", "seed"], help="Data source")
    parser.add_argument("--limit", type=int, default=500, help="Number of jobs to ingest")
    args = parser.parse_args()

    if args.source == "seed":
        asyncio.run(run_ingestion(source="seed", limit=args.limit, use_seed_fallback=True))
    else:
        asyncio.run(run_ingestion(source=args.source, limit=args.limit))
