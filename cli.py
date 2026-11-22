import argparse
import asyncio
import logging
import time
from typing import List

from app.core.database import AsyncSessionLocal, init_db
from app.schemas.candidate import CandidateProfile
from app.services.ingest_service import execute_ingestion
from app.services.search_service import search_and_rank_jobs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cli")

SAMPLE_PROFILES: List[CandidateProfile] = [
    CandidateProfile(
        candidate_name="Alex Rivera",
        target_roles=["AI Engineer", "LLM Systems Engineer"],
        skills=["Python", "FastAPI", "PostgreSQL", "pgvector", "PyTorch", "LangChain", "Docker"],
        preferred_locations=["Remote", "San Francisco, CA"],
        is_remote_only=True,
        seniority_level="mid",
        candidate_summary="Experienced in designing production RAG pipelines, pgvector semantic search, and fine-tuning open-source LLMs with FastAPI microservices.",
    ),
    CandidateProfile(
        candidate_name="Sarah Chen",
        target_roles=["Senior Backend Engineer"],
        skills=["Python", "Go", "PostgreSQL", "Redis", "Kafka", "Docker", "Kubernetes", "AWS"],
        preferred_locations=["Remote", "Seattle, WA"],
        is_remote_only=False,
        seniority_level="senior",
        candidate_summary="Backend architect with 6 years scaling asynchronous microservices, high-throughput message brokers, and relational databases.",
    ),
    CandidateProfile(
        candidate_name="David Patel",
        target_roles=["AI Software Engineer Intern"],
        skills=["Python", "PyTorch", "NumPy", "Scikit-Learn", "FastAPI", "Git"],
        preferred_locations=["Remote"],
        is_remote_only=True,
        seniority_level="intern",
        candidate_summary="Computer Science student passionate about machine learning research, automated evaluations, and building prototype LLM applications.",
    ),
    CandidateProfile(
        candidate_name="Elena Rostova",
        target_roles=["Full Stack AI Engineer"],
        skills=["React", "Next.js", "TypeScript", "Tailwind", "Python", "FastAPI", "OpenAI"],
        preferred_locations=["San Francisco, CA", "New York, NY"],
        is_remote_only=False,
        seniority_level="mid",
        candidate_summary="Full stack product engineer specializing in modern React/Next.js interfaces connected to generative AI and LLM APIs.",
    ),
    CandidateProfile(
        candidate_name="Marcus Vance",
        target_roles=["MLOps / DevOps Engineer"],
        skills=["Kubernetes", "Docker", "Terraform", "GitHub Actions", "Prometheus", "Python", "AWS"],
        preferred_locations=["Remote"],
        is_remote_only=True,
        seniority_level="senior",
        candidate_summary="DevOps specialist automating machine learning infrastructure, model monitoring, scalable GPU deployments, and CI/CD pipelines.",
    ),
]


async def run_benchmark():
    """Runs latency and relevance benchmarks across 5 synthetic candidate profiles."""
    logger.info("Initializing DB connection for evaluation...")
    await init_db()

    print("\n" + "=" * 90)
    print(f"{'CANDIDATE / TARGET ROLE':<30} | {'LATENCY':<10} | {'TOP MATCHED JOB':<30} | {'SCORE':<7} | {'SKILLS MATCHED'}")
    print("=" * 90)

    total_latency = 0.0

    async with AsyncSessionLocal() as session:
        for profile in SAMPLE_PROFILES:
            start_time = time.perf_counter()

            matches = await search_and_rank_jobs(
                db=session,
                candidate=profile,
                user_preferences="Remote preferred" if profile.is_remote_only else None,
                top_k=5,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            total_latency += elapsed_ms

            if matches:
                top = matches[0]
                matched_skills_str = ", ".join(top.matched_skills[:3]) or "General Fit"
                top_job_str = f"{top.job.title[:20]} @ {top.job.company[:8]}"
                print(
                    f"{profile.target_roles[0][:28]:<30} | {elapsed_ms:>7.1f} ms | {top_job_str:<30} | {top.match_score:>5.1f}% | {matched_skills_str}"
                )
            else:
                print(f"{profile.target_roles[0][:28]:<30} | {elapsed_ms:>7.1f} ms | {'No jobs found':<30} | {'N/A':<7} | N/A")

    avg_latency = total_latency / len(SAMPLE_PROFILES)
    print("=" * 90)
    print(f"Evaluation Complete! Average Retrieval & Rerank Latency: {avg_latency:.2f} ms")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AI Job Matcher CLI Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest job listings into pgvector")
    ingest_parser.add_argument("--source", type=str, default="all", choices=["all", "greenhouse", "remoteok", "seed"])
    ingest_parser.add_argument("--limit", type=int, default=100)

    # Eval command
    subparsers.add_parser("eval", help="Run retrieval latency and relevance benchmark")

    # Init DB command
    subparsers.add_parser("init-db", help="Initialize PostgreSQL schema, vector extension, and triggers")

    args = parser.parse_args()

    if args.command == "ingest":
        asyncio.run(execute_ingestion(source=args.source, limit=args.limit))
    elif args.command == "eval":
        asyncio.run(run_benchmark())
    elif args.command == "init-db":
        asyncio.run(init_db())
        print("Database schema and extensions initialized successfully.")


if __name__ == "__main__":
    main()
