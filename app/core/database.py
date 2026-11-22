import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

from app.core.config import settings

logger = logging.getLogger(__name__)

# Configure SSL for remote cloud databases (Supabase, Neon, etc.)
connect_args = {}
if settings.is_cloud_db:
    connect_args["ssl"] = "require"

engine = create_async_engine(
    settings.sanitized_database_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    connect_args=connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

Base = declarative_base()


async def get_db():
    """FastAPI dependency yielding an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """
    Initializes PostgreSQL schema:
    1. Installs 'vector' extension
    2. Builds ORM tables (importing Job model so Base.metadata is populated)
    3. Builds HNSW index on embeddings for fast cosine similarity (<=>)
    4. Builds GIN index on tsvector for keyword retrieval
    5. Sets up trigger for weighted tsvector updates (Title=A, Skills/Company=B, Desc=C)
    6. Auto-seeds initial jobs if database is empty
    """
    # CRITICAL: Import model so Base.metadata knows about the 'jobs' table
    from app.models.job import Job  # noqa: F401

    async with engine.begin() as conn:
        # 1. Enable pgvector
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # 2. Create tables
        await conn.run_sync(Base.metadata.create_all)

        # 3. Create HNSW index
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_jobs_embedding_hnsw 
            ON jobs USING hnsw (embedding vector_cosine_ops);
        """))

        # 4. Create GIN index on tsvector
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_jobs_search_vector 
            ON jobs USING gin (search_vector);
        """))

        # 5. Create weighted search_vector trigger function & trigger
        await conn.execute(text("""
            CREATE OR REPLACE FUNCTION jobs_search_vector_update() RETURNS trigger AS $$
            BEGIN
                NEW.search_vector :=
                    setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(NEW.company, '')), 'B') ||
                    setweight(to_tsvector('english', coalesce(array_to_string(NEW.skills, ' '), '')), 'B') ||
                    setweight(to_tsvector('english', coalesce(NEW.description, '')), 'C');
                RETURN NEW;
            END
            $$ LANGUAGE plpgsql;

            DROP TRIGGER IF EXISTS trg_jobs_search_vector_update ON jobs;
            CREATE TRIGGER trg_jobs_search_vector_update
            BEFORE INSERT OR UPDATE ON jobs
            FOR EACH ROW EXECUTE FUNCTION jobs_search_vector_update();
        """))

    # 6. Auto-seed if table is empty
    try:
        await auto_seed_if_empty()
    except Exception as e:
        logger.warning(f"Auto-seeding check skipped: {e}")


async def auto_seed_if_empty():
    """Seeds initial jobs if the database table has 0 rows."""
    from app.services.ingest_service import generate_seed_dataset, upsert_jobs_batch
    from app.services.embedding_service import get_embeddings

    async with AsyncSessionLocal() as session:
        count = (await session.execute(text("SELECT COUNT(*) FROM jobs;"))).scalar() or 0
        if count == 0:
            logger.info("Database is empty. Auto-seeding 25 sample tech & AI jobs...")
            seeds = generate_seed_dataset(count=25)
            for j in seeds:
                j["is_remote"] = "Remote" in j.get("location", "")
                j["role_type"] = "internship" if "intern" in j["title"].lower() else "full-time"
                j["skills"] = ["Python", "FastAPI", "PostgreSQL", "Docker", "PyTorch"]
                j["cleaned_summary"] = j["description"][:200]

            texts = [f"Title: {j['title']}\nCompany: {j['company']}\nDescription: {j['description']}" for j in seeds]
            embeddings = await get_embeddings(texts)
            for j, emb in zip(seeds, embeddings):
                j["embedding"] = emb

            await upsert_jobs_batch(seeds)
            logger.info("Auto-seeded 25 initial jobs successfully.")
