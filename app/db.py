from uuid import uuid4
from sqlalchemy import (
    Column,
    String,
    Boolean,
    Text,
    DateTime,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, TSVECTOR
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

from app.config import settings

# -------------------------------------------------------------
# Cloud & SSL Connect Args (Neon / Supabase Compatibility)
# -------------------------------------------------------------
connect_args = {}
if settings.is_cloud_db:
    # asyncpg expects ssl="require" for cloud providers like Neon and Supabase
    connect_args["ssl"] = "require"

engine = create_async_engine(
    settings.sanitized_database_url,
    echo=False,
    pool_size=5,        # Optimized for Supabase/Neon free-tier connection limits
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


# -------------------------------------------------------------
# Job Database Model
# -------------------------------------------------------------
class Job(Base):
    """
    Job listing model optimized for hybrid search (dense pgvector + sparse tsvector).
    """
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    title = Column(String(255), nullable=False, index=True)
    company = Column(String(255), nullable=False, index=True)
    location = Column(String(255), default="Unknown")
    is_remote = Column(Boolean, default=False, index=True)
    role_type = Column(String(50), default="full-time", index=True)  # 'full-time', 'internship', etc.
    skills = Column(ARRAY(String), default=list)
    description = Column(Text, nullable=False)
    apply_url = Column(String(1024), unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    
    # 1536-dimensional dense embedding (OpenAI text-embedding-3-small)
    embedding = Column(Vector(1536), nullable=True)
    
    # PostgreSQL full-text search vector for BM25/keyword retrieval
    search_vector = Column(TSVECTOR, nullable=True)

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, title='{self.title}', company='{self.company}', remote={self.is_remote})>"


# -------------------------------------------------------------
# Dependency for FastAPI routes
# -------------------------------------------------------------
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# -------------------------------------------------------------
# Database Initializer (Extensions, Indexes, Triggers)
# -------------------------------------------------------------
async def init_db():
    """
    Initializes PostgreSQL database:
    1. Installs 'vector' extension
    2. Creates tables
    3. Builds HNSW index on embeddings for fast cosine similarity (<=>)
    4. Builds GIN index on tsvector for rapid full-text keyword retrieval
    5. Sets up an automated trigger for weighted tsvector updates
    """
    async with engine.begin() as conn:
        # 1. Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # 2. Create tables
        await conn.run_sync(Base.metadata.create_all)

        # 3. Create HNSW index on vector cosine distance
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_jobs_embedding_hnsw 
            ON jobs USING hnsw (embedding vector_cosine_ops);
        """))

        # 4. Create GIN index on tsvector
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_jobs_search_vector 
            ON jobs USING gin (search_vector);
        """))

        # 5. Create weighted tsvector update trigger (Title: A, Company/Skills: B, Desc: C)
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
