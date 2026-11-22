from uuid import uuid4
from sqlalchemy import (
    Column,
    String,
    Boolean,
    Text,
    DateTime,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, TSVECTOR
from pgvector.sqlalchemy import Vector

from app.core.database import Base


class Job(Base):
    """
    SQLAlchemy ORM Model representing an ingested tech job.
    Includes pgvector dense embeddings and a PostgreSQL TSVECTOR for hybrid search.
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

    # PostgreSQL full-text search vector for BM25 keyword retrieval
    search_vector = Column(TSVECTOR, nullable=True)

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, title='{self.title}', company='{self.company}', remote={self.is_remote})>"
