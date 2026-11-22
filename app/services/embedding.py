import hashlib
import logging
import math
from typing import List, Optional
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> Optional[AsyncOpenAI]:
    global _client
    if _client is None and settings.OPENAI_API_KEY:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate 1536-dimensional embeddings for a batch of texts using text-embedding-3-small.
    Falls back to deterministic normalized pseudo-embeddings if OPENAI_API_KEY is not set.
    """
    if not texts:
        return []

    client = get_openai_client()

    if client:
        try:
            # Clean and truncate inputs to avoid token overflow
            cleaned_texts = [t.replace("\n", " ").strip()[:8000] for t in texts]
            response = await client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=cleaned_texts,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}. Falling back to deterministic vector.")

    # Offline / Mock Fallback: 1536-dimensional deterministic pseudo-embedding
    return [_generate_pseudo_embedding(t) for t in texts]


async def get_single_embedding(text: str) -> List[float]:
    """Helper for a single text embedding."""
    results = await get_embeddings([text])
    return results[0]


def _generate_pseudo_embedding(text: str, dim: int = 1536) -> List[float]:
    """Generates a deterministic unit-length vector from text hash for offline testing."""
    vec = []
    # Seed values based on MD5 and SHA256 hashes of text
    h = hashlib.sha256(text.encode("utf-8")).digest()
    for i in range(dim):
        byte_val = h[i % len(h)]
        val = (byte_val / 128.0) - 1.0 + math.sin(i + (byte_val % 7))
        vec.append(val)

    # Normalize to unit length (L2 norm = 1.0)
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec
