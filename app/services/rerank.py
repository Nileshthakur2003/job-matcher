import logging
from typing import Any, Dict, List, Optional
from app.config import settings

logger = logging.getLogger(__name__)

_flashrank_client = None


def get_flashrank_client():
    global _flashrank_client
    if _flashrank_client is None:
        try:
            from flashrank import Ranker
            # Uses ultra-fast 12-layer MiniLM ONNX cross-encoder (~30MB)
            _flashrank_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
        except Exception as e:
            logger.warning(f"Could not initialize FlashRank: {e}")
            _flashrank_client = None
    return _flashrank_client


async def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Rerank candidate documents against the candidate query using a Cross-Encoder.
    Prioritizes FlashRank (local ONNX, zero cost, <50ms latency) or Cohere API.
    """
    if not documents:
        return []

    # Format documents for cross-encoder
    passages = []
    for doc in documents:
        # Build dense text representation for reranking
        text = f"Title: {doc.get('title')}\nCompany: {doc.get('company')}\nSkills: {', '.join(doc.get('skills', []))}\nDescription: {doc.get('description', '')[:1000]}"
        passages.append({"id": str(doc.get("id")), "text": text, "metadata": doc})

    # Option 1: Cohere Rerank API if key provided
    if settings.COHERE_API_KEY:
        try:
            import cohere
            co = cohere.Client(settings.COHERE_API_KEY)
            docs_text = [p["text"] for p in passages]
            response = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs_text,
                top_n=min(top_n, len(documents)),
            )
            reranked = []
            for item in response.results:
                matched_doc = passages[item.index]["metadata"]
                matched_doc["rerank_score"] = round(float(item.relevance_score) * 100, 2)
                reranked.append(matched_doc)
            return reranked
        except Exception as e:
            logger.warning(f"Cohere reranking failed: {e}. Falling back to FlashRank.")

    # Option 2: FlashRank Local Cross-Encoder
    flash_ranker = get_flashrank_client()
    if flash_ranker:
        try:
            from flashrank import RerankRequest
            rerank_request = RerankRequest(query=query, passages=passages)
            results = flash_ranker.rerank(rerank_request)
            reranked = []
            for item in results[:top_n]:
                matched_doc = item["metadata"]
                # Convert logit/score to percentage (0 - 100)
                score = round(float(item.get("score", 0.0)) * 100, 2)
                matched_doc["rerank_score"] = score
                reranked.append(matched_doc)
            return reranked
        except Exception as e:
            logger.warning(f"FlashRank reranking failed: {e}. Falling back to heuristic.")

    # Option 3: Fallback based on pre-computed RRF score
    sorted_docs = sorted(documents, key=lambda x: x.get("rrf_score", 0.0), reverse=True)
    for doc in sorted_docs:
        doc["rerank_score"] = round(doc.get("rrf_score", 0.5) * 100, 2)
    return sorted_docs[:top_n]
