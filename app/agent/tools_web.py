"""Web search fallback — Buổi 6 (CRAG). NATIVE Tavily SDK (không LangChain wrapper).

Dùng khi retrieval từ Qdrant không đủ tốt (grade_documents node quyết định).
Trả về RetrievedChunk để tái sử dụng đúng type với phần còn lại của pipeline
(prompts.build_messages đã biết render RetrievedChunk).
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.retrieval.retriever import RetrievedChunk


@lru_cache(maxsize=1)
def _get_client():
    from tavily import TavilyClient

    if not settings.tavily_api_key:
        raise ValueError(
            "Web search fallback cần TAVILY_API_KEY. Đặt trong .env "
            "(lấy free key tại https://tavily.com)."
        )
    return TavilyClient(api_key=settings.tavily_api_key)


def web_search(query: str, max_results: int | None = None) -> list[RetrievedChunk]:
    """Tìm kiếm web qua Tavily, trả về dạng RetrievedChunk (source = URL)."""
    max_results = max_results or settings.agent_web_search_results
    response = _get_client().search(query=query, max_results=max_results)

    return [
        RetrievedChunk(
            text=r.get("content", ""),
            source=r.get("url", "web"),
            score=r.get("score", 0.0),
            metadata={"title": r.get("title", ""), "origin": "web_search"},
        )
        for r in response.get("results", [])
    ]
