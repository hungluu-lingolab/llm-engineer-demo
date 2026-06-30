"""Vector store — STUB. Sẽ implement ở Buổi 4 (Qdrant client thuần).

Dự kiến: qdrant-client để upsert/search vectors. Gọi native API (không qua LangChain).
"""

from __future__ import annotations


def search(query_vector: list[float], top_k: int = 5) -> list[dict]:
    """STUB — implement ở Buổi 4."""
    raise NotImplementedError("vectorstore.search() sẽ được implement ở Buổi 4.")
