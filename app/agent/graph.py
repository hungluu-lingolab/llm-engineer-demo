"""Agentic RAG — STUB. Sẽ implement ở Buổi 6 (Agentic RAG / CRAG).

Dự kiến: dùng LangGraph để orchestrate các node (retrieve → grade → generate →
web fallback), nhưng MỖI node gọi native SDK (OpenAI/Qdrant) chứ không dùng
LangChain abstraction. Thay thế lời gọi retrieve thẳng trong pipeline.py.
"""

from __future__ import annotations


def run_agent(query: str) -> str:
    """STUB — implement ở Buổi 6."""
    raise NotImplementedError("Agentic RAG sẽ được implement ở Buổi 6.")
