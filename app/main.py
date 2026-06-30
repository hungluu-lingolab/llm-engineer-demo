"""FastAPI app entry — Vietnamese Legal Assistant (Module I).

Chạy dev server:
    uvicorn app.main:app --reload

Mở docs tương tác: http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api import routes_chat
from app.config import settings

app = FastAPI(
    title=settings.app_name,
    description="RAG chatbot pháp lý — xây dựng xuyên suốt Module I (LLM Engineer).",
    version="0.1.0",
)

app.include_router(routes_chat.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Health check — không gọi LLM, không cần API key."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "model": settings.llm_model,
        "keys_configured": len(settings.api_keys),
    }
