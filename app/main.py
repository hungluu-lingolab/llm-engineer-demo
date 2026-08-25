"""FastAPI app entry — Vietnamese Legal Assistant (Module I).

Chạy dev server:
    uvicorn app.main:app --reload

Mở docs tương tác: http://localhost:8000/docs
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes_admin, routes_assistant, routes_chat, routes_multi_agent
from app.config import settings
from app.guardrails.checks import GuardrailViolation

app = FastAPI(
    title=settings.app_name,
    description="RAG chatbot pháp lý (Module I) + Personal Assistant agent (Module II).",
    version="0.1.0",
)

app.include_router(routes_chat.router)
app.include_router(routes_admin.router)
app.include_router(routes_assistant.router)
app.include_router(routes_multi_agent.router)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", tags=["meta"])
def chat_ui() -> FileResponse:
    """Demo UI — chat đơn giản, lịch sử lưu ở localStorage (xem app/static/chat.html)."""
    return FileResponse(_STATIC_DIR / "chat.html")


@app.exception_handler(GuardrailViolation)
def guardrail_violation_handler(request: Request, exc: GuardrailViolation) -> JSONResponse:
    """Input bị guardrails chặn (Buổi 7) → HTTP 400 với lý do rõ ràng."""
    return JSONResponse(
        status_code=400,
        content={"error": "input_rejected", "reason": exc.reason, "details": exc.details},
    )


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Health check — không gọi LLM, không cần API key."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "model": settings.llm_model,
        "keys_configured": len(settings.api_keys),
    }
