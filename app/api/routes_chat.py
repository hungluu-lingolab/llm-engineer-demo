"""Chat routes — Bài 1: endpoint /chat, /chat/stream, /chat/structured.

API-first: chatbot lộ ra qua HTTP để dễ tích hợp (web, mobile, test).
Streaming dùng StreamingResponse để đẩy token real-time (Section 6).
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse
from app.llm.params import GenerationParams
from app.pipeline import answer, answer_stream, answer_structured
from app.schemas.domain import LegalAnswer

router = APIRouter(prefix="/chat", tags=["chat"])


def _params_from(req: ChatRequest) -> GenerationParams | None:
    """Dựng GenerationParams từ override trong request (nếu có)."""
    overrides = {}
    if req.temperature is not None:
        overrides["temperature"] = req.temperature
    if req.max_completion_tokens is not None:
        overrides["max_completion_tokens"] = req.max_completion_tokens
    return GenerationParams(**overrides) if overrides else None


@router.post("", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """Trả lời non-streaming."""
    text = answer(req.question, _params_from(req))
    return ChatResponse(answer=text)


@router.post("/stream")
def chat_stream_endpoint(req: ChatRequest) -> StreamingResponse:
    """Trả lời streaming (text/plain, từng token một)."""
    gen = answer_stream(req.question, _params_from(req))
    return StreamingResponse(gen, media_type="text/plain; charset=utf-8")


@router.post("/structured", response_model=LegalAnswer)
def chat_structured_endpoint(req: ChatRequest) -> LegalAnswer:
    """Trả lời structured output theo schema LegalAnswer (Section 4)."""
    return answer_structured(req.question, _params_from(req))
