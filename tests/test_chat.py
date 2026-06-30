"""Test Buổi 1 — không gọi API thật (mock LLM) nên chạy được không cần key.

Mục tiêu: kiểm tra wiring của pipeline + API layer, và xác nhận retriever là stub.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.retrieval.retriever import retrieve

client = TestClient(app)


def test_health_ok():
    """Health check chạy không cần API key."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_retriever_is_stub_in_lesson1():
    """Điểm sư phạm Buổi 1: retriever luôn trả rỗng (chưa có RAG)."""
    assert retrieve("Công ty TNHH có tối đa bao nhiêu thành viên?") == []


def test_chat_endpoint_wiring(monkeypatch):
    """Endpoint /chat gọi đúng pipeline. Mock LLM để không tốn API."""
    from app import pipeline

    monkeypatch.setattr(pipeline.completion, "chat", lambda messages, params: "Trả lời demo.")

    r = client.post("/chat", json={"question": "Xin chào"})
    assert r.status_code == 200
    assert r.json()["answer"] == "Trả lời demo."


def test_chat_validation_rejects_empty_question():
    """Pydantic chặn câu hỏi rỗng (min_length=1)."""
    r = client.post("/chat", json={"question": ""})
    assert r.status_code == 422
