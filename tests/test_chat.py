"""Test chat endpoint + pipeline wiring — mock LLM & retrieval, không gọi API thật.

Lưu ý: từ Buổi 5, pipeline.answer() gọi retrieve() (RAG thật) trước khi chat, nên
phải mock CẢ retrieve lẫn chat để test không chạm OpenAI/Qdrant.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    """Health check chạy không cần API key."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_chat_endpoint_wiring(monkeypatch):
    """Endpoint /chat gọi đúng pipeline. Mock retrieve + LLM để không tốn API."""
    from app import pipeline

    monkeypatch.setattr(pipeline, "retrieve", lambda question: [])
    monkeypatch.setattr(pipeline.completion, "chat", lambda messages, params: "Trả lời demo.")

    r = client.post("/chat", json={"question": "Xin chào"})
    assert r.status_code == 200
    assert r.json()["answer"] == "Trả lời demo."


def test_chat_validation_rejects_empty_question():
    """Pydantic chặn câu hỏi rỗng (min_length=1)."""
    r = client.post("/chat", json={"question": ""})
    assert r.status_code == 422
