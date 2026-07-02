"""Test Buổi 2 — Local Serving backend switching.

Kiểm tra logic chọn backend + tạo client (không gọi model thật).
"""

from __future__ import annotations

import pytest

from app.llm import client as client_module
from app.llm.backends import BACKENDS, get_backend


def test_all_backends_have_valid_config():
    """3 backend openai/ollama/vllm đều khai báo đầy đủ."""
    assert set(BACKENDS) == {"openai", "ollama", "vllm"}
    assert BACKENDS["openai"].requires_real_key is True
    assert BACKENDS["ollama"].requires_real_key is False
    assert BACKENDS["vllm"].requires_real_key is False


def test_get_backend_rejects_invalid():
    with pytest.raises(ValueError, match="không hợp lệ"):
        get_backend("gemini")


def test_ollama_client_uses_local_url_and_dummy_key(monkeypatch):
    """Backend ollama: client trỏ localhost:11434, không cần key thật."""
    monkeypatch.setattr(client_module.settings, "llm_backend", "ollama")
    monkeypatch.setattr(client_module.settings, "llm_base_url", "")
    monkeypatch.setattr(client_module.settings, "openai_api_keys", "")  # không có key thật

    c = client_module.get_client()
    assert str(c.base_url).rstrip("/").endswith("11434/v1")
    assert c.api_key == "ollama"


def test_base_url_override_wins(monkeypatch):
    """LLM_BASE_URL override thắng mặc định của backend."""
    monkeypatch.setattr(client_module.settings, "llm_backend", "ollama")
    monkeypatch.setattr(client_module.settings, "llm_base_url", "http://remote-gpu:9000/v1")
    monkeypatch.setattr(client_module.settings, "openai_api_keys", "")

    c = client_module.get_client()
    assert "remote-gpu:9000" in str(c.base_url)


def test_mark_limited_is_noop_for_local(monkeypatch):
    """Local backend: mark_current_key_limited không đụng pool (không lỗi khi thiếu key)."""
    monkeypatch.setattr(client_module.settings, "llm_backend", "ollama")
    monkeypatch.setattr(client_module.settings, "openai_api_keys", "")

    c = client_module.get_client()
    # Không được raise dù pool chưa khởi tạo / không có key.
    client_module.mark_current_key_limited(c)
