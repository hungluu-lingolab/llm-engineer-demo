"""OpenAI client factory — Bài 1 (Section 1 + 6) + Buổi 2 (Local Serving).

Bọc việc tạo client vào một chỗ. Phần còn lại của app chỉ gọi `get_client()`
mà không cần biết đang nói chuyện với OpenAI Cloud hay model local.

Bài 1 — NGUYÊN TẮC BẢO MẬT: key đọc từ env qua settings, KHÔNG hardcode/commit.
Buổi 2 — LOCAL SERVING: chọn backend qua LLM_BACKEND. Ollama/vLLM cũng dùng
OpenAI-compatible API nên chỉ khác `base_url` + `api_key` (key giả cho local).
Key rotation chỉ áp dụng khi backend cần key thật (OpenAI Cloud).
"""

from __future__ import annotations

from openai import OpenAI

from app.config import settings
from app.llm.backends import get_backend
from app.llm.resilience import RotatingKeyPool

# Khởi tạo pool một lần ở module level (lazy: chỉ tạo khi backend cần key thật).
_pool: RotatingKeyPool | None = None


def _get_pool() -> RotatingKeyPool:
    global _pool
    if _pool is None:
        _pool = RotatingKeyPool(settings.api_keys)
    return _pool


def _resolve_base_url(default_base_url: str) -> str | None:
    """LLM_BASE_URL (nếu đặt) > mặc định của backend > None (SDK tự dùng OpenAI Cloud)."""
    if settings.llm_base_url:
        return settings.llm_base_url
    return default_base_url or None


def get_client() -> OpenAI:
    """Trả về OpenAI client trỏ tới backend đang cấu hình.

    - openai: dùng key kế tiếp trong pool (round-robin).
    - ollama/vllm: dùng key giả, không cần pool.
    """
    backend = get_backend(settings.llm_backend)
    base_url = _resolve_base_url(backend.default_base_url)

    if backend.requires_real_key:
        api_key = _get_pool().get_key()
    else:
        api_key = backend.dummy_key

    return OpenAI(api_key=api_key, base_url=base_url)


def mark_current_key_limited(client: OpenAI, cooldown_seconds: float = 60.0) -> None:
    """Đưa key của `client` vào cooldown sau khi gặp 429.

    No-op với backend local (không dùng key thật nên không có gì để cooldown).
    """
    backend = get_backend(settings.llm_backend)
    if backend.requires_real_key and client.api_key:
        _get_pool().mark_limited(client.api_key, cooldown_seconds)
