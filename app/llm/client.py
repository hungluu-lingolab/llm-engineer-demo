"""OpenAI client factory — Bài 1, Section 1 + Section 6.

Bọc việc tạo client + key rotation vào một chỗ. Phần còn lại của app chỉ gọi
`get_client()` mà không cần biết key đến từ đâu hay xoay vòng thế nào.

NGUYÊN TẮC BẢO MẬT (Bài 1): key luôn đọc từ env qua settings — KHÔNG hardcode,
KHÔNG commit. Xem app/config.py và .env.example.
"""

from __future__ import annotations

from openai import OpenAI

from app.config import settings
from app.llm.resilience import RotatingKeyPool

# Khởi tạo pool một lần ở module level (lazy: chỉ raise khi thực sự cần key).
_pool: RotatingKeyPool | None = None


def _get_pool() -> RotatingKeyPool:
    global _pool
    if _pool is None:
        _pool = RotatingKeyPool(settings.api_keys)
    return _pool


def get_client() -> OpenAI:
    """Trả về OpenAI client dùng key kế tiếp trong pool (round-robin)."""
    return OpenAI(api_key=_get_pool().get_key())


def mark_current_key_limited(client: OpenAI, cooldown_seconds: float = 60.0) -> None:
    """Đưa key của `client` vào cooldown sau khi gặp 429."""
    if client.api_key:
        _get_pool().mark_limited(client.api_key, cooldown_seconds)
