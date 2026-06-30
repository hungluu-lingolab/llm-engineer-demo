"""Resilience — Bài 1, Section 6 (Rate Limits & API Key Rotation).

Hai cơ chế độc lập nhưng bổ sung nhau:
  1. RotatingKeyPool  — round-robin nhiều API key, đưa key bị 429 vào cooldown.
  2. retry_with_backoff — exponential backoff + jitter khi gặp RateLimitError.

Cả hai đều thuần native (không framework) để học viên thấy rõ control flow.
"""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Callable
from typing import TypeVar

from openai import RateLimitError

T = TypeVar("T")


class RotatingKeyPool:
    """Round-robin pool nhiều API key, tự đưa key đang bị rate-limit vào cooldown.

    Xem Bài 1 — slide "Code - RotatingKeyPool Nâng Cao".
    """

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError(
                "RotatingKeyPool cần ít nhất 1 API key. "
                "Đặt OPENAI_API_KEYS trong .env (xem .env.example)."
            )
        self._keys = deque(keys)
        self._cooldown: dict[str, float] = {}  # key -> thời điểm hết cooldown

    def get_key(self) -> str:
        """Trả về key sẵn sàng kế tiếp. Nếu tất cả đang cooldown thì đợi ngắn rồi thử lại."""
        now = time.time()
        for _ in range(len(self._keys)):
            key = self._keys[0]
            self._keys.rotate(-1)
            if self._cooldown.get(key, 0.0) <= now:
                return key
        # Tất cả key đang cooldown -> đợi rồi đệ quy.
        time.sleep(1.0)
        return self.get_key()

    def mark_limited(self, key: str, cooldown_seconds: float = 60.0) -> None:
        """Đánh dấu một key vừa bị 429 -> tạm loại trong `cooldown_seconds` giây."""
        self._cooldown[key] = time.time() + cooldown_seconds


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int = 5,
    on_rate_limit: Callable[[], None] | None = None,
) -> T:
    """Gọi `fn()` với exponential backoff + jitter khi gặp RateLimitError.

    Công thức chờ: 2**attempt + random(0, 1)  (xem Bài 1 — slide Exponential Backoff).

    Args:
        fn: hàm không tham số thực hiện lời gọi API.
        max_retries: số lần thử tối đa trước khi raise.
        on_rate_limit: callback chạy mỗi khi gặp 429 (ví dụ: mark key vào cooldown).
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            if on_rate_limit is not None:
                on_rate_limit()
            if attempt == max_retries - 1:
                raise
            wait = (2**attempt) + random.uniform(0, 1)
            time.sleep(wait)
    # Không bao giờ tới đây (vòng lặp luôn return hoặc raise), nhưng để type-checker yên tâm.
    raise RuntimeError("retry_with_backoff: hết số lần thử")
