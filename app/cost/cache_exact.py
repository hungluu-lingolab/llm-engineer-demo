"""Exact Response Cache (tầng 1) — Module III, Bài 3, Section 3.

Khác prompt caching (của provider, cache PREFIX): đây là cache TOÀN BỘ câu trả
lời cho input GIỐNG HỆT, ở tầng ứng dụng — cache hit thì KHÔNG gọi LLM chút nào.

QUAN TRỌNG NHẤT của tầng này (bài học nhấn mạnh bằng !!! warning riêng):
`prompt_version` PHẢI nằm trong cache key. Không có nó, promote prompt v2 → v3
vẫn trả lời theo v2 trong suốt TTL — cache "quên" rebuild.

Store tách khỏi logic qua `CacheStore` protocol — Bài 3 chỉ cần `InMemoryStore`
(đủ minh hoạ cache key/TTL/invalidation theo version). Module III, Bài 4
(Containerisation) thêm `RedisStore` implement ĐÚNG 2 method dưới đây, không
đổi gì ở ExactCache — đây là lý do tách interface ngay từ đầu thay vì viết cứng
vào Redis rồi phải refactor sau.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Protocol


class CacheStore(Protocol):
    """Interface tối thiểu 1 cache store cần có — InMemoryStore (ở đây) và
    RedisStore (Bài 4, dùng `redis.setex`) đều implement 2 method này."""

    def get(self, key: str) -> str | None: ...
    def setex(self, key: str, ttl_seconds: int, value: str) -> None: ...


class InMemoryStore:
    """Store demo — dict + TTL tự quản lý bằng timestamp, không cần Redis.

    KHÔNG dùng cho production nhiều instance (mỗi process 1 cache riêng, không
    chia sẻ) — đó chính xác là lý do Bài 4 thay bằng RedisStore.
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, str]] = {}  # key -> (expires_at, value)

    def get(self, key: str) -> str | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() >= expires_at:
            del self._data[key]
            return None
        return value

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        self._data[key] = (time.time() + ttl_seconds, value)

    def clear(self) -> None:
        self._data.clear()


def cache_key(
    prompt_name: str, prompt_version: int, model: str, rendered_prompt: str, params: dict
) -> str:
    """Section 3: prompt_version + model NẰM TRONG key — bump version tự động
    là "namespace" mới, bản cache cũ hết hạn tự nhiên (không cần xoá tay)."""
    raw = json.dumps(
        {
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "model": model,
            "rendered_prompt": rendered_prompt,
            "params": params,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return "resp:" + hashlib.sha256(raw.encode()).hexdigest()


class ExactCache:
    """Cache tầng 1: input giống hệt (theo cache_key) -> trả lại answer cũ.

    Dùng như:
        cache = ExactCache()
        answer, cache_hit = cache.get_or_call(
            prompt_name="rag_answer", prompt_version=2, model="gpt-4o-mini",
            rendered_prompt=question, params={}, call_fn=lambda: real_llm_call(question),
        )
    """

    def __init__(self, store: CacheStore | None = None, ttl_seconds: int = 86400):
        self.store = store or InMemoryStore()
        self.ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get_or_call(
        self,
        *,
        prompt_name: str,
        prompt_version: int,
        model: str,
        rendered_prompt: str,
        params: dict,
        call_fn,
    ) -> tuple[str, bool]:
        """Trả (answer, cache_hit). call_fn() chỉ được gọi khi cache MISS.

        Dùng khi tầng 1 là cache DUY NHẤT (không xen thêm tầng khác ở giữa
        check và store). Khi cần chèn semantic cache (tầng 2) GIỮA lúc kiểm
        tra exact-miss và lúc gọi LLM thật (đúng kiến trúc Section 4: tầng 1 ->
        tầng 2 -> gọi LLM), dùng `peek()`/`store_result()` riêng — xem
        scripts/cost_replay_demo.py.
        """
        key = cache_key(prompt_name, prompt_version, model, rendered_prompt, params)

        cached = self.store.get(key)
        if cached is not None:
            self._hits += 1
            return cached, True

        self._misses += 1
        answer = call_fn()
        self.store.setex(key, self.ttl_seconds, answer)
        return answer, False

    def peek(self, *, prompt_name: str, prompt_version: int, model: str, rendered_prompt: str, params: dict) -> str | None:
        """Chỉ KIỂM TRA cache, không gọi LLM nếu miss — cập nhật hits/misses.
        Dùng khi caller cần tự quyết định bước tiếp theo (vd thử tầng 2 trước
        khi gọi LLM thật), xem docstring get_or_call()."""
        key = cache_key(prompt_name, prompt_version, model, rendered_prompt, params)
        cached = self.store.get(key)
        if cached is not None:
            self._hits += 1
        else:
            self._misses += 1
        return cached

    def store_result(self, *, prompt_name: str, prompt_version: int, model: str, rendered_prompt: str, params: dict, answer: str) -> None:
        """Ghi answer vào cache — dùng cặp với peek() khi không dùng get_or_call()."""
        key = cache_key(prompt_name, prompt_version, model, rendered_prompt, params)
        self.store.setex(key, self.ttl_seconds, answer)

    def stats(self) -> dict[str, float]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }
