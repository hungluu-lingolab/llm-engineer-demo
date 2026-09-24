"""Semantic Caching — Buổi 8 (Production Optimization); nâng cấp ở Module III,
Bài 3, Section 4 (Caching tầng 2: Semantic Cache).

Cache question-answer pairs không dựa vào token mà dựa vào NGỮ NGHĨA.
Nếu câu hỏi mới gần giống (theo embedding similarity) với câu cũ, trả
lại câu trả lời cũ ngay — không gọi LLM lần nữa.

Nâng cấp so với bản Buổi 8 (bài học Section 4, "Bổ sung cho production"):
  - `ttl_seconds`     : cache stale khi tài liệu nguồn cập nhật (False hit table).
  - `is_volatile()`   : loại câu có tham số ngày/số cụ thể khỏi cache — 2 câu
    khác NĂM vẫn gần giống nhau về embedding ("... năm 2020" vs "... năm 2024")
    nhưng câu trả lời hoàn toàn khác nhau (false hit nguy hiểm nhất, theo bảng
    "False hit và cách phòng" trong bài học).
  - `stats()`         : hit_rate — quyết định "semantic cache có đáng dùng
    không" (bài học: FAQ 20-40% mới đáng, câu hỏi đa dạng <5% thì không).

Production dùng vector store (Redis, Chroma) thay cho in-memory list — vẫn
hoãn sang Bài 4 (Containerisation), cùng lý do như cache_exact.py.
"""

from __future__ import annotations

import re
import time

import numpy as np

# Câu có ngày cụ thể (2020, 2024...) hoặc số tiền/tỷ lệ cụ thể — dễ false hit
# vì embedding không phân biệt tốt các con số khác nhau trong câu gần giống hệt.
_VOLATILE_PATTERNS = [
    re.compile(r"\b(19|20)\d{2}\b"),  # năm 4 chữ số
    re.compile(r"\d+\s*%"),  # tỷ lệ phần trăm
]


def is_volatile(question: str) -> bool:
    """Section 4, bảng False hit: câu có tham số ngày/số -> KHÔNG nên semantic-cache
    (2 câu chỉ khác năm/số vẫn similarity cao nhưng câu trả lời khác hẳn)."""
    return any(p.search(question) for p in _VOLATILE_PATTERNS)


class SemanticCache:
    """Bộ nhớ cache dựa trên similarity embedding.

    threshold: ngưỡng cosine similarity để coi là "gần giống" (0-1).
              Quá thấp -> cache nhầm; quá cao -> ít hit.
              Recommend: 0.90-0.95 cho most cases (bài học: 0.92-0.95).
    ttl_seconds: None = không hết hạn (giữ hành vi cũ). Đặt số giây cụ thể để
                 mô phỏng cache stale khi tài liệu nguồn đổi (Section 4).
    skip_volatile: True (mặc định) -> get()/set() tự bỏ qua câu có tham số
                   ngày/số (is_volatile()), tránh false hit nguy hiểm nhất.
    """

    def __init__(
        self,
        embedder,
        threshold: float = 0.92,
        ttl_seconds: float | None = None,
        skip_volatile: bool = True,
    ):
        """embedder: function(text: str) -> np.ndarray (normalized)."""
        self.embedder = embedder
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.skip_volatile = skip_volatile
        self.questions: list[str] = []
        self.embeddings: list[np.ndarray] = []
        self.answers: list[str] = []
        self._stored_at: list[float] = []
        self._hits = 0
        self._misses = 0

    def get(self, question: str) -> str | None:
        """Tìm câu trả lời cached cho question nếu similarity đủ cao.
        Trả về None nếu cache miss (kể cả khi câu hỏi bị đánh dấu volatile)."""
        if self.skip_volatile and is_volatile(question):
            self._misses += 1
            return None

        self._evict_expired()
        if not self.embeddings:
            self._misses += 1
            return None

        q_emb = self.embedder(question)
        sims = np.dot(np.array(self.embeddings), q_emb)
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= self.threshold:
            self._hits += 1
            return self.answers[best_idx]  # Cache HIT
        self._misses += 1
        return None

    def set(self, question: str, answer: str) -> None:
        """Lưu (question, answer) vào cache — bỏ qua nếu question bị đánh dấu
        volatile (không lưu thứ sẽ gây false hit sau này)."""
        if self.skip_volatile and is_volatile(question):
            return
        self.questions.append(question)
        self.embeddings.append(self.embedder(question))
        self.answers.append(answer)
        self._stored_at.append(time.time())

    def _evict_expired(self) -> None:
        if self.ttl_seconds is None or not self._stored_at:
            return
        now = time.time()
        keep = [i for i, t in enumerate(self._stored_at) if now - t < self.ttl_seconds]
        if len(keep) == len(self._stored_at):
            return
        self.questions = [self.questions[i] for i in keep]
        self.embeddings = [self.embeddings[i] for i in keep]
        self.answers = [self.answers[i] for i in keep]
        self._stored_at = [self._stored_at[i] for i in keep]

    def clear(self) -> None:
        """Xóa toàn bộ cache."""
        self.questions.clear()
        self.embeddings.clear()
        self.answers.clear()
        self._stored_at.clear()

    def size(self) -> int:
        """Số item trong cache."""
        return len(self.answers)

    def stats(self) -> dict[str, float]:
        """Section 4: hit_rate quyết định semantic cache có đáng dùng không."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }
