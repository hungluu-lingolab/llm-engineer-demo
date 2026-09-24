"""Token Economics — Module III, Bài 3, Section 1-2 (Token Economics; Đo trước khi tối ưu).

Trước khi tối ưu bất kỳ thứ gì (cache, routing, cascade) phải ĐO ĐƯỢC tiền đi
đâu. File này chỉ làm 2 việc:
  - cost_of()          : tính tiền 1 lần gọi từ (model, input_tokens, output_tokens).
  - record_cost()       : ghi 1 bản ghi cost, gắn feature/user/model/cache_hit —
    đủ để trả lời 3 lát cắt bài học nêu (per feature, per user, per request).
  - breakdown_tokens()  : ước lượng phần trăm token đến từ system/context/history/
    question — minh hoạ trực quan "context bloat là kẻ giết ngân sách" (Section 1).

Store: in-memory list (đủ cho demo 1 process, xem CostTracker.records). Production
thật sẽ đẩy qua metrics store (Prometheus/Datadog) + Redis cho per-user counter —
KHÔNG implement ở đây, đúng lý do cache_exact.py cũng hoãn Redis sang Bài 4
(containerisation): khái niệm cần dạy là "đo cái gì", không phải hạ tầng lưu trữ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# USD / 1M tokens — giá minh hoạ, LUÔN tra bảng giá hiện hành trước khi dùng thật.
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
}


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Section 1: cost_request ≈ input×giá_in + output×giá_out.

    Model không có trong PRICING → coi giá 0 thay vì raise, để demo/test không
    vỡ khi thêm model mới mà quên cập nhật bảng giá (log rõ thay vì crash).
    """
    price = PRICING.get(model)
    if price is None:
        return 0.0
    return prompt_tokens / 1e6 * price["in"] + completion_tokens / 1e6 * price["out"]


@dataclass(slots=True)
class CostRecord:
    feature: str
    user_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cache_hit: bool
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """Section 2: gắn nhãn mọi request — per feature / per user / per request.

    In-memory, KHÔNG thread-safe theo nghĩa production (đủ cho demo 1 process).
    """

    def __init__(self) -> None:
        self.records: list[CostRecord] = []

    def record(
        self,
        *,
        feature: str,
        user_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_hit: bool = False,
    ) -> CostRecord:
        cost = 0.0 if cache_hit else cost_of(model, prompt_tokens, completion_tokens)
        rec = CostRecord(
            feature=feature,
            user_id=user_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            cache_hit=cache_hit,
        )
        self.records.append(rec)
        return rec

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def by_feature(self) -> dict[str, float]:
        """Section 2, lát cắt 'per feature': tính năng nào ngốn ngân sách?"""
        out: dict[str, float] = {}
        for r in self.records:
            out[r.feature] = out.get(r.feature, 0.0) + r.cost_usd
        return out

    def by_user(self) -> dict[str, float]:
        """Section 2, lát cắt 'per user': ai đang lạm dụng / cost bất thường?"""
        out: dict[str, float] = {}
        for r in self.records:
            out[r.user_id] = out.get(r.user_id, 0.0) + r.cost_usd
        return out

    def cache_hit_rate(self) -> float:
        if not self.records:
            return 0.0
        hits = sum(1 for r in self.records if r.cache_hit)
        return hits / len(self.records)

    def percentile(self, p: float) -> float:
        """Section 2, lát cắt 'per request phân bố': p50 vs p99 chênh bao nhiêu?
        p99 cao = có request context khổng lồ / loop agent dài."""
        costs = sorted(r.cost_usd for r in self.records)
        if not costs:
            return 0.0
        idx = min(int(len(costs) * p), len(costs) - 1)
        return costs[idx]

    def clear(self) -> None:
        self.records.clear()


# Instance dùng chung cho demo/script — mỗi process 1 tracker, không persist.
tracker = CostTracker()


def breakdown_tokens(
    *, system: str = "", context: str = "", history: str = "", question: str = ""
) -> dict[str, int]:
    """Section 1: ước lượng phân bổ token theo NGUỒN (system/context/history/
    question), không chỉ tổng input_tokens — đây là cách "thấy" context bloat.

    Ước lượng thô 1 token ≈ 4 ký tự tiếng Anh — với tiếng Việt tỷ lệ thực tế
    thường DÀY hơn (nhiều token/ký tự hơn do dấu + từ ghép), nên số ở đây là
    CẬN DƯỚI, đủ để so sánh TỶ LỆ giữa các phần, không dùng để tính tiền chính
    xác (cost_of() dùng usage thật từ API cho việc đó).
    """
    parts = {"system": system, "context": context, "history": history, "question": question}
    return {name: len(text) // 4 for name, text in parts.items()}


def breakdown_pct(tokens: dict[str, int]) -> dict[str, float]:
    """Chuyển breakdown_tokens() sang % — trực quan hơn khi in báo cáo."""
    total = sum(tokens.values())
    if total == 0:
        return {k: 0.0 for k in tokens}
    return {k: round(v / total * 100, 1) for k, v in tokens.items()}
