"""Cost Governance — Module III, Bài 3, Section 7 (Budget & Alert; Hard limit).

Đọc trực tiếp từ `app.cost.tracker.tracker` (cùng CostTracker đã ghi ở
cascade.py/exact cache/script demo) — không có metrics store riêng, đúng tinh
thần "demo 1 process" của cả package `app/cost/`. Production thật sẽ đẩy
`record_cost()` sang Prometheus/Datadog + Redis `incrbyfloat` cho per-user
counter (bài học Section 2) — hoãn hạ tầng đó sang Bài 4, giống cache_exact.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.cost.tracker import CostTracker, tracker

MONTHLY_BUDGET_USD = 800.0
ALERT_THRESHOLDS = [0.5, 0.8, 1.0]  # 50% / 80% / 100%


class BudgetExceeded(Exception):
    """Section 7: hard limit per user — nối vào FastAPI exception handler
    (giống GuardrailViolation ở app/main.py) để trả HTTP 429."""

    def __init__(self, reason: str, user_id: str, spent: float, cap: float):
        self.reason = reason
        self.user_id = user_id
        self.spent = spent
        self.cap = cap
        super().__init__(reason)


@dataclass(slots=True)
class BudgetStatus:
    spent: float
    budget: float
    projected: float
    alerts_fired: list[float] = field(default_factory=list)


def check_budget(
    days_elapsed: int, days_in_month: int, *, tracker_: CostTracker | None = None
) -> BudgetStatus:
    """Section 7: spent hiện tại + dự phóng cuối tháng theo tốc độ chi tiêu.

    `days_elapsed`/`days_in_month` truyền vào thay vì tự tính từ datetime.now()
    để hàm THUẦN, test được dễ dàng không phụ thuộc ngày chạy thật.
    """
    t = tracker_ or tracker
    spent = t.total_cost()
    projected = spent / days_elapsed * days_in_month if days_elapsed > 0 else spent

    alerts_fired = [
        threshold for threshold in ALERT_THRESHOLDS if spent >= MONTHLY_BUDGET_USD * threshold
    ]
    return BudgetStatus(spent=spent, budget=MONTHLY_BUDGET_USD, projected=projected, alerts_fired=alerts_fired)


def guard_user_budget(
    user_id: str, daily_cap_usd: float = 2.0, *, tracker_: CostTracker | None = None
) -> None:
    """Section 7: raise BudgetExceeded nếu user đã chi vượt hạn mức NGÀY.

    Tính spent HÔM NAY từ tracker.records (lọc theo user_id + timestamp trong
    24h gần nhất) — thay vì Redis key `cost:user:{id}:{date}` như bài học, vì
    tracker đã là nguồn dữ liệu chung của cả package (không cần thêm store).
    """
    import time

    t = tracker_ or tracker
    one_day_ago = time.time() - 86400
    spent_today = sum(
        r.cost_usd for r in t.records if r.user_id == user_id and r.timestamp >= one_day_ago
    )
    if spent_today >= daily_cap_usd:
        raise BudgetExceeded(
            f"Vượt hạn mức ngày (${daily_cap_usd}). Thử lại vào ngày mai.",
            user_id=user_id,
            spent=spent_today,
            cap=daily_cap_usd,
        )
