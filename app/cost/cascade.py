"""Model Cascading — Module III, Bài 3, Section 5 (Model Routing & Cascading).

Khác ROUTING (app/optimization/routing.py — quyết định model TRƯỚC khi gọi,
dựa trên QUERY): cascading gọi model rẻ TRƯỚC, rồi quyết định có ESCALATE lên
model mạnh hay không dựa trên OUTPUT. Đánh đổi: case khó phải trả tiền CẢ HAI
model.

`_is_confident()` là tín hiệu escalate đơn giản (câu trả lời không nói "không
biết/không tìm thấy") — bài học liệt kê thêm logprobs/self-rating, ở đây chỉ
implement tín hiệu rẻ nhất để demo cơ chế, không phải vét cạn mọi tín hiệu có thể.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.cost.tracker import cost_of, tracker

CHEAP_MODEL = "gpt-4o-mini"
STRONG_MODEL = "gpt-4o"

# Tín hiệu model rẻ "không tự tin" — escalate lên model mạnh khi thấy các cụm này.
_LOW_CONFIDENCE_MARKERS = [
    "không tìm thấy",
    "không có thông tin",
    "không chắc chắn",
    "tôi không biết",
]


def _is_confident(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return not any(marker in lowered for marker in _LOW_CONFIDENCE_MARKERS)


@dataclass(slots=True)
class CascadeResult:
    answer: str
    model_used: str
    escalated: bool
    cost_usd: float


def answer_cascade(
    question: str,
    *,
    build_messages_fn,
    feature: str = "cascade_demo",
    user_id: str = "anonymous",
) -> CascadeResult:
    """Section 5: gọi CHEAP_MODEL trước; nếu không tự tin -> escalate STRONG_MODEL.

    `build_messages_fn(question) -> list[dict]` để tái dùng đúng cách dựng
    messages của app (vd app.prompts.templates.build_messages) thay vì hard-code
    prompt riêng ở đây.

    Ghi cost qua tracker CHO CẢ HAI lần gọi khi escalate — đây chính là điểm
    "trả tiền cả hai model" mà bài học cảnh báo, và là lý do cần đo escalate_rate.
    """
    from app.llm import completion
    from app.llm.params import GenerationParams

    messages = build_messages_fn(question)

    cheap_text, cheap_usage = completion.chat_with_usage(
        messages, GenerationParams(), model=CHEAP_MODEL
    )
    tracker.record(
        feature=feature,
        user_id=user_id,
        model=CHEAP_MODEL,
        prompt_tokens=cheap_usage.prompt_tokens,
        completion_tokens=cheap_usage.completion_tokens,
    )

    if _is_confident(cheap_text):
        cost = cost_of(CHEAP_MODEL, cheap_usage.prompt_tokens, cheap_usage.completion_tokens)
        return CascadeResult(answer=cheap_text, model_used=CHEAP_MODEL, escalated=False, cost_usd=cost)

    strong_text, strong_usage = completion.chat_with_usage(
        messages, GenerationParams(), model=STRONG_MODEL
    )
    tracker.record(
        feature=feature,
        user_id=user_id,
        model=STRONG_MODEL,
        prompt_tokens=strong_usage.prompt_tokens,
        completion_tokens=strong_usage.completion_tokens,
    )
    total_cost = cost_of(
        CHEAP_MODEL, cheap_usage.prompt_tokens, cheap_usage.completion_tokens
    ) + cost_of(STRONG_MODEL, strong_usage.prompt_tokens, strong_usage.completion_tokens)
    return CascadeResult(answer=strong_text, model_used=STRONG_MODEL, escalated=True, cost_usd=total_cost)


def escalate_rate(results: list[CascadeResult]) -> float:
    """Section 5: "Đo tỉ lệ escalate. Nếu 70% case phải escalate -> cascade
    đang lỗ (trả 2 lần tiền cho phần lớn traffic); chuyển sang routing hoặc
    dùng thẳng model mạnh." — ngưỡng cảnh báo 50% lấy từ bảng theo dõi Section 7."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.escalated) / len(results)
