"""Generation parameters — Bài 1, Section 2 (Parameters Quan Trọng).

Gom các tham số kiểm soát output của model vào một chỗ, có default từ settings.
Tách riêng để mỗi request có thể override mà không đụng tới logic gọi API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.config import settings


@dataclass(slots=True)
class GenerationParams:
    """Tham số sinh text. Xem Bài 1 — Section 2.

    - temperature: mức độ ngẫu nhiên (0.0 = deterministic).
    - max_completion_tokens: giới hạn token output (thay cho max_tokens đã deprecated;
      bắt buộc với các model reasoning o1/o3/o4-mini).
    - top_p: nucleus sampling. KHÔNG nên chỉnh đồng thời với temperature.
    """

    temperature: float = settings.llm_temperature
    max_completion_tokens: int = settings.llm_max_completion_tokens
    top_p: float = settings.llm_top_p

    def to_openai_kwargs(self) -> dict:
        """Map sang đúng tên tham số của OpenAI Chat Completions API."""
        return asdict(self)


# Vài preset tiện dùng — minh hoạ "chọn temperature theo task" (Bài 1, slide Temperature).
DETERMINISTIC = GenerationParams(temperature=0.0)  # code, fact extraction, structured output
BALANCED = GenerationParams(temperature=0.5)        # chatbot, Q&A
CREATIVE = GenerationParams(temperature=0.9)        # brainstorming, creative writing
