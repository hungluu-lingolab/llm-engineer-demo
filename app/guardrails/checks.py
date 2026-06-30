"""Guardrails — STUB. Sẽ implement ở Buổi 7 (Evaluation & Guardrails).

Dự kiến:
  - Input: phát hiện PII, prompt injection.
  - Output: kiểm tra faithfulness (câu trả lời có bám tài liệu không).
"""

from __future__ import annotations


def check_input(text: str) -> None:
    """STUB — implement ở Buổi 7. Buổi 1: no-op (cho phép mọi input)."""
    return None


def check_output(answer: str, context: list[str]) -> None:
    """STUB — implement ở Buổi 7. Buổi 1: no-op."""
    return None
