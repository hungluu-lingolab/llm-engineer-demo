"""LLM client dùng chung cho 4 pattern multi-agent — Module II, Bài 6.

Tách riêng khỏi nodes.py (agent đơn, Bài 2-5) vì multi-agent không cần tool
retrieval (Bài 4) — mỗi agent con ở đây chỉ bind ĐÚNG tool của domain mình
(xem tools.py: TOOL_GROUPS), không phải chọn lọc qua embedding similarity.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def base_llm():
    """Client chưa bind tool, cache 1 lần — giống app/agent_m2/nodes.py:_base_llm.

    Truyền thẳng api_key từ settings.api_keys[0] (không để ChatOpenAI tự đọc
    biến môi trường OPENAI_API_KEY số ít — repo dùng OPENAI_API_KEYS số nhiều).
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.api_keys[0] if settings.api_keys else None,
    )
