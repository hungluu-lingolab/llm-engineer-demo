"""Chat completion — Bài 1, Section 1 (Chat Completions) + 5 (Function Calling) + 6 (Streaming).

Lớp mỏng trên OpenAI SDK gom 4 cách gọi model mà Bài 1 dạy:
  - chat()            : gọi cơ bản, trả text.
  - chat_stream()     : streaming, yield từng token.
  - chat_parsed()     : structured output với Pydantic (Section 4).
  - chat_with_tools() : function calling, trả về tool_calls để code thực thi (Section 5).

Mọi lời gọi đều bọc trong retry_with_backoff + key rotation (Section 6).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from app.config import settings
from app.llm.client import get_client, mark_current_key_limited
from app.llm.params import GenerationParams
from app.llm.resilience import retry_with_backoff

TModel = TypeVar("TModel", bound=BaseModel)

Messages = list[ChatCompletionMessageParam]


def chat(messages: Messages, params: GenerationParams | None = None) -> str:
    """Gọi chat completion cơ bản, trả về nội dung text của assistant.

    Section 1: chat.completions.create(messages) -> assistant message.
    """
    params = params or GenerationParams()
    client = get_client()

    def _call() -> ChatCompletion:
        return client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            **params.to_openai_kwargs(),
        )

    response = retry_with_backoff(
        _call,
        max_retries=settings.llm_max_retries,
        on_rate_limit=lambda: mark_current_key_limited(client),
    )
    return response.choices[0].message.content or ""


def chat_stream(
    messages: Messages, params: GenerationParams | None = None
) -> Iterator[str]:
    """Streaming — yield từng đoạn token ngay khi model sinh ra (Section 6).

    Lưu ý: stream không retry giữa chừng (một khi đã bắt đầu nhận token).
    Backoff chỉ áp dụng cho bước khởi tạo stream.
    """
    params = params or GenerationParams()
    client = get_client()

    def _open_stream():
        return client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            stream=True,
            **params.to_openai_kwargs(),
        )

    stream = retry_with_backoff(
        _open_stream,
        max_retries=settings.llm_max_retries,
        on_rate_limit=lambda: mark_current_key_limited(client),
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:  # delta có thể là None ở chunk cuối
            yield delta


def chat_parsed(
    messages: Messages,
    schema: type[TModel],
    params: GenerationParams | None = None,
) -> TModel:
    """Structured output — ép model trả về đúng schema Pydantic (Section 4).

    Dùng client.chat.completions.parse() (đã ra khỏi beta).
    """
    params = params or GenerationParams()
    client = get_client()

    def _call():
        return client.chat.completions.parse(
            model=settings.llm_model,
            messages=messages,
            response_format=schema,
            **params.to_openai_kwargs(),
        )

    completion = retry_with_backoff(
        _call,
        max_retries=settings.llm_max_retries,
        on_rate_limit=lambda: mark_current_key_limited(client),
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model không trả về output khớp schema.")
    return parsed


def chat_with_tools(
    messages: Messages,
    tools: list[dict],
    params: GenerationParams | None = None,
) -> ChatCompletion:
    """Function calling — trả về response thô để code kiểm tra tool_calls (Section 5).

    Trả nguyên ChatCompletion (không rút gọn về text) vì caller cần đọc
    finish_reason và message.tool_calls để quyết định thực thi function nào.
    """
    params = params or GenerationParams()
    client = get_client()

    def _call() -> ChatCompletion:
        return client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            tools=tools,
            **params.to_openai_kwargs(),
        )

    return retry_with_backoff(
        _call,
        max_retries=settings.llm_max_retries,
        on_rate_limit=lambda: mark_current_key_limited(client),
    )
