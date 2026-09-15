"""Monitoring hooks — Buổi 7, Section 4 (LangSmith).

Tối thiểu: 1 trace cho mỗi lần gọi pipeline.answer*(), gắn question/answer/
latency/lỗi. KHÔNG bọc qua LangChain callback — dùng langsmith SDK trực tiếp
(Client.create_run/update_run) để giữ triết lý "native SDK" của repo.

Mặc định tắt (MONITORING_ENABLED=false) nên khi chưa điền LANGSMITH_API_KEY
trong .env, toàn bộ hàm ở đây là no-op — không ai bắt buộc phải cài/kích hoạt
LangSmith để chạy phần còn lại của codebase.

Khác LangFuse (bản trước của file này): LangSmith không có object "span" tiện
dụng với `.start_observation()`/`.update()`/`.end()` sẵn có — `Client.create_run`
chỉ ghi 1 run (cần tự sinh `id`), đóng run phải gọi `update_run(run_id=...)`
riêng. `_Run` bên dưới là wrapper MỎNG giả lập lại đúng 3 method đó, để
app/agent/graph.py và app/agent_m2/graph.py (gọi qua `t["_span"]`) KHÔNG PHẢI
sửa dòng nào khi đổi nền tảng tracing.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.config import settings


def _get_client():
    """Lazy import + lazy client — tránh phụ thuộc cứng vào package `langsmith`
    khi MONITORING_ENABLED=false, và tránh tạo client ở import-time."""
    from langsmith import Client

    return Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)


class _Run:
    """Wrapper mỏng quanh 1 LangSmith run — giả lập API "span" của LangFuse
    (start_observation/update/end) để call site không cần biết nền tảng dưới.

    LangSmith yêu cầu tự sinh `run_id` (uuid4) và tự gọi `create_run`/
    `update_run` riêng biệt — khác LangFuse trả về object có sẵn 2 thao tác đó.
    """

    def __init__(self, client, name: str, input: Any, metadata: dict | None = None, parent_run_id=None):
        self._client = client
        self.run_id = uuid.uuid4()
        self._parent_run_id = parent_run_id
        self._start = time.perf_counter()
        kwargs: dict[str, Any] = {"id": self.run_id, "project_name": settings.langsmith_project}
        if parent_run_id is not None:
            kwargs["parent_run_id"] = parent_run_id
        client.create_run(
            name=name,
            inputs={"input": input},
            run_type="chain",
            metadata=metadata or {},
            **kwargs,
        )

    def start_observation(self, name: str, input: Any = None, metadata: dict | None = None) -> "_Run":
        """Tạo nested child run — dùng bởi trace_step() để lồng span trong 1 graph
        (giống app/agent/nodes.py: decompose/retrieve/grade/... lồng dưới 1 trace)."""
        return _Run(self._client, name, input, metadata, parent_run_id=self.run_id)

    def update(self, output: Any = None, metadata: dict | None = None, level: str | None = None, status_message: str | None = None) -> None:
        # Merge, không ghi đè — trace_answer/trace_step gọi update() 2 LẦN khi
        # có lỗi (1 lần set error trong except, 1 lần set output trong finally).
        # Ghi đè sẽ mất _pending_error vì finally luôn gọi update(output=...)
        # sau đó mà không truyền lại level/status_message.
        if output is not None:
            self._pending_output = output
        if metadata is not None:
            self._pending_metadata = {**(getattr(self, "_pending_metadata", None) or {}), **metadata}
        if level == "ERROR":
            self._pending_error = status_message

    def end(self) -> None:
        latency = time.perf_counter() - self._start
        metadata = {"latency_s": latency, **(getattr(self, "_pending_metadata", None) or {})}
        output = getattr(self, "_pending_output", None)
        error = getattr(self, "_pending_error", None)
        self._client.update_run(
            run_id=self.run_id,
            outputs={"output": output} if output is not None else None,
            error=error,
            extra={"metadata": metadata},
        )


@contextmanager
def trace_answer(name: str, question: str, metadata: dict[str, Any] | None = None):
    """Bọc quanh 1 lần gọi pipeline (answer/answer_stream/answer_structured).

    Dùng như:
        with trace_answer("answer", question) as t:
            result = ...
            t["output"] = result

    `t["_span"]` là run cha đang mở — truyền xuống cho trace_step() để tạo
    nested run (xem trace_step()). Chỉ có mặt khi monitoring bật; các call site
    dùng t.get("_span") nên tự an toàn khi tắt.
    """
    if not settings.monitoring_enabled:
        yield {}
        return

    client = _get_client()
    run = _Run(client, name, question, metadata)
    box: dict[str, Any] = {"_span": run}
    try:
        yield box
    except Exception as exc:
        run.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        run.update(output=box.get("output"))
        run.end()


@contextmanager
def trace_step(parent_span: Any, name: str, input: Any = None, metadata: dict[str, Any] | None = None):
    """Nested child run dưới `parent_span` (lấy từ trace_answer's t["_span"]).

    Dùng trong LangGraph node để thấy từng bước (decompose/retrieve/grade/...)
    lồng nhau trong LangSmith — thay vì 1 run phẳng cho toàn bộ graph.invoke().
    No-op nếu parent_span là None (monitoring tắt, hoặc node chạy ngoài trace).

    Dùng như:
        with trace_step(parent_span, "grade_documents", input=question) as t:
            ...
            t["output"] = graded
    """
    if parent_span is None:
        yield {}
        return

    run = parent_span.start_observation(name=name, input=input, metadata=metadata or {})
    box: dict[str, Any] = {}
    try:
        yield box
    except Exception as exc:
        run.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        run.update(output=box.get("output"))
        run.end()


def trace_stream(name: str, question: str, tokens: Iterator[str]) -> Iterator[str]:
    """Bọc quanh answer_stream(): trace toàn bộ output ghép lại sau khi stream
    kết thúc (không thể tạo run "giữa chừng" cho streaming)."""
    if not settings.monitoring_enabled:
        yield from tokens
        return

    client = _get_client()
    start = time.perf_counter()
    chunks: list[str] = []
    try:
        for token in tokens:
            chunks.append(token)
            yield token
    finally:
        run_id = uuid.uuid4()
        client.create_run(
            id=run_id,
            name=name,
            inputs={"input": question},
            run_type="chain",
            project_name=settings.langsmith_project,
        )
        client.update_run(
            run_id=run_id,
            outputs={"output": "".join(chunks)},
            extra={"metadata": {"latency_s": time.perf_counter() - start, "streamed": True}},
        )
