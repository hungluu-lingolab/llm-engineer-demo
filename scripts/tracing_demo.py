"""Demo Buổi 7 — Monitoring: ingest tài liệu thật, gọi pipeline.answer() /
answer_stream() thật, trace lên LangSmith thật (dùng credentials trong .env).

Yêu cầu trong .env:
    OPENAI_API_KEYS=sk-...
    MONITORING_ENABLED=true
    LANGSMITH_API_KEY=lsv2_pt_...

Chạy:
    python -m scripts.tracing_demo
"""

from __future__ import annotations

from app.config import settings
from scripts.ingest import ingest

QUESTION = "Điều kiện thành lập công ty TNHH là gì?"


def main() -> None:
    if not settings.monitoring_enabled:
        raise SystemExit(
            "MONITORING_ENABLED=false trong .env — bật true và điền "
            "LANGSMITH_API_KEY trước khi chạy demo này."
        )

    print(f"[ingest] nạp tài liệu từ {settings.rag_source_dir} vào Qdrant ({settings.qdrant_url}) ...")
    ingest()

    from app.pipeline import answer, answer_stream

    print(f"\n[answer] question = {QUESTION!r}")
    result = answer(QUESTION)
    print(f"[answer] response = {result!r}")

    print(f"\n[answer_stream] question = {QUESTION!r}")
    tokens = []
    for token in answer_stream(QUESTION):
        tokens.append(token)
        print(token, end="", flush=True)
    print()

    print(f"\nĐã gửi 2 trace ('answer', 'answer_stream') lên LangSmith, project '{settings.langsmith_project}'")
    print(f"Kiểm tra tại {settings.langsmith_endpoint.replace('api.smith', 'smith')}")


if __name__ == "__main__":
    main()
