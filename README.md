# LLM Engineer — Demo Codebase

Codebase thực hành xuyên suốt khoá **LLM Engineer**, xây dựng dần theo từng buổi học,
gồm 3 module độc lập trên cùng 1 FastAPI app:

| Module | Chủ đề | Chi tiết |
|--------|--------|----------|
| **Module I** | Vietnamese Legal Assistant — RAG Chatbot (chat completions → RAG → agentic RAG → eval/guardrails → production optimization) | [README.module1.md](README.module1.md) |
| **Module II** | AI Agent — Personal Assistant (LangGraph: ReAct loop, HITL, memory, context engineering, tool design, MCP) | [README.module2.md](README.module2.md) |
| **Module III** | LLM Ops — Eval Pipelines (golden dataset trên LangSmith, regression gate, CI) | [README.module3.md](README.module3.md) |

> **Triết lý:** dùng **native SDK** (OpenAI, Qdrant...) thay vì framework cao cấp, để
> engineer hiểu và kiểm soát từng lời gọi. **API-first** với FastAPI. LangGraph (Module I
> Buổi 6, Module II) là ngoại lệ có chủ đích — chỉ orchestrate control flow, xem giải
> thích trong từng README module.

---

## Cài đặt

```bash
conda create -n llm-engineer python==3.10
pip install -r requirements.txt

cp .env.example .env
# Mở .env, điền OPENAI_API_KEYS (một hoặc nhiều key, ngăn cách bằng dấu phẩy)
```

## Chạy

```bash
uvicorn app.main:app --reload
```

- Docs tương tác: <http://localhost:8000/docs>
- Health check (không cần key): <http://localhost:8000/health>
- **Demo UI**: <http://localhost:8000/> — chat interface đơn giản (`app/static/chat.html`),
  lịch sử chat lưu ở `localStorage` của trình duyệt (không có backend session/DB).
  Bấm **"Ingest dữ liệu"** trong UI trước khi hỏi lần đầu — với `QDRANT_URL=:memory:`
  (mặc định), mỗi process server có Qdrant riêng, ingest ở CLI process khác sẽ
  không nạp được cho server đang chạy; nút này gọi `POST /admin/ingest` để ingest
  đúng vào trong process server. Toggle **Streaming** (`/chat/stream`) và
  **Agent (CRAG)** (`/chat/agent`, hiện thêm sources + web-search fallback badge)
  để so sánh 2 pipeline trực tiếp (Module I). Câu hỏi bị chặn bởi guardrails hiện
  dạng bubble lỗi riêng (đọc HTTP 400 từ exception handler trong `main.py`).
  Module II (Personal Assistant agent) chạy qua `/assistant/message` +
  `/assistant/approve` — xem ví dụ curl trong [README.module2.md](README.module2.md).

## Test

```bash
pytest          # không gọi API thật (mock LLM), không cần key
```

Test cho cả 2 module nằm chung trong `tests/` (`test_chat.py`, `test_rag.py`,
`test_agent.py`, ... cho Module I; `test_agent_m2*.py` cho Module II).

---

## Cấu trúc tổng quan

```
Modelfile              # Module I, Buổi 2 — Ollama model có sẵn persona pháp lý
app/
├── config.py          # đọc .env (điểm duy nhất chạm secrets, dùng chung 2 module)
├── main.py            # FastAPI app + phục vụ static/chat.html tại "/"
├── pipeline.py         # Module I — orchestrator: retrieve → prompt → llm
├── static/             # ✓ chat.html — demo UI dùng chung, lịch sử lưu localStorage
├── api/                 # FastAPI routes + schemas (routes_chat/routes_admin: Module I;
│                        #   routes_assistant: Module II)
├── llm/                 # Module I — native SDK: completion, streaming, backoff, key rotation
├── prompts/              # Module I — role prompting, few-shot, chèn context RAG
├── schemas/              # Module I — Pydantic structured output
├── tools/                # Module I, Bài 1 — function calling viết tay
├── retrieval/            # Module I — RAG hoàn chỉnh (loader, chunking, embeddings, vectorstore, retriever)
├── agent/                # Module I, Buổi 6 — CRAG + Query Decomposition (LangGraph)
├── agent_m2/             # Module II — Personal Assistant agent (LangGraph) — chi tiết ở README.module2.md
├── guardrails/           # Module I — injection.py, pii.py, checks.py
├── eval/                 # Module I — judge.py (LLM-as-Judge), ragas_native.py, metrics.py
├── eval_pipeline/        # Module III — golden dataset (LangSmith) + runner + regression gate
├── monitoring/           # Module I — tracing.py, LangSmith hooks tối thiểu
└── optimization/         # Module I — prompt_cache.py, caching.py, routing.py
```

Chi tiết từng buổi học, từng file, ví dụ curl: xem
[README.module1.md](README.module1.md), [README.module2.md](README.module2.md),
[README.module3.md](README.module3.md).

> **Bảo mật:** `.env` đã nằm trong `.gitignore`. Không bao giờ commit API key.
