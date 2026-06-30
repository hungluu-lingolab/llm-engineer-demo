# Vietnamese Legal Assistant — RAG Chatbot

Dự án thực hành **xuyên suốt Module I (LLM Engineer)**. Mỗi buổi học bồi đắp thêm code
vào cùng một codebase, hội tụ dần thành hệ thống capstone: một trợ lý pháp lý tiếng Việt
dùng RAG.

> **Triết lý:** dùng **native SDK** (OpenAI, Qdrant...) thay vì framework cao cấp, để
> engineer hiểu và kiểm soát từng lời gọi. **API-first** với FastAPI.

---

## Lộ trình xây dựng

| Buổi | Chủ đề | Lắp vào codebase |
|------|--------|------------------|
| **1** | LLM APIs Hands-on | `llm/`, `api/`, `prompts/`, `schemas/`, `tools/` — **chatbot chạy được** |
| 2 | Local LLMs | thêm backend local vào `llm/client.py` |
| 3 | Fine-tuning LoRA/QLoRA | swap model checkpoint |
| 4 | Embeddings & Vector DB | `retrieval/embeddings.py`, `vectorstore.py` |
| 5 | RAG Pipeline | `retrieval/chunking.py`, `loader.py`, `retriever.py` → **RAG thật** |
| 6 | Agentic RAG | `agent/graph.py` (LangGraph orchestrate, native SDK call) |
| 7 | Evaluation & Guardrails | `guardrails/`, `eval/` |
| 8 | Production Optimization | `optimization/` (caching, routing) |
| 9 | Capstone | Gradio UI + deploy HF Spaces |

### Buổi 1 đang ở đâu?

Chatbot **đã chạy được** với đầy đủ kiến thức Bài 1: chat completions, parameters,
streaming, structured output, function calling, rate-limit handling.

`retrieval/retriever.py` hiện là **stub trả về rỗng** → chatbot trả lời chỉ dựa vào
kiến thức sẵn có của model, **chưa dựa trên tài liệu thật**. Đây là "lỗ hổng RAG" có
chủ đích, sẽ được lấp ở Buổi 4–5.

---

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate
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

### Thử nghiệm

```bash
# Non-streaming
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Công ty TNHH hai thành viên có tối đa bao nhiêu thành viên?"}'

# Streaming
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Tóm tắt điều kiện thành lập công ty cổ phần."}'

# Structured output
curl -X POST http://localhost:8000/chat/structured \
  -H "Content-Type: application/json" \
  -d '{"question": "Công ty cổ phần cần tối thiểu mấy cổ đông?"}'
```

## Test

```bash
pytest          # không gọi API thật (mock LLM), không cần key
```

---

## Cấu trúc

```
app/
├── config.py          # đọc .env (điểm duy nhất chạm secrets)
├── main.py            # FastAPI app
├── pipeline.py        # orchestrator: retrieve(stub) → prompt → llm
├── api/               # FastAPI routes + request/response schemas
├── llm/               # native OpenAI SDK: completion, streaming, backoff, key rotation
├── prompts/           # role prompting, few-shot, chèn context RAG
├── schemas/           # Pydantic structured output
├── tools/             # function calling
├── retrieval/         # STUB Buổi 1 → RAG thật từ Buổi 4–5
├── agent/             # STUB → Buổi 6
├── guardrails/        # STUB → Buổi 7
├── eval/              # STUB → Buổi 7
└── optimization/      # STUB → Buổi 8
```

> **Bảo mật:** `.env` đã nằm trong `.gitignore`. Không bao giờ commit API key.
