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
| **2** | Local LLMs | `llm/backends.py` + `client.py` đa backend, `Modelfile` — **chạy model local** |
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

### Buổi 2 — Local Serving

Cùng một codebase giờ chạy được với **model local** (Ollama / vLLM) — không đổi
`completion.py` hay `pipeline.py`, vì Ollama/vLLM đều dùng **OpenAI-compatible API**.
Chỉ cần đổi backend trong `.env`.

#### Phương án A — Ollama (khuyến nghị cho laptop / GPU yếu / Apple Silicon)

```bash
# 1. Cài Ollama
#    macOS:  brew install ollama   (hoặc tải app tại https://ollama.com/download)
#    Linux:  curl -fsSL https://ollama.ai/install.sh | sh
#    Windows: tải installer tại https://ollama.com/download

# 2. Khởi động Ollama server (mặc định cổng 11434)
ollama serve          # để chạy nền; trên macOS app tự chạy sau khi mở

# 3. Pull model (terminal khác)
ollama pull llama3.2:3b

# 4. (Tùy chọn) Tạo model có sẵn persona pháp lý từ Modelfile
ollama create legal-assistant -f Modelfile

# 5. Kiểm tra nhanh
ollama run llama3.2:3b "Xin chào"

# 6. Trỏ app sang Ollama trong .env:
#    LLM_BACKEND=ollama
#    LLM_MODEL=llama3.2:3b        # hoặc legal-assistant
```

#### Phương án B — vLLM (GPU NVIDIA, production / nhiều người dùng)

```bash
# 1. Cài vLLM (cần GPU NVIDIA + CUDA)
pip install vllm

# 2. Serve model với OpenAI-compatible API (mặc định cổng 8000)
vllm serve meta-llama/Llama-3.2-3B-Instruct

# 3. Trỏ app sang vLLM trong .env:
#    LLM_BACKEND=vllm
#    LLM_MODEL=meta-llama/Llama-3.2-3B-Instruct
```

> vLLM chạy trên Linux + GPU NVIDIA. Trên macOS/Windows không có GPU NVIDIA thì
> dùng Ollama (Phương án A).

Sau khi backend chạy, khởi động app như thường: `uvicorn app.main:app --reload`.
Giờ mọi request đi tới model local — **không tốn chi phí API, dữ liệu không rời máy**.

| Backend | `LLM_BACKEND` | base_url mặc định | Cần key thật | Yêu cầu |
|---------|---------------|-------------------|--------------|---------|
| OpenAI Cloud | `openai` | api.openai.com | ✅ | — |
| Ollama | `ollama` | localhost:11434/v1 | ❌ | CPU / GPU yếu / Apple Silicon |
| vLLM | `vllm` | localhost:8000/v1 | ❌ | GPU NVIDIA + CUDA |

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
Modelfile              # Buổi 2 — Ollama model có sẵn persona pháp lý
app/
├── config.py          # đọc .env (điểm duy nhất chạm secrets)
├── main.py            # FastAPI app
├── pipeline.py        # orchestrator: retrieve(stub) → prompt → llm
├── api/               # FastAPI routes + request/response schemas
├── llm/               # native SDK: completion, streaming, backoff, key rotation
│                      #   + backends.py (Buổi 2: openai/ollama/vllm)
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
