# Fine-tuning — Buổi 3 (LoRA / QLoRA)

Pipeline **offline** để fine-tune model cho domain pháp lý tiếng Việt. Tách khỏi
`app/` (app là inference server; đây là training). Sau khi train + merge, model
được deploy qua backend Buổi 2 — app **không đổi dòng nào**, chỉ đặt lại `LLM_MODEL`.

---

## Quyết định trước khi fine-tune

> Fine-tuning là bước **đắt nhất**. Thử theo thứ tự chi phí tăng dần:

```
Prompt Engineering  →  RAG  →  Fine-tuning
   (rẻ nhất)                     (đắt nhất)
```

| Vấn đề | Giải pháp đúng |
|--------|----------------|
| Cần kiến thức từ tài liệu cụ thể, cập nhật liên tục | **RAG** (Buổi 4–5) |
| Đổi hành vi/style/format, prompting làm được | **Prompt Engineering** (Buổi 1) |
| Đổi hành vi sâu, nhất quán + có ≥ 100–1000 example | **Fine-tuning** (buổi này) |

Với trợ lý pháp lý: **kiến thức luật → dùng RAG**. Fine-tune chỉ để cố định
*giọng văn pháp lý* và *format trích dẫn điều luật*.

---

## Cài đặt (cần GPU NVIDIA + CUDA)

```bash
pip install -r training/requirements-train.txt
```

> `bitsandbytes` (QLoRA 4-bit) **chỉ chạy trên GPU NVIDIA**. Không có GPU → dùng
> Google Colab / cloud GPU. Đó là lý do các dep này tách khỏi `requirements.txt` chính.

## Quy trình

```bash
# 1. Kiểm tra format dataset (chạy được không cần GPU)
python -m training.prepare_data

# 2. Train QLoRA adapter (cần GPU)
python -m training.train_qlora

# 3. Merge adapter vào base model → model độc lập
python -m training.merge_adapter

# 4. Deploy qua backend Buổi 2, ví dụ vLLM:
#    vllm serve ./training/output/llama-3.2-3b-legal-vi-merged
#    rồi trong .env:  LLM_BACKEND=vllm
#                     LLM_MODEL=./training/output/llama-3.2-3b-legal-vi-merged
```

## File

| File | Vai trò |
|------|---------|
| `config.py` | Mọi hyperparameter (LoRA r/alpha, QLoRA, SFT) |
| `data/legal_seed.jsonl` | Legal Q&A seed (Alpaca format) — **cần mở rộng cho thực tế** |
| `prepare_data.py` | Load + format Alpaca prompt |
| `train_qlora.py` | QLoRA SFTTrainer |
| `merge_adapter.py` | `merge_and_unload()` + tùy chọn push HF |

## Lưu ý (pitfalls)

- **Chất lượng > số lượng:** 500 example tốt hơn 5000 example kém. Seed ở đây chỉ
  ~10 dòng để minh hoạ pipeline — thực tế cần nhiều hơn.
- **Catastrophic forgetting:** train quá nhiều epoch → model quên kiến thức gốc.
  Dùng `r` nhỏ hơn hoặc ít epoch hơn.
- **Learning rate:** `2e-4` là điểm khởi đầu tốt; quá cao sẽ phá model.
