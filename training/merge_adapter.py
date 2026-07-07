"""Merge LoRA adapter vào base model — Buổi 3, Section 6.

Sau khi train, adapter (A, B matrices) nằm riêng. merge_and_unload() gộp chúng
vào weights gốc → một model độc lập, inference không cần thư viện PEFT.

Model đã merge có thể deploy qua backend Buổi 2:
  - vLLM: trỏ thẳng tới thư mục output (hoặc HF repo).
  - Ollama: convert sang GGUF (llama.cpp) rồi `ollama create`.
→ Sau đó đặt LLM_MODEL trong .env là xong — app không phải sửa gì.

Chạy:
    python -m training.merge_adapter
"""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.config import CONFIG

MERGED_DIR = CONFIG.output_dir + "-merged"


def main() -> None:
    # 1. Load base model (full precision để merge chính xác)
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 2. Load adapter và merge
    model = PeftModel.from_pretrained(base_model, CONFIG.output_dir)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_id)

    # 3. Lưu model đã merge
    model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Merged! Model độc lập đã lưu ở: {MERGED_DIR}")

    # 4. (Tùy chọn) Push lên HF Hub
    if CONFIG.push_to_hub:
        # Yêu cầu đã `huggingface-cli login` hoặc đặt HF_TOKEN.
        model.push_to_hub(CONFIG.hub_repo_id)
        tokenizer.push_to_hub(CONFIG.hub_repo_id)
        print(f"Đã push: https://huggingface.co/{CONFIG.hub_repo_id}")


if __name__ == "__main__":
    main()
