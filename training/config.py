"""Cấu hình fine-tuning — Buổi 3.

Gom mọi hyperparameter vào một chỗ để dễ thử nghiệm và tái lập.
Các giá trị mặc định bám theo handbook (QLoRA cho Llama 3.2 3B trên GPU 24GB).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # ── Model & output ──────────────────────────────────────────────────────
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir: str = "./training/output/llama-3.2-3b-legal-vi"

    # ── Data ────────────────────────────────────────────────────────────────
    # Dataset legal Q&A tự tạo (Alpaca format). Xem training/data/legal_seed.jsonl.
    data_path: str = "./training/data/legal_seed.jsonl"
    max_seq_length: int = 1024

    # ── LoRA (Section 3) ────────────────────────────────────────────────────
    lora_r: int = 16                 # rank: 8 (đơn giản) · 16 (balanced) · 32-64 (phức tạp)
    lora_alpha: int = 32             # thường = 2×r → scaling = 2
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",       # FFN
        ]
    )

    # ── QLoRA quantization (Section 4) ──────────────────────────────────────
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # ── Training (Section 5) ────────────────────────────────────────────────
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4   # effective batch = 2×4 = 8
    learning_rate: float = 2e-4            # điểm khởi đầu tốt; quá cao sẽ phá model
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    logging_steps: int = 10
    packing: bool = True                   # đóng gói nhiều example vào 1 sequence

    # ── Push lên HF Hub (Section 6, tùy chọn) ───────────────────────────────
    push_to_hub: bool = False
    hub_repo_id: str = "your-username/llama-3.2-3b-legal-vi"


CONFIG = TrainConfig()
