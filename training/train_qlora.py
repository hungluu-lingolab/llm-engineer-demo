"""Fine-tune Llama 3.2 3B với QLoRA — Buổi 3, Section 5.

Train LoRA adapter trên legal Q&A dataset dùng TRL SFTTrainer + 4-bit quantization.

YÊU CẦU: GPU NVIDIA + CUDA (bitsandbytes 4-bit không chạy trên CPU/Apple Silicon).
Cài deps: pip install -r training/requirements-train.txt

Chạy:
    python -m training.train_qlora

Sau khi train xong, adapter được lưu ở CONFIG.output_dir. Bước tiếp theo:
merge adapter vào base model bằng training/merge_adapter.py.
"""

from __future__ import annotations

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from training.config import CONFIG
from training.prepare_data import load_training_dataset


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "Cần GPU NVIDIA + CUDA để train QLoRA (bitsandbytes 4-bit).\n"
            "Trên máy không có GPU: dùng Google Colab / cloud GPU, hoặc bỏ qua bước train "
            "và dùng model đã fine-tune sẵn."
        )

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Quantization config — QLoRA (Section 4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG.load_in_4bit,
        bnb_4bit_quant_type=CONFIG.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=CONFIG.bnb_4bit_use_double_quant,
    )

    # 3. Load model 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    # 4. LoRA config (Section 3)
    peft_config = LoraConfig(
        r=CONFIG.lora_r,
        lora_alpha=CONFIG.lora_alpha,
        target_modules=CONFIG.target_modules,
        lora_dropout=CONFIG.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Training config (Section 5)
    sft_config = SFTConfig(
        output_dir=CONFIG.output_dir,
        num_train_epochs=CONFIG.num_train_epochs,
        per_device_train_batch_size=CONFIG.per_device_train_batch_size,
        gradient_accumulation_steps=CONFIG.gradient_accumulation_steps,
        learning_rate=CONFIG.learning_rate,
        bf16=True,
        max_grad_norm=CONFIG.max_grad_norm,
        warmup_ratio=CONFIG.warmup_ratio,
        lr_scheduler_type=CONFIG.lr_scheduler_type,
        logging_steps=CONFIG.logging_steps,
        save_strategy="epoch",
        # max_seq_length=CONFIG.max_seq_length,
        dataset_text_field="text",
        packing=CONFIG.packing,
    )

    # 6. Dataset
    dataset = load_training_dataset(CONFIG.data_path)

    # 7. Train
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()
    print(f"\nAdapter đã lưu ở: {CONFIG.output_dir}")
    print("Bước tiếp: python -m training.merge_adapter")


if __name__ == "__main__":
    main()
