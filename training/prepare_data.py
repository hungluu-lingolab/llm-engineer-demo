"""Chuẩn bị dataset — Buổi 3, Section 5.

Load legal Q&A seed (Alpaca format: instruction / input / output) và format
thành prompt text để đưa vào SFTTrainer.

Dataset ở đây là seed nhỏ để minh hoạ pipeline. Trong thực tế cần ≥ 100–1000
example chất lượng cao (xem Pitfalls trong handbook: chất lượng > số lượng).

Có thể chạy độc lập để kiểm tra format:
    python -m training.prepare_data
"""

from __future__ import annotations

from datasets import load_dataset


def format_prompt(example: dict) -> str:
    """Alpaca prompt template (khớp handbook)."""
    if example.get("input"):
        return (
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Input:\n"
            f"{example['input']}\n\n"
            "### Response:\n"
            f"{example['output']}"
        )
    return (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Response:\n"
        f"{example['output']}"
    )


def load_training_dataset(data_path: str):
    """Load JSONL và thêm trường `text` đã format cho SFTTrainer."""
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(lambda x: {"text": format_prompt(x)})
    return dataset


if __name__ == "__main__":
    from training.config import CONFIG

    ds = load_training_dataset(CONFIG.data_path)
    print(f"Loaded {len(ds)} examples từ {CONFIG.data_path}\n")
    print("─── Ví dụ prompt đã format ───")
    print(ds[0]["text"])
