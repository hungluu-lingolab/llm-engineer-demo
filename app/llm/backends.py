"""Backend presets — Buổi 2 (Local Serving).

Điểm cốt lõi của buổi này: Ollama và vLLM đều expose **OpenAI-compatible API**.
Nghĩa là ta vẫn dùng đúng `openai` SDK, chỉ đổi `base_url` + `api_key`.
Toàn bộ completion.py / pipeline.py KHÔNG phải sửa dòng nào.

    OpenAI Cloud : https://api.openai.com/v1   (cần key thật)
    Ollama       : http://localhost:11434/v1   (key giả "ollama")
    vLLM         : http://localhost:8000/v1    (key giả hoặc token nội bộ)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Backend:
    name: str
    default_base_url: str  # "" = dùng mặc định của SDK (OpenAI Cloud)
    requires_real_key: bool
    dummy_key: str = "not-needed"


BACKENDS: dict[str, Backend] = {
    "openai": Backend(
        name="openai",
        default_base_url="",  # SDK tự dùng https://api.openai.com/v1
        requires_real_key=True,
    ),
    "ollama": Backend(
        name="ollama",
        default_base_url="http://localhost:11434/v1",
        requires_real_key=False,
        dummy_key="ollama",
    ),
    "vllm": Backend(
        name="vllm",
        default_base_url="http://localhost:8000/v1",
        requires_real_key=False,
        dummy_key="vllm",
    ),
}


def get_backend(name: str) -> Backend:
    """Trả về Backend theo tên, báo lỗi rõ ràng nếu không hợp lệ."""
    key = name.strip().lower()
    if key not in BACKENDS:
        raise ValueError(
            f"LLM_BACKEND='{name}' không hợp lệ. "
            f"Chọn một trong: {', '.join(BACKENDS)}."
        )
    return BACKENDS[key]
