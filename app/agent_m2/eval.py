"""Agent Evaluation — Module II, Bài 5, Section 1-2 (Eval Dimensions, Methodologies).

RAGAS/LLM-as-judge (Module I, Bài 7) chấm OUTPUT CUỐI — đủ cho RAG nhưng không
đủ cho agent, vì agent có thể ra đúng đáp án bằng con đường sai (lặp tool thừa,
tốn kém, chọn tool sai tham số). File này thêm 2 chiều đánh giá agent-specific:

  - evaluate_task_success : giống judge.py (Module I) nhưng chấm "có đạt mục
    tiêu không", không phải chất lượng văn phong.
  - evaluate_trajectory   : chấm TOÀN BỘ chuỗi tool call — efficiency, thứ tự
    logic, tool/tham số đúng không, recovery khi có lỗi. Đây là phần MỚI so
    với Module I, chỉ agent mới cần (RAG không có "trajectory" nhiều bước).

Cả 2 đều dùng chat_parsed (native, Bài 1) + rubric tuyệt đối — tái áp dụng
nguyên tắc giảm bias đã học ở judge.py (Module I, Bài 7): không so sánh
ranking, không thưởng verbosity, chấm độc lập từng tiêu chí. Với agent,
verbosity bias CÀNG dễ xảy ra vì trajectory dài — judge dễ nhầm "nhiều bước
= kỹ lưỡng" trong khi thực ra là lãng phí (xem docstring evaluate_trajectory).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.llm import completion
from app.llm.params import GenerationParams


class TaskSuccessResult(BaseModel):
    """Section 2: End-to-end Evaluation — agent có đạt mục tiêu không."""

    success: bool = Field(description="Agent có hoàn thành đúng yêu cầu của user không")
    score: float = Field(ge=0.0, le=1.0, description="Điểm hoàn thành 0-1 (1 = hoàn hảo)")
    reasoning: str = Field(description="Giải thích ngắn gọn cho kết luận")


class TrajectoryResult(BaseModel):
    """Section 2: Trajectory Evaluation — chấm CẢ chuỗi hành động, không chỉ đích đến.

    4 tiêu chí độc lập (rubric tuyệt đối, 1-5) — giữ nguyên tinh thần judge.py
    (Module I): không để 1 tiêu chí ảnh hưởng tiêu chí khác, không thưởng
    trajectory dài/nhiều bước nếu không thực sự cần thiết (verbosity bias).
    """

    efficiency: int = Field(ge=1, le=5, description="Có bước thừa/lặp không cần thiết không")
    logical_order: int = Field(ge=1, le=5, description="Thứ tự bước có hợp lý không")
    tool_correctness: int = Field(ge=1, le=5, description="Tool và tham số có phù hợp mục đích từng bước không")
    recovery: int = Field(ge=1, le=5, description="Nếu có bước lỗi, agent có xử lý hợp lý không (5 nếu không có lỗi)")
    issues: list[str] = Field(default_factory=list, description="Vấn đề cụ thể phát hiện được, nếu có")

    @property
    def overall(self) -> float:
        return (self.efficiency + self.logical_order + self.tool_correctness + self.recovery) / 4


class AgentEvalResult(BaseModel):
    """Gộp cả 2 chiều — trả về cho API/UI trong 1 lần gọi (Section 1: 4 eval dimensions,
    2 chiều đầu ở đây; Tool Accuracy nằm trong trajectory.tool_correctness, Cost/Latency
    lấy trực tiếp từ LangFuse span thay vì LLM judge — xem monitoring/tracing.py)."""

    task_success: TaskSuccessResult
    trajectory: TrajectoryResult
    step_count: int = Field(description="Số tool call thực tế trong trajectory")


_TASK_SUCCESS_SYSTEM = """Bạn là giám khảo đánh giá agent AI có hoàn thành nhiệm vụ không.

Chỉ chấm KẾT QUẢ CUỐI CÙNG so với yêu cầu ban đầu của user — không quan tâm
agent đi đường nào để tới đó (điều đó được chấm riêng ở trajectory eval)."""


def evaluate_task_success(
    task: str, final_output: str, success_criteria: str = ""
) -> TaskSuccessResult:
    """Section 2: agent có đạt mục tiêu user đề ra không (không xét quá trình)."""
    parts = [f"Nhiệm vụ (yêu cầu của user): {task}", f"Kết quả agent trả về: {final_output}"]
    if success_criteria:
        parts.append(f"Tiêu chí thành công: {success_criteria}")

    messages = [
        {"role": "system", "content": _TASK_SUCCESS_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    return completion.chat_parsed(messages, TaskSuccessResult, GenerationParams(temperature=0.0))


_TRAJECTORY_SYSTEM = """Bạn là giám khảo đánh giá CHUỖI HÀNH ĐỘNG của agent AI, không phải
chỉ câu trả lời cuối. Một agent có thể ra đúng đáp án bằng con đường lãng phí
(gọi tool thừa/lặp) hoặc rủi ro — nhiệm vụ của bạn là phát hiện việc đó.

Chấm 4 tiêu chí ĐỘC LẬP theo thang 1-5. KHÔNG thưởng điểm cao cho trajectory
dài/nhiều bước — nhiều bước hơn thường là DẤU HIỆU KÉM HIỆU QUẢ, không phải
"kỹ lưỡng hơn" (verbosity bias)."""


def _format_trajectory(trajectory: list[dict]) -> str:
    if not trajectory:
        return "(không có tool call nào — agent trả lời trực tiếp)"
    return "\n".join(
        f"Bước {i + 1}: gọi {step['tool']}({step['args']}) → {str(step['observation'])[:200]}"
        for i, step in enumerate(trajectory)
    )


def evaluate_trajectory(task: str, trajectory: list[dict]) -> TrajectoryResult:
    """Section 2: chấm toàn bộ chuỗi tool call.

    Args:
        task: yêu cầu gốc của user.
        trajectory: [{"tool": str, "args": dict, "observation": str}, ...] —
            xem graph.py:_extract_trajectory() để biết cách trích từ state thật.
    """
    formatted = _format_trajectory(trajectory)
    messages = [
        {"role": "system", "content": _TRAJECTORY_SYSTEM},
        {
            "role": "user",
            "content": f"Nhiệm vụ: {task}\n\nChuỗi hành động:\n{formatted}",
        },
    ]
    return completion.chat_parsed(messages, TrajectoryResult, GenerationParams(temperature=0.0))


def evaluate_run(task: str, final_output: str, trajectory: list[dict]) -> AgentEvalResult:
    """Chấm cả 2 chiều trong 1 lần gọi — dùng bởi POST /assistant/evaluate."""
    return AgentEvalResult(
        task_success=evaluate_task_success(task, final_output),
        trajectory=evaluate_trajectory(task, trajectory),
        step_count=len(trajectory),
    )
