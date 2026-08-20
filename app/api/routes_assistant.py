"""Assistant routes — Module II, Bài 2 (Building Agents với LangGraph).

Endpoint riêng khỏi /chat (Module I) vì đây là track khác: không phải RAG,
mà là ReAct agent tổng quát với tool calling + memory + HITL (Section 7).

Bài 4: async def — start_conversation/resume_conversation (graph.py) giờ là
coroutine (agent_node cần await MCP client, Section 4). FastAPI chạy route
handler async natively, không cần thay đổi gì thêm ở tầng HTTP.

Bài 5: /evaluate — chấm điểm lượt hội thoại mới nhất (task success + trajectory,
eval.py). Tách khỏi /message vì đây là hành động ON-DEMAND (bấm nút "Đánh giá"
trên UI), không chạy tự động mỗi lượt như tracing (graph.py:trace_answer).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.agent_m2.graph import evaluate_thread, resume_conversation, start_conversation
from app.api.schemas import (
    AssistantApprovalRequest,
    AssistantEvaluateRequest,
    AssistantEvaluateResponse,
    AssistantMessageRequest,
    AssistantResponse,
)

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/message", response_model=AssistantResponse)
async def send_message(req: AssistantMessageRequest) -> AssistantResponse:
    """Gửi 1 lượt tin nhắn. Có thể trả về `pending_approval` nếu agent muốn gọi tool.

    `user_id` (Bài 3): bật long-term memory — agent recall thông tin đã biết về
    user (dị ứng, sở thích...) và tự lưu sự thật mới sau mỗi lượt.
    """
    return AssistantResponse(
        **await start_conversation(req.thread_id, req.message, req.user_id)
    )


@router.post("/approve", response_model=AssistantResponse)
async def approve_tool_call(req: AssistantApprovalRequest) -> AssistantResponse:
    """Duyệt/từ chối tool call đang chờ (HITL, Section 6), rồi tiếp tục graph."""
    return AssistantResponse(
        **await resume_conversation(req.thread_id, req.approve, req.rejection_note)
    )


@router.post("/evaluate", response_model=AssistantEvaluateResponse)
async def evaluate(req: AssistantEvaluateRequest) -> AssistantEvaluateResponse:
    """Bài 5 — chấm task success + trajectory cho lượt mới nhất của thread_id.

    Đọc lại checkpointer (không cần frontend gửi kèm lịch sử/trajectory) — xem
    docstring evaluate_thread (graph.py) cho lý do thiết kế.
    """
    out = await evaluate_thread(req.thread_id)
    if "error" in out:
        return AssistantEvaluateResponse(error=out["error"])

    result = out["result"]
    return AssistantEvaluateResponse(
        task_success=result.task_success.model_dump(),
        trajectory={**result.trajectory.model_dump(), "overall": result.trajectory.overall},
        trajectory_steps=out["trajectory"],
    )
