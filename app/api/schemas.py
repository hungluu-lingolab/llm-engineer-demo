"""API request/response schemas (FastAPI layer).

Tách riêng khỏi schemas/domain.py: đây là "hợp đồng" HTTP với client,
còn domain.py là hình dạng dữ liệu nội bộ. Giữ tách biệt để hai thứ tiến hoá độc lập.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import settings


class OptimizationStats(BaseModel):
    """Thống kê Buổi 8: prompt caching, semantic cache, routing."""

    routing_model: str = Field(default="", description="Model được chọn qua routing (e.g., 'gpt-4o-mini')")
    routing_method: str = Field(default="", description="Phương pháp routing ('rule_based', 'embedding', 'classifier')")
    cache_hit: bool = Field(default=False, description="Semantic cache HIT (true) hay MISS (false)")
    prompt_cache_created_tokens: int = Field(default=0, description="Tokens tạo cache mới (OpenAI)")
    prompt_cache_read_tokens: int = Field(default=0, description="Tokens đọc từ cache (OpenAI)")
    prompt_cache_hit_ratio: float = Field(default=0.0, description="Tỷ lệ cache hit 0-1 (Buổi 8)")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="Câu hỏi của người dùng")
    # Cho phép override generation params mỗi request (Bài 1, Section 2).
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_completion_tokens: int | None = Field(default=None, gt=0)


class ChatResponse(BaseModel):
    answer: str
    model: str = Field(default_factory=lambda: settings.llm_model)
    optimization: OptimizationStats = Field(default_factory=OptimizationStats)


class AgentSource(BaseModel):
    text: str
    source: str
    score: float


class AgentChatResponse(BaseModel):
    """Response cho /chat/agent — trả thêm bằng chứng để thấy 'vì sao trả lời vậy'."""

    answer: str
    sources: list[AgentSource]
    web_search_used: bool
    sub_questions: list[str]
    optimization: OptimizationStats = Field(default_factory=OptimizationStats)


class AssistantMessageRequest(BaseModel):
    """Module II, Bài 2-3 — gửi 1 lượt tin nhắn tới Personal Assistant (LangGraph)."""

    thread_id: str = Field(min_length=1, description="Định danh hội thoại (session) — short-term memory qua checkpointer")
    message: str = Field(min_length=1)
    user_id: str = Field(
        default="",
        description="Bài 3: định danh user cho long-term memory (xuyên session). Bỏ trống → không dùng long-term memory.",
    )


class AssistantApprovalRequest(BaseModel):
    """Phê duyệt/từ chối tool call đang chờ (HITL, Section 6)."""

    thread_id: str = Field(min_length=1)
    approve: bool
    rejection_note: str = Field(default="", description="Lý do từ chối, nếu approve=false")


class PendingToolCall(BaseModel):
    name: str
    args: dict


class AssistantResponse(BaseModel):
    """status='done' → answer có giá trị. status='pending_approval' → tool_call có giá trị."""

    status: str
    answer: str | None = None
    tool_call: PendingToolCall | None = None


class AssistantEvaluateRequest(BaseModel):
    """Bài 5 — chấm điểm lượt hội thoại mới nhất của 1 thread (task success + trajectory)."""

    thread_id: str = Field(min_length=1)


class TaskSuccessSchema(BaseModel):
    success: bool
    score: float
    reasoning: str


class TrajectoryStep(BaseModel):
    tool: str
    args: dict
    observation: str


class TrajectorySchema(BaseModel):
    efficiency: int
    logical_order: int
    tool_correctness: int
    recovery: int
    overall: float
    issues: list[str]


class AssistantEvaluateResponse(BaseModel):
    """error != None khi chưa có gì để chấm (chưa chat, hoặc đang chờ duyệt tool)."""

    error: str | None = None
    task_success: TaskSuccessSchema | None = None
    trajectory: TrajectorySchema | None = None
    trajectory_steps: list[TrajectoryStep] = Field(default_factory=list)
