"""Multi-agent routes — Module II, Bài 6 (Multi-Agent Systems).

4 endpoint, mỗi endpoint 1 pattern (Section 1), tất cả trên cùng domain trợ
lý cá nhân để dễ so sánh trực tiếp trên UI. Tách khỏi /assistant (Bài 2-5,
1 agent đơn) vì đây là 4 KIẾN TRÚC multi-agent độc lập, không phải mở rộng
thêm cho agent hiện có.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.agent_m2.multi_agent.collaborative import run_collaborative
from app.agent_m2.multi_agent.hierarchical import run_hierarchical
from app.agent_m2.multi_agent.sequential import run_sequential
from app.agent_m2.multi_agent.swarm import run_swarm
from app.api.schemas import (
    CollaborativeResponse,
    HierarchicalResponse,
    MultiAgentRequest,
    SequentialResponse,
    SwarmRequest,
    SwarmResponse,
)

router = APIRouter(prefix="/multi-agent", tags=["multi-agent"])


@router.post("/sequential", response_model=SequentialResponse)
async def sequential(req: MultiAgentRequest) -> SequentialResponse:
    """Planner → Scheduler → Notifier, thứ tự cố định (Section 1: Sequential)."""
    return SequentialResponse(**await run_sequential(req.request))


@router.post("/hierarchical", response_model=HierarchicalResponse)
async def hierarchical(req: MultiAgentRequest) -> HierarchicalResponse:
    """Supervisor điều phối Worker theo domain — nền tảng capstone Bài 7 (Section 1: Hierarchical)."""
    return HierarchicalResponse(**await run_hierarchical(req.request))


@router.post("/collaborative", response_model=CollaborativeResponse)
async def collaborative(req: MultiAgentRequest) -> CollaborativeResponse:
    """Planner ⇄ Critic phản biện lặp vòng tới khi duyệt hoặc hết vòng (Section 1: Collaborative)."""
    return CollaborativeResponse(**await run_collaborative(req.request))


@router.post("/swarm", response_model=SwarmResponse)
async def swarm(req: SwarmRequest) -> SwarmResponse:
    """3 agent domain ngang hàng, tự handoff qua Command — không Supervisor trung tâm (Section 1: Swarm)."""
    return SwarmResponse(**await run_swarm(req.message, req.entry_agent))
