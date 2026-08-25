"""Collaborative pattern — Module II, Bài 6, Section 1.

Nhiều agent PHẢN BIỆN lẫn nhau trước khi chốt kết quả — không có thứ bậc cố
định (khác Hierarchical: không có Supervisor trung tâm quyết định thay).
Phù hợp khi task cần nhiều góc nhìn để giảm sai sót.

Ở đây: Planner đề xuất kế hoạch cho 1 ngày (dùng ghi chú từ tools.py domain
liên quan), Critic phản biện (xung đột lịch? chi phí hợp lý? thiếu bước nào?)
— lặp lại tới khi Critic DUYỆT hoặc hết số vòng tối đa.

    START → planner → critic ─┬─(duyệt)──────→ END
                               └─(còn vấn đề)→ planner → critic → ...

Không dùng ToolNode ở đây — Collaborative pattern tập trung vào CHẤT LƯỢNG
qua phản biện text-to-text, không phải hành động qua tool (đó là trọng tâm
Sequential/Hierarchical). Planner có thể tham khảo dữ liệu tool nhưng không
tự thực thi side-effect — giữ demo tập trung đúng pattern đang minh hoạ.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.agent_m2.multi_agent._llm import base_llm

MAX_ROUNDS = 3  # giới hạn cứng — bài học Section 3: luôn giới hạn vòng lặp multi-agent


class CollaborativeState(TypedDict):
    messages: Annotated[list, add_messages]
    request: str
    plan: str  # bản kế hoạch mới nhất từ Planner
    feedback: str  # phản biện mới nhất từ Critic
    approved: bool
    round: int


class _CriticVerdict(BaseModel):
    approved: bool = Field(description="Kế hoạch đã ổn, không còn vấn đề đáng kể")
    feedback: str = Field(description="Vấn đề cụ thể cần sửa (nếu approved=false), hoặc lời khen ngắn")


_PLANNER_SYSTEM = """Bạn là Planner — đề xuất kế hoạch cho 1 ngày/sự kiện theo
yêu cầu user. Nếu có phản hồi từ Critic ở lượt trước, SỬA kế hoạch theo đúng
góp ý đó, đừng lặp lại y hệt bản cũ."""


async def planner_node(state: CollaborativeState) -> dict:
    prompt = f"Yêu cầu: {state['request']}"
    if state.get("feedback"):
        prompt += f"\n\nKế hoạch trước: {state['plan']}\nPhản biện cần sửa: {state['feedback']}"

    response = await base_llm().ainvoke(
        [{"role": "system", "content": _PLANNER_SYSTEM}, {"role": "user", "content": prompt}]
    )
    return {"plan": response.content, "messages": [response], "round": state["round"] + 1}


_CRITIC_SYSTEM = """Bạn là Critic — phản biện kế hoạch một cách khắt khe nhưng
công bằng. Kiểm tra: có xung đột thời gian không? có bước nào thiếu thực tế
không (thiếu địa điểm/giờ cụ thể)? có hợp lý về chi phí/công sức không?

Chỉ approved=true khi kế hoạch THỰC SỰ ổn — đừng duyệt chỉ vì đã đủ vòng."""


async def critic_node(state: CollaborativeState) -> dict:
    verdict = await base_llm().with_structured_output(_CriticVerdict).ainvoke(
        [
            {"role": "system", "content": _CRITIC_SYSTEM},
            {"role": "user", "content": f"Yêu cầu gốc: {state['request']}\n\nKế hoạch cần phản biện:\n{state['plan']}"},
        ]
    )
    return {"approved": verdict.approved, "feedback": verdict.feedback}


def route_after_critic(state: CollaborativeState) -> str:
    if state["approved"]:
        return "end"
    if state["round"] >= MAX_ROUNDS:
        # Hết vòng mà vẫn chưa duyệt — dừng an toàn, KHÔNG lặp vô hạn (Section 3).
        return "end"
    return "planner"


@lru_cache(maxsize=1)
def _build_graph():
    graph = StateGraph(CollaborativeState)
    graph.add_node("planner", planner_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"planner": "planner", "end": END})

    return graph.compile()


async def run_collaborative(request: str) -> dict:
    """Chạy vòng lặp Planner ⇄ Critic tới khi duyệt hoặc hết MAX_ROUNDS.

    Trả kèm `rounds`/`approved` để UI thấy được QUÁ TRÌNH phản biện, không chỉ
    kế hoạch cuối — đúng tinh thần minh hoạ "chất lượng qua nhiều góc nhìn".
    """
    app = _build_graph()
    result = await app.ainvoke(
        {"messages": [], "request": request, "plan": "", "feedback": "", "approved": False, "round": 0},
        config={"recursion_limit": MAX_ROUNDS * 3 + 5},
    )
    return {
        "plan": result["plan"],
        "feedback": result["feedback"],
        "approved": result["approved"],
        "rounds": result["round"],
    }
