"""Sequential pattern — Module II, Bài 6, Section 1.

Agent chạy THEO THỨ TỰ CỐ ĐỊNH, output của agent trước là input của agent sau
— không quay lại bước trước. Phù hợp khi quy trình có thứ tự rõ ràng, biết
trước các bước (đúng ví dụ handbook: research → phân tích → viết báo cáo).

Ở đây: lên kế hoạch một việc cần làm (vd "đặt lịch ăn tối cuối tuần") theo 3
bước cố định:

    Planner (phân tích yêu cầu → kế hoạch từng bước, KHÔNG gọi tool)
        → Scheduler (thực thi kế hoạch bằng tool calendar/dining thật)
            → Notifier (thông báo kết quả qua tool productivity)

Không có "quay lại" — nếu Scheduler thất bại, lỗi được đưa vào state để
Notifier báo cho user biết, KHÔNG quay lại Planner lập kế hoạch khác (đó là
việc của Collaborative/Hierarchical, xem collaborative.py/hierarchical.py).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agent_m2.multi_agent._llm import base_llm
from app.agent_m2.tools import TOOL_GROUPS


class SequentialState(TypedDict):
    """State chảy tuần tự qua 3 agent — mỗi agent chỉ ĐỌC field agent trước
    ghi, và GHI field của riêng mình. Không có field nào bị 2 agent cùng ghi."""

    request: str  # yêu cầu gốc của user
    messages: Annotated[list, add_messages]

    plan: str  # Planner ghi — kế hoạch dạng text, Scheduler đọc để biết làm gì

    schedule_result: str  # Scheduler ghi — kết quả thực thi tool, Notifier đọc

    notification: str  # Notifier ghi — nội dung thông báo cuối cùng


_PLANNER_SYSTEM = """Bạn là Planner — chỉ LẬP KẾ HOẠCH, không thực thi hành động.
Phân tích yêu cầu của user thành kế hoạch ngắn gọn 2-4 bước cụ thể (ngày giờ,
địa điểm nếu có). Không gọi tool, chỉ trả về kế hoạch dạng text."""


def planner_node(state: SequentialState) -> dict:
    response = base_llm().invoke(
        [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": state["request"]},
        ]
    )
    return {"plan": response.content, "messages": [response]}


# Scheduler chỉ cần 2 domain đầu (calendar, dining) — đúng phạm vi "đặt lịch ăn
# tối" của demo này, không phải toàn bộ 15 tool (đó là việc của Hierarchical).
_SCHEDULER_TOOLS = TOOL_GROUPS["calendar"] + TOOL_GROUPS["dining"]

_SCHEDULER_SYSTEM = """Bạn là Scheduler — THỰC THI kế hoạch bằng tool lịch/nhà
hàng. Đọc kế hoạch đã có, gọi đúng tool cần thiết. KHÔNG tự ý lập lại kế
hoạch khác với những gì Planner đã đưa ra."""


@lru_cache(maxsize=1)
def _scheduler_llm():
    return base_llm().bind_tools(_SCHEDULER_TOOLS)


async def scheduler_node(state: SequentialState) -> dict:
    response = await _scheduler_llm().ainvoke(
        [
            {"role": "system", "content": _SCHEDULER_SYSTEM},
            {"role": "user", "content": f"Kế hoạch cần thực thi:\n{state['plan']}"},
        ]
    )
    return {"messages": [response]}


def should_execute_tools(state: SequentialState) -> str:
    last = state["messages"][-1]
    return "scheduler_tools" if getattr(last, "tool_calls", None) else "summarize_schedule"


def summarize_schedule_node(state: SequentialState) -> dict:
    """Sau khi Scheduler (có thể) đã chạy tool, gộp kết quả thành text cho Notifier đọc."""
    tool_outputs = [
        m.content for m in state["messages"] if getattr(m, "type", None) == "tool"
    ]
    result = "\n".join(tool_outputs) if tool_outputs else state["messages"][-1].content
    return {"schedule_result": result}


_NOTIFIER_SYSTEM = """Bạn là Notifier — thông báo NGẮN GỌN cho user kết quả đã
thực thi. Nếu có tool cần phê duyệt (side-effect) trong kết quả, nhắc user
kiểm tra lại. Không lặp lại toàn bộ chi tiết kỹ thuật, chỉ tóm tắt điều user
cần biết."""


def notifier_node(state: SequentialState) -> dict:
    response = base_llm().invoke(
        [
            {"role": "system", "content": _NOTIFIER_SYSTEM},
            {
                "role": "user",
                "content": f"Yêu cầu gốc: {state['request']}\n\nKết quả thực thi:\n{state['schedule_result']}",
            },
        ]
    )
    return {"notification": response.content, "messages": [response]}


@lru_cache(maxsize=1)
def _build_graph():
    graph = StateGraph(SequentialState)
    graph.add_node("planner", planner_node)
    graph.add_node("scheduler", scheduler_node)
    graph.add_node("scheduler_tools", ToolNode(_SCHEDULER_TOOLS))
    graph.add_node("summarize_schedule", summarize_schedule_node)
    graph.add_node("notifier", notifier_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "scheduler")
    graph.add_conditional_edges(
        "scheduler",
        should_execute_tools,
        {"scheduler_tools": "scheduler_tools", "summarize_schedule": "summarize_schedule"},
    )
    graph.add_edge("scheduler_tools", "summarize_schedule")
    graph.add_edge("summarize_schedule", "notifier")
    graph.add_edge("notifier", END)

    # Không cần interrupt_before ở demo Bài 6 (trọng tâm là PATTERN điều phối,
    # HITL đã minh hoạ đủ ở Bài 2) — nhưng recursion_limit vẫn nên set khi invoke
    # cho mọi multi-agent graph (bài học Section 3, cảnh báo vòng lặp vô hạn).
    return graph.compile()


async def run_sequential(request: str) -> dict:
    """Chạy toàn bộ pipeline Planner → Scheduler → Notifier cho 1 yêu cầu.

    Trả về đủ 3 sản phẩm trung gian (plan/schedule_result/notification) để
    UI/API thấy được TỪNG BƯỚC, không chỉ kết quả cuối — đúng tinh thần "thấy
    được con đường đi" đã nhấn mạnh ở Bài 5 (Trajectory Evaluation).
    """
    app = _build_graph()
    result = await app.ainvoke(
        {"request": request, "messages": [], "plan": "", "schedule_result": "", "notification": ""},
        config={"recursion_limit": 15},
    )
    return {
        "plan": result["plan"],
        "schedule_result": result["schedule_result"],
        "notification": result["notification"],
    }
