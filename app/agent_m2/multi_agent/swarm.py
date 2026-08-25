"""Swarm pattern — Module II, Bài 6, Section 1.

Agent TỰ QUYẾT ĐỊNH chuyển giao (handoff) trực tiếp cho agent khác dựa trên
ngữ cảnh — KHÔNG qua Supervisor trung tâm (khác Hierarchical: ở đó Supervisor
luôn là điểm quyết định duy nhất). Phù hợp khi hội thoại chuyển hướng linh
hoạt, không có "trung tâm điều phối" tự nhiên.

3 agent domain ngang hàng (calendar/dining/wellness) — mỗi agent thấy 2 tool
"handoff" sang 2 agent còn lại, dùng khi câu hỏi lấn sang domain khác:

    calendar_agent ⇄ dining_agent
           ⇅              ⇅
        wellness_agent ────┘

Kỹ thuật: node trả về `Command(goto=..., update=...)` (LangGraph) — vừa CẬP
NHẬT state vừa CHỈ ĐỊNH node kế tiếp trong 1 bước. Đây chính là cơ chế
"handoff" thật (khác Hierarchical dùng `add_conditional_edges` từ 1 node
Supervisor riêng biệt đọc state rồi route — ở Swarm, CHÍNH agent con quyết
định đích đến, không có ai điều phối ở giữa).

Entry point ĐỘNG (agent nào nhận tin nhắn đầu tiên) dùng
`add_conditional_edges(START, ...)` — không dùng `Command` làm input top-level
cho `ainvoke()`, vì LangGraph vẫn chạy qua entry point cố định của
`set_entry_point()` trước khi áp Command, gây ghi đè state 2 lần trong cùng
1 step (`InvalidUpdateError`). Route động phải nằm TRONG graph, không phải ở input.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from app.agent_m2.multi_agent._llm import base_llm
from app.agent_m2.tools import TOOL_GROUPS

_AGENTS = ["calendar", "dining", "wellness"]


class SwarmState(TypedDict):
    messages: Annotated[list, add_messages]
    entry_agent: str  # agent nhận tin nhắn đầu tiên — chọn entry point động (xem module docstring)
    active_agent: str  # agent hiện đang xử lý — UI/test dùng để thấy handoff xảy ra
    handoff_log: Annotated[list[str], lambda a, b: a + b]  # vết handoff, chỉ để quan sát


def _make_handoff_tool(target_agent: str):
    """1 tool "chuyển giao cho agent X" — model gọi tool này CHÍNH LÀ hành động
    handoff, khác tool nghiệp vụ thường (check_calendar, ...) vốn trả dữ liệu.
    Tool này không chạy qua ToolNode — agent_node tự đọc tool_calls và goto
    thẳng, xem _make_agent_node."""

    def _handoff_impl() -> str:
        return f"Đã chuyển giao cho agent {target_agent}."

    _handoff_impl.__name__ = f"handoff_to_{target_agent}"
    _handoff_impl.__doc__ = f"Chuyển giao hội thoại cho agent '{target_agent}' khi câu hỏi thuộc domain đó."
    return tool(_handoff_impl)


def _make_agent_node(agent_name: str):
    """1 agent Swarm = LLM bind [tool domain mình] + [handoff tool tới 2 agent
    còn lại]. Model tự quyết định: trả lời trực tiếp, gọi tool domain mình,
    hoặc gọi handoff tool để chuyển sang agent khác — không ai ép nó."""
    domain_tools = TOOL_GROUPS[agent_name]
    other_agents = [a for a in _AGENTS if a != agent_name]
    handoff_tools = [_make_handoff_tool(a) for a in other_agents]
    all_tools = domain_tools + handoff_tools

    @lru_cache(maxsize=1)
    def _llm():
        return base_llm().bind_tools(all_tools)

    system_prompt = (
        f"Bạn là agent chuyên domain '{agent_name}' trong hệ thống trợ lý cá "
        f"nhân. Nếu câu hỏi thuộc domain khác ({', '.join(other_agents)}), gọi "
        f"tool handoff_to_<agent> tương ứng thay vì cố trả lời sai domain."
    )

    async def agent_node(state: SwarmState) -> Command:
        response = await _llm().ainvoke(
            [{"role": "system", "content": system_prompt}] + state["messages"]
        )

        tool_calls = getattr(response, "tool_calls", None) or []
        handoff_call = next((c for c in tool_calls if c["name"].startswith("handoff_to_")), None)

        if handoff_call:
            target = handoff_call["name"].removeprefix("handoff_to_")
            log_entry = f"{agent_name} → {target}"
            # KHÔNG append response (chứa handoff tool_call) vào messages — không
            # có ToolNode nào chạy tool handoff nên tool_call đó sẽ "mồ côi"
            # (OpenAI API yêu cầu mọi tool_call phải có tool response tương ứng
            # trong lượt sau). Thay vào đó chèn 1 system note cho agent đích.
            return Command(
                goto=target,
                update={
                    "active_agent": target,
                    "handoff_log": [log_entry],
                    "messages": [
                        {
                            "role": "system",
                            "content": f"[Handoff từ {agent_name}]: câu hỏi được chuyển sang domain '{target}'.",
                        }
                    ],
                },
            )

        if tool_calls:
            return Command(goto=f"{agent_name}_tools", update={"messages": [response], "active_agent": agent_name})

        return Command(goto=END, update={"messages": [response], "active_agent": agent_name})

    return agent_node


def _route_entry(state: SwarmState) -> str:
    return state["entry_agent"] if state["entry_agent"] in _AGENTS else "calendar"


@lru_cache(maxsize=1)
def _build_graph():
    graph = StateGraph(SwarmState)

    for agent_name in _AGENTS:
        graph.add_node(agent_name, _make_agent_node(agent_name))
        graph.add_node(f"{agent_name}_tools", ToolNode(TOOL_GROUPS[agent_name]))
        # Sau khi chạy tool nghiệp vụ, quay lại CHÍNH agent đó để tổng hợp trả lời
        # (không tự động handoff — model có thể handoff ở lượt invoke tiếp theo).
        graph.add_edge(f"{agent_name}_tools", agent_name)

    # Entry point ĐỘNG: route từ START dựa trên state["entry_agent"] — không
    # dùng set_entry_point() cố định, vì mỗi lần gọi run_swarm() có thể chọn
    # agent nhận tin nhắn đầu tiên khác nhau (xem module docstring).
    graph.add_conditional_edges(START, _route_entry, {a: a for a in _AGENTS})

    return graph.compile()


async def run_swarm(message: str, entry_agent: str = "calendar") -> dict:
    """Gửi 1 tin nhắn, để hệ thống Swarm tự route/handoff tới agent phù hợp.

    `entry_agent`: agent nhận tin nhắn ĐẦU TIÊN — mô phỏng "kênh liên hệ" user
    chọn (vd chat trong tab Dining thì entry_agent='dining'). Agent đó có thể
    tự handoff tiếp nếu câu hỏi không thuộc domain mình.
    """
    if entry_agent not in _AGENTS:
        entry_agent = "calendar"

    app = _build_graph()
    result = await app.ainvoke(
        {
            "messages": [{"role": "user", "content": message}],
            "entry_agent": entry_agent,
            "active_agent": entry_agent,
            "handoff_log": [],
        },
        config={"recursion_limit": 15},
    )
    return {
        "answer": result["messages"][-1].content,
        "final_agent": result["active_agent"],
        "handoff_log": result["handoff_log"],
    }

def save_graph_visualization(path: str = "app/agent/graph.png") -> str:
    """Xuất sơ đồ graph ra file — hữu ích để debug/trình bày cấu trúc CRAG.

    Thử vẽ PNG trước (draw_mermaid_png — gọi API mermaid.ink, cần mạng).
    Nếu không có mạng/lỗi, fallback ghi ra Mermaid text thuần (.mmd, không cần mạng) —
    dán vào https://mermaid.live hoặc preview trực tiếp trong VSCode/GitHub.

    Returns:
        Đường dẫn file thực sự đã ghi (có thể khác `path` nếu fallback sang .mmd).
    """
    graph = _build_graph().get_graph()

    try:
        png_bytes = graph.draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_bytes)
        return path
    except Exception:
        # Offline hoặc mermaid.ink không khả dụng — fallback text thuần, luôn thành công.
        mmd_path = path.rsplit(".", 1)[0] + ".mmd"
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(graph.draw_mermaid())
        return mmd_path

if __name__ == "__main__":
    # python -m app.agent_m2.multi_agent.swarm
    save_graph_visualization("images/swarm_agent_graph.png")