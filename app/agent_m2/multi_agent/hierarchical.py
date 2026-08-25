"""Hierarchical (Supervisor) pattern — Module II, Bài 6, Section 1 + 3 (Hands-on).

Một Supervisor nhận task, QUYẾT ĐỊNH giao cho Worker nào (định tuyến động —
khác Sequential vốn thứ tự CỐ ĐỊNH), tổng hợp kết quả. Worker KHÔNG giao tiếp
trực tiếp với nhau — mọi thứ đi qua Supervisor. Đây là pattern capstone Bài 7
dùng (Planner → Researcher/Analyzer/Writer), nên demo này bám sát ví dụ
handbook nhất, chỉ thay `search_web` giả bằng tool THẬT đã có (tools.py).

    START → supervisor ─┬─(calendar)→ [calendar SUBGRAPH] ─┐
                         ├─(dining)  → [dining SUBGRAPH]   ─┼→ supervisor → ... → END
                         └─(wellness)→ [wellness SUBGRAPH] ─┘

Supervisor tự quyết định worker tiếp theo dựa trên task + kết quả đã có —
vòng lặp supervisor → worker → supervisor có thể chạy vô hạn nếu Supervisor
không bao giờ chọn "done" (bài học Section 3, warning) → LUÔN set
recursion_limit khi invoke.

── Cô lập state bằng SUBGRAPH (khác bản đầu tiên của file này) ──────────────

Bản đầu: mọi worker là node PHẲNG trong CÙNG StateGraph(HierarchicalState),
dùng chung 1 list `messages` toàn graph. Hệ quả (bug thật đã xảy ra và phải
vá): worker B chạy sau worker A có thể đọc nhầm ToolMessage của A nếu lọc
không đủ cẩn thận — vì không có ranh giới nào ngăn nó, chỉ có kỷ luật code.
Bản chất: 1 node LangGraph chỉ là 1 hàm (state) -> dict, không hơn 1 node
function bình thường — "shared state" ở LangGraph nghĩa là mọi node cùng ghi
vào 1 blackboard, không phải mỗi agent có bộ nhớ riêng.

Bản này: mỗi Worker là 1 SUBGRAPH ĐỘC LẬP với `WorkerState` RIÊNG (messages
nội bộ, tool-call bookkeeping — hoàn toàn không tồn tại trong HierarchicalState
của Supervisor). Khi 1 compiled subgraph được add làm node của graph cha,
LangGraph CHỈ truyền qua các field TRÙNG TÊN giữa 2 schema (ở đây: `task` đi
vào, `result` đi ra) — mọi field nội bộ khác của WorkerState (`messages`,
`_pending_tool_call_ids`) không hề lộ ra Supervisor, và Supervisor cũng không
lộ `notes`/`next_agent` vào trong subgraph. Đây là ranh giới CƠ CHẾ (LangGraph
tự đảm bảo), không phải kỷ luật tự giác như bản trước.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app.agent_m2.multi_agent._llm import base_llm
from app.agent_m2.tools import TOOL_GROUPS

# 3 domain worker cho demo (đủ minh hoạ định tuyến động; capstone Bài 7 dùng
# cùng cơ chế này với domain khác: Researcher/Analyzer/Writer).
_WORKER_DOMAINS = ["calendar", "dining", "wellness"]


# ── Worker: state RIÊNG, không chia sẻ với Supervisor hay worker khác ────────

class WorkerState(TypedDict):
    """State NỘI BỘ của 1 worker subgraph.

    Chỉ `task` (đọc) và `result` (ghi) là 2 field "cửa sổ" ra bên ngoài — trùng
    tên với field tương ứng trong HierarchicalState nên LangGraph tự truyền
    qua khi nhúng subgraph. `messages` ở đây là lịch sử hội thoại NỘI BỘ của
    riêng worker (system prompt, tool call, tool result) — không đụng gì tới
    `notes`/`next_agent` của Supervisor, và Supervisor cũng không thấy được
    những message trung gian này (chỉ thấy `result` cuối cùng).
    """

    task: str
    messages: Annotated[list, add_messages]
    result: str


def _make_worker_graph(domain: str):
    """Build 1 subgraph hoàn chỉnh cho domain — compile 1 lần, tái dùng.

    3 node nội bộ (act → tools → summarize) chỉ nhìn thấy WorkerState, hoàn
    toàn độc lập với 2 worker domain khác dù chạy trong cùng 1 process/thread_id.
    """
    worker_tools = TOOL_GROUPS[domain]

    @lru_cache(maxsize=1)
    def _worker_llm():
        return base_llm().bind_tools(worker_tools)

    async def act(state: WorkerState) -> dict:
        messages = state["messages"] or [
            {
                "role": "system",
                "content": f"Bạn là worker chuyên domain '{domain}'. Thực hiện phần việc của "
                f"nhiệm vụ sau liên quan tới domain này: {state['task']}",
            }
        ]
        response = await _worker_llm().ainvoke(messages)
        return {"messages": [response]}

    def route_tools(state: WorkerState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "summarize"

    def summarize(state: WorkerState) -> dict:
        # State RIÊNG của subgraph này — an toàn đọc toàn bộ messages, không lo
        # lẫn dữ liệu worker khác vì list này KHÔNG hề chung với ai.
        last = state["messages"][-1]
        tool_msgs = [m.content for m in state["messages"] if getattr(m, "type", None) == "tool"]
        # Có ToolMessage → ưu tiên nội dung đó; không thì model đã trả lời thẳng (không cần tool).
        result = "\n".join(tool_msgs) if tool_msgs else last.content
        return {"result": result}

    graph = StateGraph(WorkerState)
    graph.add_node("act", act)
    graph.add_node("tools", ToolNode(worker_tools))
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "act")
    graph.add_conditional_edges("act", route_tools, {"tools": "tools", "summarize": "summarize"})
    graph.add_edge("tools", "act")  # sau khi có tool result, quay lại act để worker tổng hợp
    graph.add_edge("summarize", END)

    return graph.compile()


# ── Supervisor: state RIÊNG cho tầng điều phối ───────────────────────────────

class HierarchicalState(TypedDict):
    """State của Supervisor — KHÔNG chứa `messages` nội bộ của worker nào.

    `task`/`result` trùng tên với WorkerState là CÓ CHỦ Ý (xem docstring
    module) — đây là "hợp đồng" duy nhất giữa Supervisor và mỗi worker.
    """

    task: str
    result: str  # field TẠM để nhận output từ worker vừa chạy, trước khi gộp vào notes
    notes: dict[str, str]  # domain -> kết quả worker đã trả (Supervisor đọc để quyết định)
    next_agent: str  # "calendar" | "dining" | "wellness" | "done"
    final_answer: str


class _RoutingDecision(BaseModel):
    next_agent: str = Field(description="Một trong: calendar, dining, wellness, done")
    reasoning: str = Field(description="Lý do ngắn gọn cho quyết định")


_SUPERVISOR_SYSTEM = """Bạn điều phối 3 worker chuyên biệt: 'calendar' (lịch làm
việc, sự kiện), 'dining' (nhà hàng, đặt bàn), 'wellness' (tập luyện, giấc ngủ,
phòng gym). Worker KHÔNG tự giao tiếp với nhau — bạn quyết định worker nào
chạy tiếp theo dựa trên nhiệm vụ và ghi chú đã có.

Chọn 'done' khi đã đủ thông tin để trả lời user, kể cả khi chỉ cần 1 worker."""


async def supervisor_node(state: HierarchicalState) -> dict:
    notes_text = "\n".join(f"[{d}] {n}" for d, n in state["notes"].items()) or "(chưa có)"
    prompt = f"Nhiệm vụ: {state['task']}\n\nGhi chú từ worker đã chạy:\n{notes_text}"

    decision = await base_llm().with_structured_output(_RoutingDecision).ainvoke(
        [{"role": "system", "content": _SUPERVISOR_SYSTEM}, {"role": "user", "content": prompt}]
    )
    next_agent = decision.next_agent if decision.next_agent in [*_WORKER_DOMAINS, "done"] else "done"
    return {"next_agent": next_agent}


def route_supervisor(state: HierarchicalState) -> str:
    return state["next_agent"] if state["next_agent"] in _WORKER_DOMAINS else "final_answer"


def _make_collect_node(domain: str):
    """Chạy NGAY SAU subgraph của `domain` — gộp `result` (field tạm, do
    subgraph vừa ghi) vào `notes[domain]` rồi quay lại Supervisor.

    Tách riêng khỏi worker subgraph vì `notes` (dict theo domain) thuộc về
    tầng Supervisor, không phải khái niệm worker cần biết tới.
    """

    def collect(state: HierarchicalState) -> dict:
        return {"notes": {**state["notes"], domain: state["result"]}}

    return collect


_FINAL_ANSWER_SYSTEM = """Tổng hợp ghi chú từ các worker thành câu trả lời cuối
cho user — ngắn gọn, đầy đủ thông tin đã thu thập được."""


def final_answer_node(state: HierarchicalState) -> dict:
    notes_text = "\n".join(f"[{d}] {n}" for d, n in state["notes"].items())
    response = base_llm().invoke(
        [
            {"role": "system", "content": _FINAL_ANSWER_SYSTEM},
            {"role": "user", "content": f"Nhiệm vụ: {state['task']}\n\nGhi chú:\n{notes_text}"},
        ]
    )
    return {"final_answer": response.content}


@lru_cache(maxsize=1)
def _build_graph():
    graph = StateGraph(HierarchicalState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("final_answer", final_answer_node)

    routing_map = {"final_answer": "final_answer"}
    for domain in _WORKER_DOMAINS:
        # Nhúng THẲNG compiled subgraph làm 1 node — LangGraph tự khớp field
        # theo tên (task vào, result ra) giữa WorkerState và HierarchicalState.
        graph.add_node(f"{domain}_worker", _make_worker_graph(domain))
        graph.add_node(f"{domain}_collect", _make_collect_node(domain))

        graph.add_edge(f"{domain}_worker", f"{domain}_collect")
        graph.add_edge(f"{domain}_collect", "supervisor")  # Quay lại Supervisor sau khi worker xong
        routing_map[domain] = f"{domain}_worker"

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_supervisor, routing_map)
    graph.add_edge("final_answer", END)

    return graph.compile()


async def run_hierarchical(task: str) -> dict:
    """Chạy Supervisor điều phối worker cho tới khi quyết định 'done'.

    recursion_limit bắt buộc (bài học Section 3) — vòng supervisor→worker→
    supervisor có thể vô hạn nếu Supervisor không bao giờ chọn done.
    """
    app = _build_graph()
    result = await app.ainvoke(
        {"task": task, "result": "", "notes": {}, "next_agent": "", "final_answer": ""},
        config={"recursion_limit": 15},
    )
    return {"notes": result["notes"], "answer": result["final_answer"]}


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
    # python -m app.agent_m2.multi_agent.hierarchical
    save_graph_visualization("images/hierarchical_agent_graph.png")