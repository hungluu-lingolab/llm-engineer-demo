"""Agentic RAG graph — Buổi 6 (CRAG + Query Decomposition), dùng LangGraph.

LangGraph CHỈ orchestrate control flow. Mọi lời gọi thực tế (LLM, retriever, web
search) đều xuyên qua native SDK đã xây ở app/llm, app/retrieval, app/agent/tools_web.

Luồng (retrieve chạy SONG SONG cho từng sub-question qua LangGraph Send API):
    decompose_query
        → send_retrieve (router: fan-out 1 Send/sub-question)
        → parallel_retrieve (N nhánh chạy song song)
        → dedupe_documents (fan-in: gộp + khử trùng sau khi hội tụ)
        → grade_documents ─┬─(đủ relevant)→ generate → END
                            └─(thiếu)→ web_search → generate → END

So với RAG cố định (Buổi 5): thêm khả năng "quyết định" — có chunk đủ tốt không,
nếu không thì tự tìm thêm trên web trước khi generate.
"""

from __future__ import annotations

from functools import lru_cache

from app.agent.nodes import (
    decompose_query,
    dedupe_documents,
    generate,
    grade_documents,
    parallel_retrieve,
    route_after_grading,
    send_retrieve,
    web_search_node,
)
from app.agent.state import GraphState


@lru_cache(maxsize=1)
def _build_graph():
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(GraphState)

    workflow.add_node("decompose", decompose_query)
    workflow.add_node("parallel_retrieve", parallel_retrieve)
    workflow.add_node("dedupe", dedupe_documents)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("decompose")
    # Conditional edge với router trả về list[Send]: LangGraph fan-out sang
    # node "parallel_retrieve" một lần cho mỗi Send, chạy song song.
    workflow.add_conditional_edges("decompose", send_retrieve, ["parallel_retrieve"])
    # Mọi nhánh song song hội tụ về "dedupe" (fan-in) trước khi grade.
    workflow.add_edge("parallel_retrieve", "dedupe")
    workflow.add_edge("dedupe", "grade")
    workflow.add_conditional_edges(
        "grade",
        route_after_grading,
        {"generate": "generate", "web_search": "web_search"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_agent(query: str) -> dict:
    """Chạy toàn bộ graph cho một câu hỏi, trả về state cuối cùng.

    Trả cả state (không chỉ answer) để caller thấy được documents dùng, có
    web_search_used không, v.v. — hữu ích để debug/hiển thị "vì sao trả lời vậy".
    """
    graph = _build_graph()
    initial_state: GraphState = {"question": query}
    return graph.invoke(initial_state)


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
    # python -m app.agent.graph
    save_graph_visualization("images/agent_graph.png")