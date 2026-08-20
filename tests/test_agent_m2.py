"""Test Module II, Bài 2 — Building Agents với LangGraph (ReAct loop, HITL).

Mock `_llm_with_tools()` — LangGraph (graph, ToolNode, checkpointer, interrupt)
chạy THẬT để verify control flow: agent → tools (chờ duyệt) → tools chạy → agent → END.

Bài 4: agent_node/_llm_with_tools/start_conversation/resume_conversation đều
async (agent_node await MCP client) — FakeLLM dùng `ainvoke`, test dùng `async def`.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolCall

from app.agent_m2 import graph as graph_mod
from app.agent_m2 import nodes


def _ai_message_with_tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id=call_id)])


def _ai_message_text(text: str) -> AIMessage:
    return AIMessage(content=text)


# ── should_continue / repetition detection ───────────────────────────────────

def test_should_continue_routes_to_tools_when_tool_call_present():
    state = {"messages": [_ai_message_with_tool_call("check_calendar", {"date": "2026-08-11"})]}
    assert nodes.should_continue(state) == "tools"


def test_should_continue_ends_when_no_tool_call():
    from langgraph.graph import END

    state = {"messages": [_ai_message_text("Xin chào!")]}
    assert nodes.should_continue(state) == END


def test_detect_repetition_true_when_same_call_repeated():
    call = _ai_message_with_tool_call("search_restaurant", {"location": "Hà Nội"})
    state = {"messages": [call, call, call, call]}
    assert nodes._detect_repetition(state, window=4) is True


def test_detect_repetition_false_when_calls_differ():
    state = {
        "messages": [
            _ai_message_with_tool_call("check_calendar", {"date": "2026-08-11"}),
            _ai_message_with_tool_call("check_calendar", {"date": "2026-08-12"}),
            _ai_message_with_tool_call("check_calendar", {"date": "2026-08-13"}),
            _ai_message_with_tool_call("check_calendar", {"date": "2026-08-14"}),
        ]
    }
    assert nodes._detect_repetition(state, window=4) is False


def test_detect_repetition_false_when_not_enough_history():
    state = {"messages": [_ai_message_with_tool_call("check_calendar", {"date": "2026-08-11"})]}
    assert nodes._detect_repetition(state, window=4) is False


# ── agent_node ────────────────────────────────────────────────────────────────

async def test_agent_node_calls_llm_and_appends_message(monkeypatch):
    fake_response = _ai_message_text("Chào bạn!")

    class FakeLLM:
        async def ainvoke(self, messages):
            return fake_response

    async def fake_llm_with_tools(query=""):
        return FakeLLM()

    monkeypatch.setattr(nodes, "_llm_with_tools", fake_llm_with_tools)

    out = await nodes.agent_node({"messages": [{"role": "user", "content": "Xin chào"}]})
    assert out == {"messages": [fake_response]}


# ── Full graph: ReAct loop with HITL interrupt ───────────────────────────────

async def test_full_graph_stops_before_tool_call_pending_approval(monkeypatch):
    """Agent muốn gọi tool → graph phải DỪNG trước tool node (interrupt_before)."""
    graph_mod._build_graph.cache_clear()

    call = _ai_message_with_tool_call("send_reminder", {"message": "Họp", "time": "15:00"})

    class FakeLLM:
        async def ainvoke(self, messages):
            return call

    async def fake_llm_with_tools(query=""):
        return FakeLLM()

    monkeypatch.setattr(nodes, "_llm_with_tools", fake_llm_with_tools)

    result = await graph_mod.start_conversation("thread-1", "Nhắc tôi họp lúc 15h")

    assert result["status"] == "pending_approval"
    assert result["tool_call"]["name"] == "send_reminder"
    assert result["tool_call"]["args"] == {"message": "Họp", "time": "15:00"}
    graph_mod._build_graph.cache_clear()


async def test_full_graph_resumes_and_runs_tool_after_approval(monkeypatch):
    """Sau khi duyệt (approve=True), graph chạy tiếp: tool thực thi → agent trả lời."""
    graph_mod._build_graph.cache_clear()

    call = _ai_message_with_tool_call("check_calendar", {"date": "2026-08-11"}, call_id="call_x")
    responses = iter([call, _ai_message_text("Chiều nay bạn có 2 cuộc họp.")])

    class FakeLLM:
        async def ainvoke(self, messages):
            return next(responses)

    async def fake_llm_with_tools(query=""):
        return FakeLLM()

    monkeypatch.setattr(nodes, "_llm_with_tools", fake_llm_with_tools)

    pending = await graph_mod.start_conversation("thread-2", "Chiều nay tôi có lịch gì?")
    assert pending["status"] == "pending_approval"

    result = await graph_mod.resume_conversation("thread-2", approve=True)
    assert result["status"] == "done"
    assert result["answer"] == "Chiều nay bạn có 2 cuộc họp."
    graph_mod._build_graph.cache_clear()


async def test_full_graph_rejection_injects_tool_message_instead_of_running(monkeypatch):
    """Từ chối (approve=False) → tool KHÔNG chạy thật, agent thấy lý do từ chối."""
    graph_mod._build_graph.cache_clear()

    call = _ai_message_with_tool_call("send_reminder", {"message": "Gọi khách", "time": "16h"}, call_id="call_y")
    responses = iter([call, _ai_message_text("Đã huỷ lời nhắc theo yêu cầu.")])

    seen_messages = []

    class FakeLLM:
        async def ainvoke(self, messages):
            seen_messages.append(messages)
            return next(responses)

    async def fake_llm_with_tools(query=""):
        return FakeLLM()

    monkeypatch.setattr(nodes, "_llm_with_tools", fake_llm_with_tools)

    pending = await graph_mod.start_conversation("thread-3", "Nhắc tôi gọi khách lúc 16h")
    assert pending["status"] == "pending_approval"

    result = await graph_mod.resume_conversation("thread-3", approve=False, rejection_note="chưa cần")
    assert result["status"] == "done"
    assert result["answer"] == "Đã huỷ lời nhắc theo yêu cầu."

    # Agent's 2nd call phải thấy được tool message chứa lý do từ chối.
    last_call_messages = seen_messages[-1]
    assert any("TỪ CHỐI" in str(m) for m in last_call_messages)
    graph_mod._build_graph.cache_clear()


async def test_memory_persists_across_turns_same_thread(monkeypatch):
    """Checkpointer (Section 5): 2 lượt invoke cùng thread_id → agent thấy lịch sử cũ."""
    graph_mod._build_graph.cache_clear()

    seen_message_counts = []
    responses = iter([_ai_message_text("Chào Minh!"), _ai_message_text("Bạn tên Minh.")])

    class FakeLLM:
        async def ainvoke(self, messages):
            seen_message_counts.append(len(messages))
            return next(responses)

    async def fake_llm_with_tools(query=""):
        return FakeLLM()

    monkeypatch.setattr(nodes, "_llm_with_tools", fake_llm_with_tools)

    await graph_mod.start_conversation("thread-4", "Tôi tên Minh.")
    await graph_mod.start_conversation("thread-4", "Tôi tên gì?")

    # Lượt 2 phải thấy nhiều message hơn lượt 1 (system + lịch sử cũ + câu mới).
    assert seen_message_counts[1] > seen_message_counts[0]
    graph_mod._build_graph.cache_clear()
