"""Test Module II, Bài 6 — Multi-Agent Systems.

End-to-end, giản lược theo yêu cầu: 1 test/pattern chứng minh graph chạy
ĐÚNG LUỒNG (không lặp vô hạn, đi đúng node, trả đúng field) — không unit-test
chi tiết từng node như các bài trước. Mock base_llm() (native LLM client
dùng chung 4 pattern) bằng FakeLLM trả response theo kịch bản cố định.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agent_m2.multi_agent import collaborative, hierarchical, sequential, swarm


def _ai(content: str = "", tool_calls: list | None = None) -> AIMessage:
    return AIMessage(content=content, tool_calls=tool_calls or [])


# ── Sequential: Planner → Scheduler → Notifier ────────────────────────────────

async def test_sequential_runs_full_pipeline(monkeypatch):
    """3 bước cố định chạy đúng thứ tự, không quay lại (Section 1: Sequential)."""
    sequential._build_graph.cache_clear()

    responses = iter(
        [
            _ai("Kế hoạch: đặt bàn tối thứ 6."),  # planner_node (base_llm().invoke, sync)
            _ai("", tool_calls=[{"name": "check_calendar", "args": {"start_date": "2026-08-20"}, "id": "c1"}]),  # scheduler
            _ai("Đã kiểm tra lịch, rảnh cả ngày."),  # notifier_node
        ]
    )

    class FakeLLM:
        def invoke(self, messages):
            return next(responses)

        async def ainvoke(self, messages):
            return next(responses)

        def bind_tools(self, tools):
            return self

    monkeypatch.setattr(sequential, "base_llm", lambda: FakeLLM())

    from app.agent_m2 import tools as tools_mod
    monkeypatch.setattr(
        tools_mod.check_calendar, "func",
        lambda start_date, end_date="": "Rảnh cả ngày.", raising=False,
    )

    result = await sequential.run_sequential("Đặt bàn tối thứ 6 cho 4 người")

    assert result["plan"] == "Kế hoạch: đặt bàn tối thứ 6."
    assert "Rảnh cả ngày" in result["schedule_result"]
    assert result["notification"] == "Đã kiểm tra lịch, rảnh cả ngày."
    sequential._build_graph.cache_clear()


# ── Hierarchical: Supervisor điều phối Worker theo domain ────────────────────

async def test_hierarchical_supervisor_routes_to_worker_then_done(monkeypatch):
    """Supervisor chọn đúng worker, worker chạy tool, Supervisor thấy đủ → done (Section 1: Hierarchical)."""
    hierarchical._build_graph.cache_clear()

    class _Decision:
        def __init__(self, next_agent):
            self.next_agent = next_agent
            self.reasoning = "test"

    decisions = iter([_Decision("dining"), _Decision("done")])

    class FakeStructuredLLM:
        async def ainvoke(self, messages):
            return next(decisions)

    # Worker subgraph giờ chạy act -> tools -> act -> summarize: lượt "act" đầu
    # gọi tool, lượt "act" thứ 2 (sau khi có ToolMessage) phải trả TEXT thuần
    # để route_tools thoát vòng lặp và đi tới summarize.
    worker_act_calls = {"n": 0}

    class FakeBaseLLM:
        def with_structured_output(self, schema):
            return FakeStructuredLLM()

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            worker_act_calls["n"] += 1
            if worker_act_calls["n"] == 1:
                return _ai("", tool_calls=[{"name": "search_restaurant", "args": {"location": "Q1"}, "id": "c1"}])
            return _ai("Đã tìm xong nhà hàng.")

        def invoke(self, messages):
            return _ai("Gợi ý: Quán A, Quán B.")

    fake = FakeBaseLLM()
    monkeypatch.setattr(hierarchical, "base_llm", lambda: fake)

    from app.agent_m2 import tools as tools_mod
    monkeypatch.setattr(
        tools_mod.search_restaurant, "func",
        lambda location, cuisine="": "3 nhà hàng gần Q1: A, B, C.", raising=False,
    )

    result = await hierarchical.run_hierarchical("Tìm nhà hàng gần Quận 1")

    assert "dining" in result["notes"]
    assert "A, B, C" in result["notes"]["dining"]
    assert result["answer"] == "Gợi ý: Quán A, Quán B."
    hierarchical._build_graph.cache_clear()


# ── Collaborative: Planner ⇄ Critic phản biện lặp vòng ────────────────────────

async def test_collaborative_stops_when_critic_approves(monkeypatch):
    """Vòng lặp dừng NGAY khi Critic duyệt — không chạy hết MAX_ROUNDS nếu không cần (Section 1: Collaborative)."""
    collaborative._build_graph.cache_clear()

    class _Verdict:
        def __init__(self, approved, feedback):
            self.approved = approved
            self.feedback = feedback

    class FakeStructuredLLM:
        async def ainvoke(self, messages):
            return _Verdict(True, "Kế hoạch ổn.")

    class FakeBaseLLM:
        def with_structured_output(self, schema):
            return FakeStructuredLLM()

        async def ainvoke(self, messages):
            return _ai("Kế hoạch: đi cafe rồi ăn tối.")

    monkeypatch.setattr(collaborative, "base_llm", lambda: FakeBaseLLM())

    result = await collaborative.run_collaborative("Lên kế hoạch hẹn hò cuối tuần")

    assert result["approved"] is True
    assert result["rounds"] == 1  # duyệt ngay vòng đầu, không lặp thêm
    collaborative._build_graph.cache_clear()


async def test_collaborative_stops_at_max_rounds_when_never_approved(monkeypatch):
    """Không bao giờ duyệt → dừng an toàn ở MAX_ROUNDS, KHÔNG lặp vô hạn (Section 3, warning)."""
    collaborative._build_graph.cache_clear()

    class _Verdict:
        approved = False
        feedback = "Vẫn còn vấn đề."

    class FakeStructuredLLM:
        async def ainvoke(self, messages):
            return _Verdict()

    class FakeBaseLLM:
        def with_structured_output(self, schema):
            return FakeStructuredLLM()

        async def ainvoke(self, messages):
            return _ai("Kế hoạch (chưa đạt).")

    monkeypatch.setattr(collaborative, "base_llm", lambda: FakeBaseLLM())

    result = await collaborative.run_collaborative("Lên kế hoạch hẹn hò cuối tuần")

    assert result["approved"] is False
    assert result["rounds"] == collaborative.MAX_ROUNDS
    collaborative._build_graph.cache_clear()


# ── Swarm: agent ngang hàng tự handoff qua Command ────────────────────────────

async def test_swarm_handoff_between_agents(monkeypatch):
    """calendar agent nhận câu hỏi ngoài domain → handoff sang dining, dining trả lời (Section 1: Swarm)."""
    swarm._build_graph.cache_clear()

    call_count = {"n": 0}

    class FakeLLM:
        async def ainvoke(self, messages):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _ai("", tool_calls=[{"name": "handoff_to_dining", "args": {}, "id": "h1"}])
            return _ai("3 nhà hàng gần Quận 1: A, B, C.")

    fake = FakeLLM()

    class FakeBase:
        def bind_tools(self, tools):
            return fake

    monkeypatch.setattr(swarm, "base_llm", lambda: FakeBase())

    result = await swarm.run_swarm("Tìm nhà hàng gần Quận 1", entry_agent="calendar")

    assert result["final_agent"] == "dining"
    assert result["handoff_log"] == ["calendar → dining"]
    assert "A, B, C" in result["answer"]
    swarm._build_graph.cache_clear()


async def test_swarm_answers_directly_without_handoff(monkeypatch):
    """Câu hỏi đúng domain entry_agent → trả lời trực tiếp, KHÔNG handoff."""
    swarm._build_graph.cache_clear()

    class FakeLLM:
        async def ainvoke(self, messages):
            return _ai("Chiều nay bạn rảnh.")

    fake = FakeLLM()

    class FakeBase:
        def bind_tools(self, tools):
            return fake

    monkeypatch.setattr(swarm, "base_llm", lambda: FakeBase())

    result = await swarm.run_swarm("Chiều nay tôi có lịch gì?", entry_agent="calendar")

    assert result["final_agent"] == "calendar"
    assert result["handoff_log"] == []
    swarm._build_graph.cache_clear()
