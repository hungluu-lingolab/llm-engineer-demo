"""Nodes — Module II, Bài 2-4 (LangGraph + Memory/Context + Tool Design).

Khác với app/agent/nodes.py (CRAG, Module I) vốn gọi thẳng app/llm/completion.py
(native OpenAI SDK): agent này dùng `ChatOpenAI.bind_tools()` (langchain_openai)
vì `ToolNode` (LangGraph prebuilt, Section 2) chỉ tự thực thi tool khi model được
gọi qua interface LangChain — đây là ngoại lệ CÓ CHỦ ĐÍCH so với triết lý "native
SDK" của README, đổi lấy vòng lặp tool-calling khỏi phải viết tay (đã viết tay ở
Bài 1/tools/registry.py, và ở app/agent/nodes.py bằng chat_with_tools).

Bài 3 thêm 2 node quanh vòng lặp agent⇄tools:
  - recall_node (đầu):  đọc long-term memory liên quan → chèn làm context.
  - extract_and_store_node (cuối): trích 1 sự thật đáng nhớ → lưu long-term.
Và nâng cấp agent_node: sliding window + summarization + re-inject chỉ dẫn
(chống context rot, Section 2-4) — mọi thao tác này ở context.py (native LLM).

Bài 4 (Tool Design & Integration, Section 2): 15 tool (tools.py) đã ở vùng
"tool sprawl" theo bài học → agent_node KHÔNG bind cố định toàn bộ TOOLS nữa,
mà bind theo top-k tool liên quan nhất tới câu hỏi hiện tại
(tool_selection.retrieve_relevant_tools, embedding similarity). ToolNode
(graph.py) vẫn được khởi tạo với ĐẦY ĐỦ TOOLS — nó cần biết cách THỰC THI mọi
tool có thể được gọi, chỉ có phần BIND cho LLM (những gì model được "thấy" khi
quyết định) mới bị giới hạn qua retrieval.

Bài 4, Section 4 (MCP): retrieve_relevant_tools còn gộp cả tool từ MCP server
độc lập (mcp/wellness_server.py qua mcp/client.py) — MCP client dùng giao thức
JSON-RPC ASYNC (không có bản sync), nên agent_node ở đây chuyển thành `async
def`, graph.py dùng `ainvoke` thay vì `invoke` (khác Bài 2-3 vốn sync hoàn toàn).
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

from langgraph.graph import END

from app.agent_m2 import context, memory
from app.agent_m2.state import AssistantState
from app.agent_m2.tool_selection import retrieve_relevant_tools
from app.config import settings

# Chỉ dẫn cốt lõi — dùng CẢ cho system prompt (đầu context) VÀ re-injection
# (cuối context, chống instruction fade-out khi hội thoại dài — Section 4).
CORE_INSTRUCTIONS = (
    "Bạn chỉ thấy MỘT PHẦN tool phù hợp nhất với câu hỏi hiện tại, không phải "
    "toàn bộ tool hệ thống có. Với câu hỏi về một khoảng thời gian (vd: 'tuần "
    "này'), tự tính start_date/end_date rồi gọi check_calendar MỘT LẦN DUY "
    "NHẤT — không gọi lặp lại cho từng ngày riêng lẻ."
)


def _system_prompt() -> str:
    # Chèn ngày hiện tại — không có dòng này, model không biết "hôm nay" là
    # ngày nào nên hay bịa năm/tháng sai khi tính "tuần này"/"ngày mai".
    return (
        "Bạn là trợ lý cá nhân tiếng Việt, thân thiện và ngắn gọn. "
        f"Hôm nay là {date.today().isoformat()}. " + CORE_INSTRUCTIONS
    )


@lru_cache(maxsize=1)
def _base_llm():
    """LLM chưa bind tool — dùng chung, chỉ khởi tạo client 1 lần (Bài 4: bind
    tool đổi theo từng lượt qua retrieval nên không thể cache llm.bind_tools()
    như Bài 2-3 nữa, nhưng client HTTP vẫn nên tái dùng).

    Truyền thẳng `api_key` từ settings.api_keys[0] thay vì để ChatOpenAI tự đọc
    biến môi trường OPENAI_API_KEY — repo này dùng OPENAI_API_KEYS (số nhiều,
    hỗ trợ key rotation ở app/llm/client.py), không phải biến số ít mặc định
    của langchain_openai.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.api_keys[0] if settings.api_keys else None,
    )


async def _llm_with_tools(query: str):
    """Bind LLM với top-k tool liên quan tới `query` (Bài 4, Section 2: Tool
    Retrieval) — KHÔNG bind cố định toàn bộ tool mỗi lượt, giảm token + tăng
    độ chính xác chọn tool (bài học: >20 tool độ chính xác giảm mạnh).

    async vì retrieve_relevant_tools cần await MCP client (Section 4) ở lần
    load tool đầu tiên (sau đó có cache, nhưng hàm vẫn giữ chữ ký async cho
    nhất quán — awaiting một future đã resolve gần như free)."""
    relevant = await retrieve_relevant_tools(query, k=settings.agent_tool_retrieval_k)
    return _base_llm().bind_tools(relevant)


# ── Loop termination (Section 3): phát hiện agent lặp lại cùng 1 tool call ────

def _detect_repetition(state: AssistantState, window: int = 4) -> bool:
    """True nếu `window` tool call gần nhất giống hệt nhau — dấu hiệu agent bị kẹt."""
    recent_calls = [
        m.tool_calls[0]
        for m in state["messages"][-window:]
        if getattr(m, "tool_calls", None)
    ]
    if len(recent_calls) < window:
        return False
    return len({str(c) for c in recent_calls}) == 1


# ── Compaction — node RIÊNG vì đây là bước DUY NHẤT mutate lịch sử (Section 4) ──
#
# Vì sao tách khỏi agent_node? Hai lý do — một về kỷ luật, một về hiệu năng:
#
#  1. Kỷ luật LangGraph "one node one function": agent_node còn lại chỉ làm ĐÚNG
#     một việc — shape prompt (tạm thời) rồi gọi LLM. Còn summarization là một
#     lời gọi LLM RIÊNG ghi ĐÈ state → xứng đáng là 1 node có thể thấy trên graph.
#
#  2. TÁI SỬ DỤNG bản nén (điểm mấu chốt): nếu để summarize trong agent_node, mỗi
#     vòng lặp agent⇄tools sẽ nén lại history cũ từ đầu (biến local, return chỉ
#     append response) → bản nén KHÔNG được lưu, lượt sau nén lại → tốn LLM call
#     lặp đi lặp lại. compact_node GHI ĐÈ state.messages bằng bản nén (RemoveMessage
#     + summary) → history chính thức đã nén, các vòng sau kế thừa, chỉ nén lại khi
#     history LẠI phình vượt ngưỡng. Đây là cách compaction thật hoạt động
#     (Claude Code, v.v.): nén một lần, dùng lại nhiều lần.


def should_compact_route(state: AssistantState) -> str:
    """Conditional edge sau recall: history vượt ngưỡng 40% → "compact", không → "agent"."""
    if context.should_compact(state["messages"], settings.agent_context_window_tokens):
        return "compact"
    return "agent"


def compact_node(state: AssistantState) -> dict:
    """Nén phần history cũ → GHI ĐÈ state.messages (persist, tái dùng lượt sau).

    Trả về diff cho reducer add_messages: [RemoveMessage(id) cho mỗi message cũ]
    + 1 summary message. add_messages xoá các message theo id rồi thêm summary →
    state.messages sau node = [summary] + keep_recent message gần nhất.

    Chỉ chạy khi should_compact_route dẫn vào đây, nên luôn có phần "cũ" để nén.
    """
    from langchain_core.messages import RemoveMessage, SystemMessage

    messages = state["messages"]
    keep_recent = settings.agent_keep_recent_messages
    if len(messages) <= keep_recent + 1:
        return {}  # an toàn: không đủ để nén (thực tế route đã lọc, đây chỉ là guard)

    old_messages = messages[:-keep_recent]
    summary = context.summarize_text(old_messages)

    # add_messages xoá theo id → cần id thật (message trong checkpointer luôn có id).
    removals = [RemoveMessage(id=m.id) for m in old_messages if getattr(m, "id", None)]
    summary_msg = SystemMessage(content=f"{context.SUMMARY_PREFIX} {summary}")
    return {"messages": removals + [summary_msg]}


async def agent_node(state: AssistantState) -> dict:
    """Gọi LLM (kèm tool binding). CHỈ shape prompt TẠM THỜI rồi gọi model.

    Khác compact_node: mọi thao tác ở đây là EPHEMERAL — dựng một list `messages`
    tạm để gửi cho LLM lần này, KHÔNG ghi lại vào state (return chỉ append response).
    Đó là lý do sliding window / repetition warning / re-inject nằm chung 1 node:
    chúng biến đổi bản-gửi-đi, không phải lịch sử thật. Compaction (mutate state)
    đã tách sang compact_node ở trên.

    Thứ tự shape (Section 2-4):
      - Sliding window: giữ N message gần nhất.
      - Repetition warning: chèn nếu agent lặp cùng 1 tool call.
      - Re-inject CORE_INSTRUCTIONS ở CUỐI (vị trí attention cao — chống fade-out).

    async (Bài 4, Section 4): _llm_with_tools cần await MCP client — LangGraph
    hỗ trợ node async natively, graph.py gọi ainvoke() thay vì invoke().
    """
    history = context.sliding_window(state["messages"], settings.agent_max_messages)

    messages = [{"role": "system", "content": _system_prompt()}] + history

    if _detect_repetition(state):
        messages = messages + [
            {
                "role": "system",
                "content": "Bạn đang lặp lại cùng 1 hành động. Hãy thử cách tiếp cận hoàn toàn khác.",
            }
        ]

    messages = context.reinject_instructions(messages, CORE_INSTRUCTIONS)

    query = _latest_human_query(state["messages"])
    llm = await _llm_with_tools(query)
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


def _latest_human_query(messages: list) -> str:
    """Tìm tin nhắn user gần nhất làm query cho tool retrieval.

    Giữa 1 vòng agent⇄tools, message cuối có thể là tool result (không phải
    câu hỏi) — nên phải quét ngược tìm đúng message role "human"/"user", không
    chỉ lấy messages[-1].
    """
    for m in reversed(messages):
        if context._role_of(m) in ("human", "user"):
            return context._text_of(m)
    return ""


def should_continue(state: AssistantState) -> str:
    """Conditional edge: có tool call → "tools", không thì dừng graph (Section 1, 3)."""
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return END
    return "tools"


# ── Bài 3: Long-term memory nodes (recall đầu lượt, store cuối lượt) ───────────

def recall_node(state: AssistantState) -> dict:
    """Truy xuất long-term memory liên quan tới tin nhắn mới nhất → chèn làm context.

    Chạy 1 lần ở đầu mỗi lượt (trước agent). Không có user_id → bỏ qua (tương
    thích ngược với Bài 2). Trả về system message chứa "thông tin đã biết về
    user" để agent cá nhân hoá câu trả lời.
    """
    user_id = state.get("user_id")
    if not user_id:
        return {}

    last_user_msg = context._text_of(state["messages"][-1])
    memories = memory.recall_long_term(user_id, last_user_msg)
    if not memories:
        return {}

    block = "Thông tin đã biết về user:\n" + "\n".join(f"- {m}" for m in memories)
    return {"messages": [{"role": "system", "content": block}]}


def extract_and_store_node(state: AssistantState) -> dict:
    """Sau lượt hội thoại, trích 1 sự thật đáng nhớ dài hạn về user → lưu long-term.

    Chạy 1 lần ở cuối lượt (sau khi agent trả lời xong, không còn tool call).
    Không lưu gì vào state — chỉ ghi ra vector store ngoài (memory.py).
    """
    user_id = state.get("user_id")
    if not user_id:
        return {}

    # Lấy vài message gần nhất làm ngữ liệu trích xuất.
    recent = state["messages"][-4:]
    prompt = (
        "Trích xuất 1 sự thật đáng nhớ DÀI HẠN về user từ đoạn hội thoại sau "
        "(sở thích, thông tin cá nhân, dị ứng, quyết định quan trọng). "
        'Chỉ trả về đúng 1 câu sự thật, hoặc "NONE" nếu không có gì đáng nhớ.\n\n'
        + context.format_messages(recent)
    )
    from app.llm import completion
    from app.llm.params import GenerationParams

    fact = completion.chat(
        [{"role": "user", "content": prompt}], GenerationParams(temperature=0.0)
    ).strip()

    if fact and fact.upper() != "NONE":
        memory.save_to_long_term(user_id, fact)
    return {}
