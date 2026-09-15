# Module II — AI Agent

Module II chuyển từ "LLM trả lời câu hỏi" sang "LLM tự hành động". Track riêng khỏi Module I (RAG pháp lý) — dùng **LangGraph** làm framework chính xuyên suốt, code nằm ở `app/agent_m2/` (tách biệt khỏi `app/agent/` của Buổi 6 Module I, vốn là CRAG chuyên cho RAG).

> Cài đặt / chạy / test chung cho cả Module I và II: xem [README.md](README.md).
> Module I (RAG pháp lý): xem [README.module1.md](README.module1.md).

---

## Lộ trình xây dựng

| Buổi | Chủ đề | Lắp vào codebase |
|------|--------|------------------|
| 1 | What Is an AI Agent? | Khái niệm — không có code |
| **2** | **Building Agents với LangGraph** | `agent_m2/` — **ReAct loop + HITL chạy được** |
| **3** | **Memory & Context Engineering** | `agent_m2/memory.py`, `context.py` — **long-term memory + compaction** |
| **4** | **Agentic Tool Design & Integration** | `agent_m2/tools.py`, `tool_selection.py`, `mcp/` — **15 tool, error handling, idempotency, tool retrieval, MCP server/client** |
| **5** | **Agent Evaluation & Observability** | `agent_m2/eval.py` — **LangSmith tracing + Task Success/Trajectory eval, nút Đánh giá trên UI** |
| **6** | **Multi-Agent Systems** | `agent_m2/multi_agent/` — **4 pattern chạy được: Sequential, Hierarchical, Collaborative, Swarm** |

### Buổi 2 — Building Agents với LangGraph

Personal Assistant tiếng Việt minh hoạ trọn vẹn nội dung bài học: state/nodes/edges, tool execution loop (`ToolNode`), loop termination + phát hiện lặp vô hạn, memory qua nhiều lượt (`MemorySaver` + `thread_id`), và human-in-the-loop trước khi chạy tool.

```
user message → agent (LLM + bind_tools) → có tool call?
                  ├─ Không → END, trả lời trực tiếp
                  └─ Có → [DỪNG — chờ người duyệt] → tools (ToolNode) → agent → ...
```

> **Vì sao dùng `langchain_openai.ChatOpenAI` ở đây thay vì native SDK như mọi
> nơi khác trong repo?** `ToolNode`/`bind_tools` (LangGraph prebuilt) chỉ tự
> thực thi tool khi model được gọi qua interface LangChain — đây là ngoại lệ
> có chủ đích, xem docstring [`app/agent_m2/nodes.py`](app/agent_m2/nodes.py).
> `app/agent/` (Module I, Buổi 6, CRAG) vẫn 100% native vì tự viết tool loop tay.

```bash
# Lượt 1 — agent muốn gửi lời nhắc (tool nhạy cảm) → graph dừng chờ duyệt
curl -X POST http://localhost:8000/assistant/message \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "demo-1", "message": "Nhắc tôi họp lúc 15h chiều nay"}'
# → {"status": "pending_approval", "tool_call": {"name": "send_reminder", ...}}

# Duyệt (approve=true) → graph chạy tiếp: tool thực thi thật, agent trả lời
curl -X POST http://localhost:8000/assistant/approve \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "demo-1", "approve": true}'

# Từ chối (approve=false) → tool KHÔNG chạy, agent nhận lý do từ chối làm quan sát
curl -X POST http://localhost:8000/assistant/approve \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "demo-1", "approve": false, "rejection_note": "chưa cần"}'
```

`thread_id` giữ nguyên qua nhiều lượt `/assistant/message` để agent nhớ hội thoại (checkpointer `MemorySaver`, chỉ lưu RAM — production dùng `SqliteSaver`). Tool chỉ đọc (`check_calendar`, `search_restaurant`, ...) chạy ngay không cần duyệt; tool có side-effect (`send_reminder`, ...) luôn dừng ở `/assistant/message` chờ `/assistant/approve` vì graph compile với `interrupt_before=["tools"]` (áp dụng cho mọi tool call). Từ Bài 4, danh sách tool cụ thể agent "thấy" mỗi lượt không còn cố định — xem mục Bài 4 bên dưới.

### Buổi 3 — Memory & Context Engineering

Bổ sung 2 tầng bộ nhớ + quản lý context cho chính agent ở Buổi 2 (không tạo agent mới). File mới: [`memory.py`](app/agent_m2/memory.py) (long-term vector memory) và [`context.py`](app/agent_m2/context.py) (sliding window, summarization, nén tool output, nguyên tắc 40-60%, re-injection). Graph thêm 3 node quanh vòng lặp:

```
message → recall → [vượt ngưỡng 40%?] ─(có)→ compact ─┐
              ↑                       └─(không)────────┴→ agent ⇄ tools → store → END
        vector store ngoài                                                    ↓
                                                                       vector store ngoài
```

| Loại memory | Sống ở đâu | Cơ chế |
|---|---|---|
| **Short-term** (session) | Graph state + checkpointer (`thread_id`) | đã có từ Buổi 2 |
| **Long-term** (xuyên session) | Vector store ngoài (`user_id`) | `memory.py` — recall/store mỗi lượt |

Long-term memory **tái dùng embeddings native của Module I** (`app/retrieval/embeddings.py`, OpenAI text-embedding-3-small) + collection Qdrant riêng `user_memory` — **không** kéo thêm `langchain_community`/`chromadb` như handbook, giữ triết lý native SDK. Khi chưa có `OPENAI_API_KEYS`, `memory.py` tự fallback sang store in-memory (keyword match) để demo/test vẫn chạy.

```bash
# Session A — user "minh" khai một sự thật đáng nhớ (long-term)
curl -X POST http://localhost:8000/assistant/message \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "sess-A", "user_id": "minh", "message": "Tôi bị dị ứng hải sản, nhớ giúp tôi."}'

# Session B — thread_id MỚI (short-term đã mất) nhưng cùng user_id → agent VẪN nhớ
curl -X POST http://localhost:8000/assistant/message \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "sess-B", "user_id": "minh", "message": "Gợi ý món ăn tối nay cho tôi."}'
# → agent tránh gợi ý hải sản nhờ recall long-term memory (không phụ thuộc thread_id)
```

**Context engineering (chống context rot)** chia làm 2 loại thao tác, đặt ở 2 nơi
khác nhau theo đúng kỷ luật "one node one function" của LangGraph:

| Thao tác | Ở đâu | Vì sao |
|---|---|---|
| **Compaction** (summarize) | Node riêng `compact_node`, có conditional edge sau `recall` | Đây là bước DUY NHẤT **ghi đè state** (dùng `RemoveMessage` + summary qua reducer `add_messages`) — bản nén **persist**, các vòng `agent⇄tools` sau đó trong CÙNG lượt kế thừa, không nén lại. Nếu để trong `agent_node`, mỗi vòng lặp sẽ nén lại từ đầu mà không tái dùng — tốn LLM call thừa. | | Sliding window, repetition warning, re-injection | Trong `agent_node` | Chỉ dựng một `messages` list **tạm thời** gửi cho LLM lần gọi này — không ghi lại state, nên gộp chung 1 node là hợp lý (không mutate gì để tách). |

`should_compact_route` kiểm tra ngưỡng **40%** window (`AGENT_CONTEXT_WINDOW_TOKENS`, đặt nhỏ để demo dễ thấy) ngay sau `recall`. `agent_node` re-inject chỉ dẫn cốt lõi ở *cuối* context (vị trí attention cao nhất) chống instruction fade-out, và áp sliding window (`AGENT_MAX_MESSAGES`) cho phần đã compact. `context.compress_tool_result()` nén output tool dài — viết sẵn nhưng **chưa nối vào graph** vì `ToolNode` prebuilt không có hook; xem ghi chú trong docstring hàm đó.

### Buổi 4 — Agentic Tool Design & Integration

Mở rộng bộ tool từ 3 (Buổi 2) thành **15 tool + 1 composite**, chia **5 domain** ("AI Personal Assistant" thật): `calendar`, `dining`, `productivity`, `shopping`, `wellness` — đủ để minh hoạ **tool sprawl** đúng ngưỡng bài học nói (>20 tool độ chính xác chọn tool giảm mạnh). File mới: [`tools.py`](app/agent_m2/tools.py) (15 tool + registry) và [`tool_selection.py`](app/agent_m2/tool_selection.py) (hierarchical grouping + tool retrieval).

**Nguyên tắc thiết kế tool (Section 1)** áp dụng trực tiếp lên tool có sẵn:

| Nguyên tắc | Ở đâu |
|---|---|
| Error handling (không raise) | Mọi tool — validate ngày/tham số, trả `"Lỗi: ..."` hoặc `{"error": ...}` |
| Idempotency | Tool có side-effect (`send_reminder`, `book_table`, `create_event`, `add_to_cart`, ...) — `idempotency_key` tự sinh từ tham số, gọi lại không nhân đôi |
| Composite tool | `book_dinner_plan` gộp `check_calendar` + `book_table` — minh hoạ trade-off atomic/composite ngay trong domain đã có |

**Tool Selection (Section 2)** — agent **không còn bind cố định** toàn bộ 15 tool mỗi lượt. `agent_node` (nodes.py) gọi `tool_selection.retrieve_relevant_tools(query, k=AGENT_TOOL_RETRIEVAL_K)`: embed câu hỏi user, so cosine similarity với embedding của **description** 15 tool (embed 1 lần, cache — tái dùng `app/retrieval/embeddings.py`, không thêm Chroma/LangChain vectorstore như handbook mẫu), chỉ bind top-k tool liên quan nhất. `ToolNode` (graph.py) vẫn giữ **toàn bộ** `TOOLS` để có thể *thực thi* bất kỳ tool nào — chỉ phần LLM *nhìn thấy* mới bị giới hạn.

```bash
# Agent chỉ bind top-5 tool liên quan tới "đặt bàn ăn tối" — không thấy log_workout,
# draft_email, v.v. dù chúng vẫn tồn tại trong hệ thống.
curl -X POST http://localhost:8000/assistant/message \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "demo-tools-1", "message": "Đặt bàn cho 4 người ở Quán A lúc 19h tối nay"}'
# → agent gọi book_table (hoặc book_dinner_plan) → pending_approval (side-effect)

curl -X POST http://localhost:8000/assistant/approve \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "demo-tools-1", "approve": true}'
```

`AGENT_TOOL_RETRIEVAL_K` (mặc định 5) kiểm soát số tool model thấy mỗi lượt — tăng lên nếu câu hỏi cần nhiều domain cùng lúc, giảm xuống để ép độ chính xác chọn tool cao hơn (đánh đổi: dễ bỏ sót tool đúng nếu quá thấp).

> `classify_intent`/`get_tools_by_group` (Hierarchical Grouping) vẫn có trong
> `tool_selection.py` như phương án thay thế — rẻ hơn (không cần embedding) nhưng
> cứng nhắc hơn (chỉ trả về đúng 1 domain, bỏ sót câu hỏi cần tool từ nhiều
> domain cùng lúc). `agent_node` dùng retrieval, không dùng grouping.

**MCP: Standards, Server, Clients (Section 4)** — domain `wellness` giờ chạy qua **MCP server độc lập** thay vì `@tool` nội bộ, minh hoạ đúng tình huống bài học nói tới: *"MCP đáng giá khi 1 tool cần phục vụ nhiều agent/app khác nhau"*.

- [`mcp/wellness_server.py`](app/agent_m2/mcp/wellness_server.py) — `FastMCP` server, expose `log_workout_tool`/`check_sleep_score_tool`/`find_gym_tool` (tái dùng logic từ `tools.py` qua `.func`, không copy code) + 1 `@mcp.resource`. Chạy độc lập qua stdio: `python -m app.agent_m2.mcp.wellness_server`.
- [`mcp/client.py`](app/agent_m2/mcp/client.py) — `MultiServerMCPClient` (langchain-mcp-adapters) tự spawn server làm subprocess con, giao tiếp qua JSON-RPC trên stdin/stdout. Tool trả về là `BaseTool` chuẩn — `tool_selection.py` gộp thẳng vào cùng index với 16 tool nội bộ; **model không phân biệt được** tool nào tới từ MCP hay tool nội bộ, miễn description đủ tốt.

```bash
# Kiểm tra nhanh (không cần chạy app) — thấy được 3 tool MCP + 16 tool nội bộ
# đều nằm trong 1 danh sách, sẵn sàng cho tool_selection.retrieve_relevant_tools:
python -c "
import asyncio
from app.agent_m2.tool_selection import _all_tools
print(asyncio.run(_all_tools()))
"
```

#### Demo trực quan MCP Server/Client bằng MCP Inspector

Cách dễ nhất để **thấy** giao thức MCP hoạt động (request/response JSON-RPC thật, không phải chỉ đọc code) — không cần viết thêm dòng nào, có UI web:

```bash
npx @modelcontextprotocol/inspector python -m app.agent_m2.mcp.wellness_server
```

Lệnh này **tự chạy cả server lẫn client** trong 1 bước — Inspector đóng vai MCP Client, tự spawn `wellness_server.py` làm subprocess con (y hệt cách `mcp/client.py` làm ở trên), không cần mở terminal riêng cho server. Terminal in ra link kèm token xác thực, mở link đó trong trình duyệt:

1. Bấm **Connect** (Inspector tự nối tới server qua stdio).
2. Tab **Tools** → thấy đúng 3 tool `log_workout_tool`, `check_sleep_score_tool`, `find_gym_tool` với schema tham số y hệt docstring trong `wellness_server.py`.
3. Chọn `find_gym_tool`, điền `location: "Quận 1"` → **Run Tool** → thấy raw JSON-RPC request gửi đi và response trả về ngay trong UI.
4. Tab **Resources** → thấy `wellness://domains` — minh hoạ khác biệt **Tool** (hành động) vs **Resource** (dữ liệu tĩnh) mà bài học nói tới.

> Lần đầu chạy `npx` sẽ tải package (~vài giây, cần mạng) — nên chạy thử trước
> giờ học để cache sẵn, tránh chờ trước lớp.

> **Vì sao toàn bộ graph chuyển sang `async`/`ainvoke` ở Bài 4?** MCP dùng
> giao thức JSON-RPC qua stdio — client chỉ có bản `get_tools()` async, không
> có bản sync. `agent_node` (nodes.py) cần `await` MCP client khi load tool lần
> đầu → kéo theo `start_conversation`/`resume_conversation` (graph.py) và 2
> route trong `routes_assistant.py` đều thành `async def`. Khác Bài 2-3 vốn
> sync hoàn toàn (`llm.invoke()`, `app.invoke()`).

### Buổi 5 — Agent Evaluation & Observability

RAGAS/LLM-as-judge (Module I, Bài 7) chấm **output cuối** — đủ cho RAG nhưng
không đủ cho agent, vì agent có thể ra đúng đáp án bằng con đường sai (lặp
tool thừa, chọn sai tham số, tốn kém). Buổi này thêm 2 lớp: **Observability**
(giám sát online, luôn bật) và **Evaluation** (chấm chất lượng offline/on-demand).

**Observability — LangSmith tracing (Section 3).** [`graph.py`](app/agent_m2/graph.py)
bọc `start_conversation`/`resume_conversation` qua `trace_answer`
([`monitoring/tracing.py`](app/monitoring/tracing.py) — **tái dùng nguyên xi**
từ Module I, không viết lại LangSmith integration). Mỗi lượt chat = 1 run, gắn
input/output/latency. Khác `app/agent/nodes.py` (CRAG, Module I) vốn tạo
NESTED run cho từng node graph, ở đây trace 1 run PHẲNG cho cả lượt — đơn
giản hơn, không phải xuyên `_trace_span` qua 5 node đã ổn định từ Bài 2-4.

```bash
# .env: điền LANGSMITH_API_KEY (như Module I, Bài 7)
MONITORING_ENABLED=true
```

> Mặc định `MONITORING_ENABLED=false` → `trace_answer` no-op hoàn toàn, không
> bắt buộc cài/kích hoạt LangSmith để chạy phần còn lại của agent.

**Evaluation — Task Success + Trajectory Quality (Section 1-2).**
[`eval.py`](app/agent_m2/eval.py) chấm 2 chiều bằng LLM-as-judge (native
`chat_parsed`, rubric tuyệt đối — tái áp dụng nguyên tắc giảm bias đã học ở
[`app/eval/judge.py`](app/eval/judge.py), Module I Bài 7):

| Chiều | Hàm | Chấm gì |
|---|---|---|
| **Task Success** | `evaluate_task_success` | Agent có đạt đúng yêu cầu user không (không xét quá trình) |
| **Trajectory Quality** | `evaluate_trajectory` | Cả CHUỖI hành động: efficiency, thứ tự logic, tool/tham số đúng không, xử lý lỗi (recovery) |

`graph.py:_extract_trajectory()` dựng lại chuỗi `{tool, args, observation}` từ
checkpointer state (khớp `tool_call_id` giữa `AIMessage.tool_calls` và
`ToolMessage`) — không cần lưu trajectory riêng, state đã có đủ.

```bash
# Endpoint riêng: POST /assistant/evaluate — chỉ cần thread_id, server tự đọc
# lại checkpointer để lấy task + trajectory + câu trả lời cuối.
curl -X POST http://localhost:8000/assistant/message \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "demo-eval", "message": "Tìm nhà hàng gần Quận 1"}'

curl -X POST http://localhost:8000/assistant/evaluate \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "demo-eval"}'
# → {"task_success": {...}, "trajectory": {...}, "trajectory_steps": [...]}
```

> **Vì sao `/evaluate` không nhận trajectory từ client?** Đúng triết lý "state
> sống trong checkpointer" (Bài 2, Section 5) — frontend chỉ cần gửi
> `thread_id`, không phải tự quản lý lịch sử/trajectory song song. Trả lỗi rõ
> ràng nếu thread rỗng hoặc agent còn đang chờ duyệt tool (chưa có final
> answer để chấm).

**Demo UI — nút "📊 Đánh giá"** ([`app/static/chat.html`](app/static/chat.html)):
sau mỗi câu trả lời "done" ở mode Assistant, bubble hiện nút Đánh giá. Bấm vào
gọi `/assistant/evaluate`, hiện ngay badge Task Success (✓/✗ + điểm %), 4 điểm
Trajectory Quality (efficiency/thứ tự/tool/recovery, thang 1-5), danh sách
issues nếu có, và `<details>` xem lại từng bước tool call + observation.

### Buổi 6 — Multi-Agent Systems

Một agent với quá nhiều tool/trách nhiệm trở nên khó kiểm soát (đã thấy rõ ở
Bài 4: 15 tool đã cần tool retrieval). Buổi này chia nhỏ công việc cho nhiều
agent chuyên biệt — **4 pattern** (Section 1), cả 4 đều code chạy được, trên
CÙNG domain trợ lý cá nhân (tái dùng `tools.py`/`TOOL_GROUPS` từ Bài 4) để dễ
so sánh trực tiếp. Tách hẳn khỏi `agent_m2/graph.py` (1 agent đơn, Bài 2-5) —
đây là 4 kiến trúc multi-agent độc lập, không phải nâng cấp thêm.

| Pattern | File | Kiến trúc demo | Cơ chế LangGraph |
|---|---|---|---|
| **Sequential** | [`sequential.py`](app/agent_m2/multi_agent/sequential.py) | Planner → Scheduler → Notifier, thứ tự CỐ ĐỊNH | `add_edge` tuyến tính |
| **Hierarchical** | [`hierarchical.py`](app/agent_m2/multi_agent/hierarchical.py) | Supervisor điều phối 3 Worker theo domain (calendar/dining/wellness) | `add_conditional_edges` từ Supervisor + mỗi Worker là 1 **subgraph** state riêng |
| **Collaborative** | [`collaborative.py`](app/agent_m2/multi_agent/collaborative.py) | Planner ⇄ Critic phản biện lặp vòng tới khi duyệt | vòng lặp có điều kiện + `MAX_ROUNDS` cứng |
| **Swarm** | [`swarm.py`](app/agent_m2/multi_agent/swarm.py) | 3 agent domain NGANG HÀNG, tự handoff | `Command(goto=..., update=...)` — agent con tự quyết định đích, không qua điều phối viên |

```bash
# Sequential — 3 bước cố định, trả về đủ cả 3 sản phẩm trung gian
curl -X POST http://localhost:8000/multi-agent/sequential \
  -H "Content-Type: application/json" \
  -d '{"request": "Đặt bàn ăn tối thứ 6 này cho 4 người ở Quận 1"}'

# Hierarchical — Supervisor tự chọn worker, có thể gọi nhiều worker liên tiếp
curl -X POST http://localhost:8000/multi-agent/hierarchical \
  -H "Content-Type: application/json" \
  -d '{"request": "Tôi muốn ăn tối ở Quận 1 và sau đó đi tập gym gần đó"}'

# Collaborative — trả kèm rounds/approved để thấy QUÁ TRÌNH phản biện
curl -X POST http://localhost:8000/multi-agent/collaborative \
  -H "Content-Type: application/json" \
  -d '{"request": "Lên kế hoạch một buổi hẹn hò cuối tuần này ở Hà Nội"}'

# Swarm — entry_agent mô phỏng "kênh liên hệ" user chọn; câu hỏi lệch domain
# sẽ tự handoff (xem handoff_log trong response)
curl -X POST http://localhost:8000/multi-agent/swarm \
  -H "Content-Type: application/json" \
  -d '{"message": "Tìm nhà hàng gần Quận 1", "entry_agent": "calendar"}'
# → {"answer": "...", "final_agent": "dining", "handoff_log": ["calendar → dining"]}
```

> **Mỗi "sub-agent" có phải 1 agent thật hay chỉ là 1 node function?** Mặc
> định trong LangGraph, một node CHỈ là hàm `(state) -> dict` — nếu mọi
> "agent" dùng chung 1 `StateGraph`, chúng chỉ khác nhau ở tên, không có ranh
> giới cô lập dữ liệu nào (Sequential/Collaborative/Swarm ở đây đều vậy: mọi
> node đọc/ghi 1 state phẳng chung). `hierarchical.py` minh hoạ cách làm ranh
> giới đó THẬT: mỗi Worker là 1 **subgraph độc lập** với `WorkerState` riêng
> (`messages` nội bộ, tool-call bookkeeping) — khi nhúng subgraph đã compile
> làm 1 node của graph cha, LangGraph chỉ truyền qua các field TRÙNG TÊN giữa
> 2 schema (`task` vào, `result` ra); mọi field nội bộ khác không hề lộ ra
> Supervisor. Đây là lý do một bug thật đã xảy ra ở bản đầu tiên (node phẳng):
> worker B đọc nhầm `ToolMessage` của worker A chạy trước nó trong cùng list
> `messages` dùng chung — subgraph loại bỏ khả năng này bằng cơ chế, không
> phải bằng kỷ luật lọc dữ liệu cẩn thận.

> **Vì sao Swarm không dùng `Command` làm input trực tiếp cho `ainvoke()`?**
> Thử nghiệm cho thấy LangGraph vẫn chạy qua entry point cố định của
> `set_entry_point()` trước khi áp `Command`, gây ghi đè state 2 lần trong
> cùng 1 step (`InvalidUpdateError`). Route "agent nào nhận tin nhắn đầu tiên"
> phải nằm TRONG graph — dùng `add_conditional_edges(START, ...)` — không thể
> chọn qua input, xem chi tiết trong docstring `swarm.py`.

> **Luôn set `recursion_limit`** khi invoke bất kỳ graph multi-agent nào
> (Section 3, warning bài học) — vòng Supervisor↔Worker hoặc Planner↔Critic
> có thể chạy vô hạn nếu không có `MAX_ROUNDS`/điều kiện dừng rõ ràng. Cả 4
> `run_*()` ở đây đều set `config={"recursion_limit": ...}`.

**Framework Comparison (Section 2)** — bài học so sánh LangGraph/CrewAI/OpenAI
Agents SDK/AutoGen; repo này **chỉ code bằng LangGraph** (đã học Bài 2-5, tái
dùng nguyên state/checkpointer/HITL) — lý do bài học nêu: *"tránh phải học lại
1 mental model mới hoàn toàn như khi chuyển sang CrewAI/AutoGen"*. Xem tài
liệu bài học (handbook Bài 6, Section 2) nếu muốn đối chiếu cú pháp
CrewAI/OpenAI Agents SDK.

---

## Cấu trúc (Module II)

```
app/
├── agent_m2/          # ✓ Module II — Personal Assistant agent (LangGraph)
│                      #   Buổi 2: graph/nodes/state (ReAct loop, ToolNode, HITL)
│                      #   Buổi 3: memory.py (long-term vector memory), context.py (compaction)
│                      #   Buổi 4: tools.py (15 tool), tool_selection.py (grouping + retrieval)
│                      #     + mcp/ (wellness_server.py, client.py — MCP Server/Client thật)
│                      #   Buổi 5: eval.py (Task Success + Trajectory eval, LLM-as-judge)
│                      #   Buổi 6: multi_agent/ (Sequential, Hierarchical, Collaborative, Swarm)
├── monitoring/        # tracing.py — LangSmith hooks (Module I, tái dùng cho agent_m2 Bài 5)
└── api/
    ├── routes_assistant.py     # /assistant/message, /assistant/approve, /assistant/evaluate
    └── routes_multi_agent.py   # /multi-agent/{sequential,hierarchical,collaborative,swarm}
```

> Cài đặt / chạy app / test / bảo mật: xem [README.md](README.md).
