"""MCP Server độc lập — Module II, Bài 4, Section 4 (MCP: Standards, Server, Clients).

Expose lại domain `wellness` (tools.py) qua giao thức MCP chuẩn (JSON-RPC qua
stdio) thay vì chỉ là `@tool` nội bộ. Đây chính là tình huống bài học nói tới:
"MCP đáng giá khi 1 tool/data source cần phục vụ NHIỀU agent/app khác nhau" —
domain wellness ở đây được viết 1 lần, expose qua server này, và bất kỳ MCP
Client nào (agent_m2, 1 agent khác, Claude Desktop, ...) đều gọi lại được mà
không cần import trực tiếp `app.agent_m2.tools`.

Tái dùng LOGIC (không copy code) từ 3 tool `@tool` gốc trong tools.py qua
`.func` — tránh 2 nguồn sự thật cho cùng 1 hành vi (vd rule validate ngày).
`@mcp.tool()` (FastMCP) và `@tool` (langchain_core) là 2 decorator RIÊNG,
không tương thích trực tiếp — đây là lý do cần viết lại wrapper, không thể
"mount" thẳng list TOOLS vào FastMCP.

Chạy độc lập (không qua uvicorn — MCP server dùng stdio, không phải HTTP):
    python -m app.agent_m2.mcp.wellness_server

Test nhanh bằng MCP Inspector (không cần code client):
    npx @modelcontextprotocol/inspector python -m app.agent_m2.mcp.wellness_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.agent_m2.tools import check_sleep_score, find_gym, log_workout

mcp = FastMCP("wellness")


@mcp.tool()
def log_workout_tool(activity: str, duration_minutes: int) -> str:
    """Ghi lại buổi tập thể dục (loại hoạt động + thời lượng phút)."""
    return log_workout.func(activity, duration_minutes)


@mcp.tool()
def check_sleep_score_tool(date: str = "") -> str:
    """Xem điểm chất lượng giấc ngủ của một ngày (mặc định hôm qua nếu để trống)."""
    return check_sleep_score.func(date)


@mcp.tool()
def find_gym_tool(location: str) -> str:
    """Tìm phòng gym gần một khu vực."""
    return find_gym.func(location)


@mcp.resource("wellness://domains")
def list_wellness_domains() -> str:
    """Resource (Bài 4, Section 4): dữ liệu tĩnh Server expose để Client đọc, không phải hành động."""
    return "Hoạt động thể chất, Giấc ngủ, Phòng tập gần đây"


if __name__ == "__main__":
    mcp.run(transport="stdio")
