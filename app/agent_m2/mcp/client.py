"""MCP Client — Module II, Bài 4, Section 4 (kết nối Client từ agent LangGraph).

Kết nối `MultiServerMCPClient` (langchain_mcp_adapters) tới wellness_server.py
(cùng thư mục) qua stdio transport — client tự spawn server làm subprocess con,
giao tiếp qua JSON-RPC trên stdin/stdout (không cần server chạy sẵn/network).

Tool trả về từ `client.get_tools()` là `BaseTool` (LangChain) y hệt `@tool` viết
tay trong tools.py — bind_tools()/ToolNode dùng được ngay, không cần adapter
thêm. Đây là điểm giá trị của MCP: agent KHÔNG phân biệt được tool nội bộ hay
tool qua MCP server khác, miễn description đủ tốt cho model chọn đúng.

`get_tools()` là ASYNC (giao thức MCP dùng JSON-RPC qua stdio, không có bản
sync) — đây là lý do toàn bộ graph.py/routes_assistant.py chuyển sang async
`ainvoke`/`async def` ở Bài 4 (khác Bài 2-3 vốn sync hoàn toàn).
"""

from __future__ import annotations

import sys
from pathlib import Path

# app/agent_m2/mcp/client.py -> lên 4 cấp = project root (chứa pyproject.toml).
# Cần cwd đúng vì server import `app.agent_m2.tools` — chạy qua `python -m` với
# cwd=project root để package `app` nằm trên sys.path của subprocess con, giống
# hệt cách uvicorn/pytest tự chạy từ project root (không phải đường dẫn file trực tiếp).
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.parent)

_cached_mcp_tools: list | None = None


async def get_mcp_tools() -> list:
    """Kết nối tới wellness_server.py, trả về list tool MCP (cache sau lần đầu).

    Cache ở module-level (không phải lru_cache) vì đây là coroutine — lru_cache
    không cache được kết quả của hàm async đúng cách (sẽ cache lại chính
    coroutine object, không phải giá trị đã await).
    """
    global _cached_mcp_tools
    if _cached_mcp_tools is not None:
        return _cached_mcp_tools

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "wellness": {
                "command": sys.executable,
                "args": ["-m", "app.agent_m2.mcp.wellness_server"],
                "cwd": _PROJECT_ROOT,
                "transport": "stdio",
            },
        }
    )
    _cached_mcp_tools = await client.get_tools()
    return _cached_mcp_tools


def reset_cache() -> None:
    """Dùng trong test — xoá cache để mock lại get_mcp_tools() giữa các test case."""
    global _cached_mcp_tools
    _cached_mcp_tools = None
