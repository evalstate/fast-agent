"""MCP server for exercising structured tool result previews.

Run with:
    uv run preview_server.py
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

app = FastMCP(name="Structured Content Preview Demo")


def _text_block(payload: Any) -> TextContent:
    return TextContent(
        type="text",
        text=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )


def _tool_result(*, text_payloads: list[Any], structured_payload: dict[str, Any]) -> CallToolResult:
    result = CallToolResult(
        content=[_text_block(payload) for payload in text_payloads],
        isError=False,
    )
    setattr(result, "structuredContent", structured_payload)
    return result


@app.tool(
    name="structured_content_match",
    description=(
        "Return multiple text blocks that match the structuredContent payload. "
        "Useful for checking the new preview path."
    ),
)
def structured_content_match() -> CallToolResult:
    tickets = [
        {"ticket_id": "T-100", "status": "open", "owner": "alex"},
        {"ticket_id": "T-101", "status": "pending", "owner": "sam"},
    ]
    return _tool_result(
        text_payloads=tickets,
        structured_payload={"tickets": tickets, "match_state": "match"},
    )


@app.tool(
    name="structured_content_mismatch",
    description=(
        "Return multiple text blocks that do not match structuredContent. "
        "Useful for seeing how the preview behaves when the two disagree."
    ),
)
def structured_content_mismatch() -> CallToolResult:
    text_tickets = [
        {"ticket_id": "T-100", "status": "closed", "owner": "alex"},
        {"ticket_id": "T-101", "status": "pending", "owner": "sam"},
    ]
    structured_tickets = [
        {"ticket_id": "T-100", "status": "open", "owner": "alex"},
        {"ticket_id": "T-101", "status": "escalated", "owner": "sam"},
    ]
    return _tool_result(
        text_payloads=text_tickets,
        structured_payload={"tickets": structured_tickets, "match_state": "mismatch"},
    )


if __name__ == "__main__":
    app.run(transport="stdio")
