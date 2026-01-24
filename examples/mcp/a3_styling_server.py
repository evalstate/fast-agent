from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel

EXTRA_TOOL_NAME = "a3_extra_tool"
_EXTRA_TOOL_ADDED = False

app = FastMCP(name="A3 Styling Demo")



def _extra_tool(note: str = "Tool list update confirmed") -> str:
    return f"Extra tool active: {note}"


class StructuredReport(BaseModel):
    title: str
    items: list[str]
    metadata: dict[str, Any]


@app.tool(
    name="a3_structured_report",
    description="Return structured and unstructured content for A3 tool result styling.",
    structured_output=True,
)
def a3_structured_report(topic: str, count: int = 3) -> StructuredReport:
    items = [f"{topic} item {index + 1}" for index in range(count)]
    return StructuredReport(
        title=f"Structured report for {topic}",
        items=items,
        metadata={
            "topic": topic,
            "count": count,
            "note": "This payload should show structured content in A3 tool results.",
        },
    )


@app.tool(
    name="a3_unstructured_echo",
    description="Return a plain-text response to show non-structured tool styling.",
)
def a3_unstructured_echo(message: str) -> str:
    return f"Echo: {message}"




@app.tool(
    name="a3_trigger_tool_update",
    description="Add a new tool and emit a tool list changed notification.",
)
async def a3_trigger_tool_update(context: Context) -> str:
    global _EXTRA_TOOL_ADDED

    if not _EXTRA_TOOL_ADDED:
        context.fastmcp.add_tool(
            _extra_tool,
            name=EXTRA_TOOL_NAME,
            description="Extra tool registered after tool list update.",
        )
        _EXTRA_TOOL_ADDED = True

    await context.request_context.session.send_tool_list_changed()
    return "Tool list change notification sent. Refresh tools to see updates."


@app.prompt("a3_daily_brief")
def a3_daily_brief(project: str, focus: str = "status") -> str:
    """Prompt with arguments to exercise /prompt selection and argument collection."""
    return (
        "You are an A3-style briefing assistant. "
        f"Provide a {focus} update for project '{project}' with three bullets."
    )


@app.prompt("a3_review")
def a3_review(area: str = "UI") -> str:
    """Simple prompt for /prompt listing and selection UI."""
    return (
        "You are reviewing the A3 display style for the following area: "
        f"{area}. Provide a short critique and a next-step checklist."
    )


if __name__ == "__main__":
    app.run()
