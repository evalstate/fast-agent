from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.agents.mcp_tool_planning import build_mcp_tool_route
from fast_agent.agents.mcp_tool_presentation import (
    attach_read_text_file_display_metadata,
    build_mcp_tool_presentation,
    tool_result_type_label,
)
from fast_agent.mcp.mcp_aggregator import MCPToolCatalog, NamespacedTool
from fast_agent.mcp.tool_result_metadata import tool_result_display_metadata


def _catalog() -> MCPToolCatalog:
    tools = [
        NamespacedTool(
            tool=Tool(name=name, input_schema={"type": "object"}),
            server_name="docs",
            namespaced_tool_name=f"docs__{name}",
        )
        for name in ("read_text_file", "search")
    ]
    return MCPToolCatalog.snapshot(
        by_namespaced_name={tool.namespaced_tool_name: tool for tool in tools},
        by_server={"docs": tools},
        server_names=["docs"],
    )


def test_mcp_tool_route_keeps_local_and_remote_precedence_explicit() -> None:
    catalog = _catalog()

    exact = build_mcp_tool_route(
        requested_name="docs__search",
        catalog=catalog,
        local_tool_exists=True,
        is_filesystem_runtime_tool=False,
    )
    local = build_mcp_tool_route(
        requested_name="search",
        catalog=catalog,
        local_tool_exists=True,
        is_filesystem_runtime_tool=False,
    )
    filesystem_collision = build_mcp_tool_route(
        requested_name="read_text_file",
        catalog=catalog,
        local_tool_exists=False,
        is_filesystem_runtime_tool=True,
    )
    remote_candidate = build_mcp_tool_route(
        requested_name="search",
        catalog=catalog,
        local_tool_exists=False,
        is_filesystem_runtime_tool=False,
    )

    assert exact.execution_name == "docs__search"
    assert exact.active_namespaced_tool is not None
    assert local.active_namespaced_tool is None
    assert local.execution_name == "search"
    assert filesystem_collision.execution_name == "docs__read_text_file"
    assert filesystem_collision.display_name == "docs__read_text_file"
    assert remote_candidate.execution_name == "search"
    assert remote_candidate.display_name == "docs__search"


def test_mcp_tool_presentation_and_file_metadata_preserve_user_visible_contract() -> None:
    catalog = _catalog()
    route = build_mcp_tool_route(
        requested_name="docs__search",
        catalog=catalog,
        local_tool_exists=False,
        is_filesystem_runtime_tool=False,
    )

    presentation = build_mcp_tool_presentation(
        route,
        catalog,
        local_tool_names=None,
        fallback_order=[],
        display_name_overrides={"read_text_file": "read"},
    )

    assert presentation.display_name == "docs__search"
    assert presentation.bottom_items == ["read", "search"]
    assert presentation.highlight_indexes == [1]

    result = CallToolResult(content=[TextContent(type="text", text="contents")])
    attach_read_text_file_display_metadata(
        result,
        display_tool_name="docs__read_text_file",
        tool_args={"path": " notes.txt ", "line": 2, "limit": 0},
    )

    assert tool_result_display_metadata(result) == {
        "read_text_file_path": "notes.txt",
        "read_text_file_line": 2,
    }
    assert tool_result_type_label("docs__read_text_file") == "file read"
