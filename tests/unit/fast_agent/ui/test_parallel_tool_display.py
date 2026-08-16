from mcp_types import CallToolResult, ImageContent, TextContent

from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.tool_display import ToolCallDisplayRequest, ToolResultDisplayRequest


def _display(
    *,
    layout: str = "compact",
    arguments: str = "auto",
    results: str = "auto",
) -> ConsoleDisplay:
    return ConsoleDisplay(
        config={
            "logger": {
                "tool_display": {
                    "layout": layout,
                    "arguments": arguments,
                    "results": results,
                    "aggregate_parallel": True,
                }
            }
        }
    )


def _call(
    *,
    tool_name: str = "docs__search",
    source_label: str | None = "MCP",
    server_name: str | None = "docs",
    name: str = "dev",
    show_hook_indicator: bool = False,
) -> ToolCallDisplayRequest:
    return ToolCallDisplayRequest(
        tool_name=tool_name,
        tool_args={"query": "needle"},
        name=name,
        tool_call_id="call_abcdef0123456789",
        source_label=source_label,
        server_name=server_name,
        show_hook_indicator=show_hook_indicator,
    )


def _result(
    text: str,
    *,
    result: CallToolResult | None = None,
    source_label: str | None = "MCP",
    server_name: str | None = "docs",
    name: str = "dev",
    tool_name: str = "docs__search",
    show_hook_indicator: bool = False,
) -> ToolResultDisplayRequest:
    return ToolResultDisplayRequest(
        result=result
        or CallToolResult(content=[TextContent(type="text", text=text)], is_error=False),
        name=name,
        tool_name=tool_name,
        timing_ms=2.5,
        tool_call_id="call_abcdef0123456789",
        source_label=source_label,
        server_name=server_name,
        show_hook_indicator=show_hook_indicator,
    )


def test_compact_parallel_generic_calls_show_individual_arguments_and_aggregate_results() -> None:
    display = _display()

    with console.console.capture() as capture:
        display.show_parallel_tool_calls(
            [
                _call(),
                _call(),
                _call(source_label="Agent"),
                _call(server_name="other"),
                _call(name="other"),
                _call(tool_name="docs__lookup"),
            ]
        )
        display.show_parallel_tool_results(
            [
                _result("abc"),
                _result("de"),
                _result("f", source_label="Agent"),
                _result("g", server_name="other"),
                _result("h", name="other"),
                _result("i", tool_name="docs__lookup"),
            ]
        )

    rendered = " ".join(capture.get().split())
    assert "2 requests" not in rendered
    assert rendered.count("needle") == 6
    assert "2 results, 5 chars" in rendered
    assert "2ms" in rendered
    assert "2 results, 5 chars · 2ms · id:" not in rendered


def test_none_argument_policy_suppresses_bodies_and_allows_call_aggregation() -> None:
    display = _display(arguments="none")

    with console.console.capture() as capture:
        display.show_parallel_tool_calls([_call(), _call()])

    rendered = " ".join(capture.get().split())
    assert "2 requests" in rendered
    assert "needle" not in rendered


def test_compact_parallel_aggregation_preserves_identity_and_exclusions() -> None:
    display = _display()
    error = CallToolResult(
        content=[TextContent(type="text", text="failure details")],
        is_error=True,
    )
    structured = CallToolResult(
        content=[TextContent(type="text", text="structured")],
        structured_content={"ok": True},
        is_error=False,
    )
    media = CallToolResult(
        content=[ImageContent(type="image", data="AAAA", mime_type="image/png")],
        is_error=False,
    )

    with console.console.capture() as capture:
        for tool_name in ("execute", "read_text_file", "apply_patch", "edit_file"):
            display.show_parallel_tool_calls(
                [_call(tool_name=tool_name), _call(tool_name=tool_name)]
            )
        display.show_parallel_tool_calls(
            [_call(show_hook_indicator=True), _call(show_hook_indicator=True)]
        )
        display.show_parallel_tool_results([_result("", result=error), _result("", result=error)])
        display.show_parallel_tool_results(
            [_result("", result=structured), _result("", result=structured)]
        )
        display.show_parallel_tool_results([_result("", result=media), _result("", result=media)])

    rendered = " ".join(capture.get().split())
    assert "2 requests" not in rendered
    assert "2 results" not in rendered
    assert rendered.count("ERROR") == 2
    assert rendered.count("failure details") == 2
    assert rendered.count("◆") == 2
    assert rendered.count("Structured ■") == 2


def test_all_argument_or_result_policies_disable_parallel_aggregation() -> None:
    display = _display(arguments="all", results="all")

    with console.console.capture() as capture:
        display.show_parallel_tool_calls([_call(), _call()])
        display.show_parallel_tool_results([_result("first"), _result("second")])

    rendered = " ".join(capture.get().split())
    assert "2 requests" not in rendered
    assert "2 results" not in rendered
    assert rendered.count("needle") == 2
    assert "first" in rendered
    assert "second" in rendered


def test_full_parallel_batches_keep_individual_tool_cards() -> None:
    display = _display(layout="full")

    with console.console.capture() as capture:
        display.show_parallel_tool_calls([_call(), _call()])
        display.show_parallel_tool_results([_result("first"), _result("second")])

    rendered = " ".join(capture.get().split())
    assert "2 requests" not in rendered
    assert "2 results" not in rendered
    assert rendered.count("needle") == 2
    assert "first" in rendered
    assert "second" in rendered
