from types import SimpleNamespace

from mcp.types import CallToolResult, ImageContent, TextContent

from fast_agent.config import Settings, ShellSettings
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.shell_output_truncation import SHELL_OUTPUT_TRUNCATION_MARKER
from fast_agent.ui.tool_display import ToolDisplay


def test_shell_tool_result_uses_styled_exit_line() -> None:
    display = ConsoleDisplay()
    result = CallToolResult(
        content=[TextContent(type="text", text="hello\nprocess exit code was 0")],
        isError=False,
    )
    setattr(result, "output_line_count", 1)

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="execute",
            tool_call_id="call_abcdef0123456789",
        )

    rendered = capture.get()
    assert "hello" in rendered
    assert "exit code 0" in rendered
    assert "1 line" in rendered
    assert "id: call_" in rendered
    assert "process exit code was 0" not in rendered


def test_shell_tool_result_no_output_adds_no_output_detail() -> None:
    display = ConsoleDisplay()
    result = CallToolResult(
        content=[TextContent(type="text", text="process exit code was 0")],
        isError=False,
    )

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="execute",
            tool_call_id="call_abcdef0123456789",
        )

    rendered = capture.get()
    assert "exit code 0" in rendered
    assert "(no output)" in rendered
    assert "0 lines" not in rendered
    assert "process exit code was 0" not in rendered
    assert "(empty text)" not in rendered


def test_shell_tool_result_truncates_with_head_and_tail_windows() -> None:
    display = ConsoleDisplay(config=Settings(shell_execution=ShellSettings(output_display_lines=6)))
    output_lines = [f"out-{i:02d}" for i in range(1, 11)]
    result_text = "\n".join([*output_lines, "process exit code was 0"])
    result = CallToolResult(content=[TextContent(type="text", text=result_text)], isError=False)
    setattr(result, "output_line_count", len(output_lines))

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="execute",
            tool_call_id="call_abcdef0123456789",
        )

    rendered = capture.get()
    assert "out-01" in rendered
    assert "out-02" in rendered
    assert "out-03" in rendered
    assert "out-08" in rendered
    assert "out-09" in rendered
    assert "out-10" in rendered
    assert "out-04" not in rendered
    assert "out-05" not in rendered
    assert "out-06" not in rendered
    assert "out-07" not in rendered
    assert SHELL_OUTPUT_TRUNCATION_MARKER in rendered
    assert "10 lines" in rendered


def test_shell_tool_result_parallel_deferred_uses_source_line_count() -> None:
    display = ConsoleDisplay(config=Settings(shell_execution=ShellSettings(output_display_lines=4)))
    output_lines = [f"line-{i}" for i in range(1, 13)]
    result_text = "\n".join([*output_lines, "process exit code was 0"])
    result = CallToolResult(content=[TextContent(type="text", text=result_text)], isError=False)

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="execute",
            tool_call_id="call_abcdef0123456789",
        )

    rendered = capture.get()
    assert "line-1" in rendered
    assert "line-2" in rendered
    assert "line-11" in rendered
    assert "line-12" in rendered
    assert "line-3" not in rendered
    assert "line-10" not in rendered
    assert SHELL_OUTPUT_TRUNCATION_MARKER in rendered
    assert "12 lines" in rendered


def test_shell_tool_result_tolerates_lightweight_config_without_shell_settings() -> None:
    display = ConsoleDisplay(config=SimpleNamespace())
    result = CallToolResult(
        content=[TextContent(type="text", text="hello\nprocess exit code was 0")],
        isError=False,
    )

    with console.console.capture() as capture:
        display.show_tool_result(result, name="dev", tool_name="execute")

    rendered = capture.get()
    assert "hello" in rendered
    assert "exit code 0" in rendered


def test_tool_result_prefers_structured_content_over_many_text_blocks() -> None:
    display = ConsoleDisplay()
    result = CallToolResult(
        content=[
            TextContent(type="text", text='{"id":"a"}'),
            TextContent(type="text", text='{"id":"b"}'),
        ],
        isError=False,
    )
    setattr(
        result,
        "structuredContent",
        {"result": [{"id": "a"}, {"id": "b"}]},
    )

    with console.console.capture() as capture:
        display.show_tool_result(result, name="dev", tool_name="voice__crm_tickets")

    rendered = capture.get()
    assert '"result"' in rendered
    assert '"id": "a"' in rendered
    assert '"id": "b"' in rendered
    assert "TextContent(" not in rendered
    assert "text only" in rendered
    assert "TextContent mismatch" not in rendered


def test_default_tool_result_status_pluralizes_text_blocks() -> None:
    result = CallToolResult(
        content=[
            TextContent(type="text", text="a"),
            TextContent(type="text", text="b"),
        ],
        isError=False,
    )

    assert ToolDisplay._default_tool_result_status(ToolDisplay(ConsoleDisplay()), result) == "2 Text Blocks"


def test_default_tool_result_status_formats_text_only_char_count() -> None:
    display = ToolDisplay(ConsoleDisplay())

    assert (
        ToolDisplay._default_tool_result_status(
            display,
            CallToolResult(content=[TextContent(type="text", text="")], isError=False),
        )
        == "text only 0 chars"
    )
    assert (
        ToolDisplay._default_tool_result_status(
            display,
            CallToolResult(content=[TextContent(type="text", text="x")], isError=False),
        )
        == "text only 1 char"
    )
    assert (
        ToolDisplay._default_tool_result_status(
            display,
            CallToolResult(content=[TextContent(type="text", text="xy")], isError=False),
        )
        == "text only 2 chars"
    )


def test_default_tool_result_status_pluralizes_mixed_content_blocks() -> None:
    result = CallToolResult(
        content=[
            TextContent(type="text", text="a"),
            ImageContent(type="image", data="abc", mimeType="image/png"),
        ],
        isError=False,
    )

    assert (
        ToolDisplay._default_tool_result_status(ToolDisplay(ConsoleDisplay()), result)
        == "2 Content Blocks"
    )


def test_tool_result_prefers_structured_content_when_text_blocks_disagree() -> None:
    display = ConsoleDisplay()
    result = CallToolResult(
        content=[
            TextContent(type="text", text='{"id":"a","status":"closed"}'),
            TextContent(type="text", text='{"id":"b","status":"pending"}'),
        ],
        isError=False,
    )
    setattr(
        result,
        "structuredContent",
        {
            "result": [
                {"id": "a", "status": "open"},
                {"id": "b", "status": "escalated"},
            ]
        },
    )

    with console.console.capture() as capture:
        display.show_tool_result(result, name="dev", tool_name="voice__crm_tickets")

    rendered = capture.get()
    assert '"status": "open"' in rendered
    assert '"status": "escalated"' in rendered
    assert '"status":"closed"' not in rendered
    assert '"status":"pending"' not in rendered
    assert "Structured ■ (TextContent mismatch)" in rendered


def test_structured_tool_result_shows_transport_timing_and_structured_footer() -> None:
    display = ConsoleDisplay()
    result = CallToolResult(
        content=[TextContent(type="text", text='{"ok": true}')],
        structuredContent={"ok": True},
        isError=False,
    )
    setattr(result, "transport_channel", "post-json")

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="demo_tool",
            timing_ms=1500,
        )

    rendered = capture.get()
    assert '{"ok": true}' in rendered
    assert "HTTP (JSON-RPC)" in rendered
    assert "1.50s" in rendered
    assert "Structured ■" in rendered


def test_structured_tool_result_shows_unknown_transport_label_uppercase() -> None:
    display = ConsoleDisplay()
    result = CallToolResult(
        content=[TextContent(type="text", text="ok")],
        structuredContent={"ok": True},
        isError=False,
    )
    setattr(result, "transport_channel", "custom")

    with console.console.capture() as capture:
        display.show_tool_result(result, name="dev", tool_name="demo_tool", timing_ms=12)

    assert "CUSTOM" in capture.get()


def test_tool_result_display_metadata_filters_unexpected_attribute_types() -> None:
    display = ToolDisplay(ConsoleDisplay())
    result = CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)
    setattr(result, "read_text_file_path", 123)
    setattr(result, "read_text_file_line", "1")
    setattr(result, "read_text_file_limit", True)
    setattr(result, "transport_channel", "   ")
    setattr(result, "output_line_count", 0)

    metadata = display._tool_result_display_metadata(result)

    assert metadata.read_text_file_path is None
    assert metadata.read_text_file_line is None
    assert metadata.read_text_file_limit is None
    assert metadata.transport_channel is None
    assert metadata.output_line_count is None


def test_tool_result_display_metadata_accepts_positive_int_attributes() -> None:
    display = ToolDisplay(ConsoleDisplay())
    result = CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)
    setattr(result, "read_text_file_line", 1)
    setattr(result, "read_text_file_limit", 30)
    setattr(result, "output_line_count", 2)

    metadata = display._tool_result_display_metadata(result)

    assert metadata.read_text_file_line == 1
    assert metadata.read_text_file_limit == 30
    assert metadata.output_line_count == 2
