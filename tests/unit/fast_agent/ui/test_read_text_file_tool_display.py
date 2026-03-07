from mcp.types import CallToolResult, TextContent

from fast_agent.config import Settings, ShellSettings
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.tool_display import ToolDisplay


def test_read_text_file_tool_call_shows_summary_with_offset() -> None:
    display = ConsoleDisplay()
    long_path = "/tmp/" + "/".join(["very-long-directory-name"] * 8) + "/target_file.py"

    with console.console.capture() as capture:
        display.show_tool_call(
            tool_name="read_text_file",
            tool_args={
                "path": long_path,
                "line": 93,
                "limit": 30,
            },
            name="dev",
        )

    rendered = capture.get()
    assert "The assistant is reading 30 lines from" in rendered
    assert "(offset 93)." in rendered
    assert "target_file.py" in rendered


def test_read_text_file_result_truncates_with_head_and_more_lines_note() -> None:
    display = ConsoleDisplay(config=Settings(shell_execution=ShellSettings(output_display_lines=4)))
    output_lines = [f"line-{i}" for i in range(1, 8)]
    result_text = "\n".join(output_lines)
    result = CallToolResult(content=[TextContent(type="text", text=result_text)], isError=False)

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="read_text_file",
        )

    rendered = capture.get()
    assert "line-1" in rendered
    assert "line-2" in rendered
    assert "line-3" in rendered
    assert "line-4" in rendered
    assert "line-5" not in rendered
    assert "line-6" not in rendered
    assert "line-7" not in rendered
    assert "(+3 more lines)" in rendered


def test_read_text_file_result_skips_truncation_when_only_two_lines_over_limit() -> None:
    display = ConsoleDisplay(config=Settings(shell_execution=ShellSettings(output_display_lines=4)))
    output_lines = [f"line-{i}" for i in range(1, 7)]
    result_text = "\n".join(output_lines)
    result = CallToolResult(content=[TextContent(type="text", text=result_text)], isError=False)

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="read_text_file",
        )

    rendered = capture.get()
    for line in output_lines:
        assert line in rendered
    assert "more lines" not in rendered


def test_read_text_file_result_hides_content_when_line_limit_is_zero() -> None:
    display = ConsoleDisplay(config=Settings(shell_execution=ShellSettings(output_display_lines=0)))
    output_lines = [f"line-{i}" for i in range(1, 4)]
    result_text = "\n".join(output_lines)
    result = CallToolResult(content=[TextContent(type="text", text=result_text)], isError=False)

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="read_text_file",
        )

    rendered = capture.get()
    for line in output_lines:
        assert line not in rendered
    assert "(+3 more lines)" in rendered
    assert "(No lines returned)" not in rendered


def test_read_text_file_result_shows_no_lines_message_when_empty() -> None:
    display = ConsoleDisplay(config=Settings(shell_execution=ShellSettings(output_display_lines=4)))
    result = CallToolResult(content=[TextContent(type="text", text="")], isError=False)
    setattr(result, "read_text_file_path", "/tmp/one/two/example.py")
    setattr(result, "read_text_file_line", 300)
    setattr(result, "read_text_file_limit", 80)

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="read_text_file",
            type_label="file read",
        )

    rendered = capture.get()
    assert "file read - two/example.py (offset 300, 80 lines)" in rendered
    assert "(No lines returned)" in rendered
    assert "(empty text)" not in rendered
    assert not [line for line in rendered.splitlines() if line and not line.strip()]


def test_read_text_file_truncation_skips_leading_blank_lines() -> None:
    tool_display = ToolDisplay(ConsoleDisplay())
    text = "\n\n   \nline-1\nline-2\nline-3\nline-4\nline-5\nline-6"

    limited, omitted = tool_display._limit_read_text_output_text(text, line_limit=4)

    assert limited.splitlines() == ["line-1", "line-2", "line-3", "line-4"]
    assert omitted == 5


def test_read_text_file_markdown_wrap_uses_language_from_path() -> None:
    tool_display = ToolDisplay(ConsoleDisplay())
    content = [TextContent(type="text", text="def f() -> int:\n    return 1")]

    wrapped = tool_display._format_read_text_file_content_as_markdown(
        content,
        path_value="/tmp/example.py",
    )

    assert wrapped is not None
    assert wrapped.startswith("```python\n")
    assert wrapped.endswith("\n```")


def test_read_text_file_result_header_uses_preview_status() -> None:
    display = ConsoleDisplay(config=Settings(shell_execution=ShellSettings(output_display_lines=3)))
    result_text = "\n".join(["a", "b", "c", "d", "e"])
    result = CallToolResult(content=[TextContent(type="text", text=result_text)], isError=False)

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="read_text_file",
            type_label="file read",
        )

    rendered = capture.get()
    assert "file read - preview" in rendered
    assert "text only" not in rendered


def test_read_text_file_result_header_shows_short_path_when_available() -> None:
    display = ConsoleDisplay(config=Settings(shell_execution=ShellSettings(output_display_lines=3)))
    result_text = "\n".join(["a", "b", "c", "d", "e"])
    result = CallToolResult(content=[TextContent(type="text", text=result_text)], isError=False)
    setattr(result, "read_text_file_path", "/tmp/one/two/example.py")

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="read_text_file",
            type_label="file read",
        )

    rendered = capture.get()
    assert "file read - two/example.py" in rendered


def test_read_text_file_result_header_includes_offset_and_limit_when_available() -> None:
    display = ConsoleDisplay(config=Settings(shell_execution=ShellSettings(output_display_lines=3)))
    result_text = "\n".join(["a", "b", "c", "d", "e"])
    result = CallToolResult(content=[TextContent(type="text", text=result_text)], isError=False)
    setattr(result, "read_text_file_path", "/tmp/one/two/example.py")
    setattr(result, "read_text_file_line", 93)
    setattr(result, "read_text_file_limit", 30)

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="read_text_file",
            type_label="file read",
        )

    rendered = capture.get()
    assert "file read - two/example.py (offset 93, 30 lines)" in rendered
