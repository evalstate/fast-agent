from mcp.types import CallToolResult, TextContent

from fast_agent.config import Settings, ShellSettings
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay


def test_shell_tool_result_uses_styled_exit_line() -> None:
    display = ConsoleDisplay()
    result = CallToolResult(
        content=[TextContent(type="text", text="hello\nprocess exit code was 0")],
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
    assert "hello" in rendered
    assert "exit code 0" in rendered
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
    assert "process exit code was 0" not in rendered
    assert "(empty text)" not in rendered


def test_shell_tool_result_truncates_with_head_and_tail_windows() -> None:
    display = ConsoleDisplay(config=Settings(shell_execution=ShellSettings(output_display_lines=6)))
    output_lines = [f"out-{i:02d}" for i in range(1, 11)]
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
    assert "..." in rendered
