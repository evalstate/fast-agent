from mcp.types import CallToolResult, TextContent

from fast_agent.config import Settings, ShellSettings
from fast_agent.constants import FAST_AGENT_SHELL_PROCESS_METADATA
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.progress_display import progress_display
from fast_agent.ui.shell_output_truncation import SHELL_OUTPUT_TRUNCATION_MARKER


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

    rendered = " ".join(capture.get().split())
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


def test_poll_process_result_hides_process_metadata_and_keeps_exit_banner() -> None:
    display = ConsoleDisplay()
    result = CallToolResult(
        content=[
            TextContent(
                type="text",
                text="finished\nprocess_id: process-1\nprocess exit code was 0",
            )
        ],
        isError=False,
    )
    setattr(result, "output_line_count", 1)

    with console.console.capture() as capture:
        display.show_tool_result(result, name="dev", tool_name="poll_process")

    rendered = capture.get()
    assert "finished" in rendered
    assert "process_id:" not in rendered
    assert "exit code 0" in rendered
    assert "1 line" in rendered


def test_running_process_result_uses_compact_lifecycle_line() -> None:
    display = ConsoleDisplay()
    result = CallToolResult(
        content=[
            TextContent(
                type="text",
                text="\n".join(
                    [
                        "building",
                        "Process is still running because it reached the foreground yield threshold.",
                        "process_id: process-2",
                        "os_pid: 4321",
                        "elapsed_seconds: 30.0",
                        "total_output_bytes: 9",
                        "Use poll_process to monitor it or terminate_process to stop it.",
                    ]
                ),
            )
        ],
        isError=False,
    )

    with console.console.capture() as capture:
        display.show_tool_result(result, name="dev", tool_name="execute")

    rendered = capture.get()
    assert "building" in rendered
    assert "▶ process-2 running • foreground yield • 30.0s • pid 4321" in rendered
    assert "total_output_bytes" not in rendered
    assert "Use poll_process" not in rendered


def test_quiet_running_poll_result_is_not_rendered() -> None:
    display = ConsoleDisplay()
    result = CallToolResult(
        content=[
            TextContent(
                type="text",
                text="\n".join(
                    [
                        "Process is still running.",
                        "process_id: process-2",
                        "elapsed_seconds: 30.0",
                        "total_output_bytes: 0",
                    ]
                ),
            )
        ],
        isError=False,
    )
    result.meta = {
        FAST_AGENT_SHELL_PROCESS_METADATA: {
            "process_id": "process-2",
            "process_status": "running",
            "poll_wait_sec": 30,
        }
    }

    for tool_name in ("poll_process", "Process"):
        with console.console.capture() as capture:
            display.show_tool_result(result, name="dev", tool_name=tool_name)

        assert capture.get() == ""


def test_process_non_poll_result_is_rendered() -> None:
    display = ConsoleDisplay()
    result = CallToolResult(
        content=[TextContent(type="text", text="Process is still running.")],
        isError=False,
    )

    with console.console.capture() as capture:
        display.show_tool_result(result, name="dev", tool_name="Process")

    assert "Process is still running" in capture.get()


def test_managed_process_poll_uses_shared_elapsed_format() -> None:
    display = ConsoleDisplay()
    progress_display.set_default_agent_name("dev")

    try:
        with console.console.capture() as capture:
            display.show_managed_process_poll(
                name=None,
                process_id="process-2",
                command="uv run worker.py",
                elapsed_seconds=7_200,
                wait_sec=50,
                has_observed_output=True,
                seconds_since_last_output=9,
                total_output_bytes=12_500,
                tool_call_id="call_abcdef0123456789",
            )
    finally:
        progress_display.set_default_agent_name(None)

    rendered = " ".join(capture.get().split())
    assert "dev" not in rendered
    assert (
        "process-2 · 2h · output 9s ago · 12.5KB · uv run worker.py · id: call_…456789"
    ) in rendered


def test_terminate_process_result_uses_compact_lifecycle_line() -> None:
    display = ConsoleDisplay()
    result = CallToolResult(
        content=[
            TextContent(
                type="text",
                text="process_id: process-3\noutcome: terminated",
            )
        ],
        isError=False,
    )

    with console.console.capture() as capture:
        display.show_tool_result(result, name="dev", tool_name="terminate_process")

    rendered = capture.get()
    assert "▶ process-3 terminated" in rendered
    assert "outcome:" not in rendered


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
