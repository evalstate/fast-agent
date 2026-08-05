import io

import pytest
from click.utils import strip_ansi
from mcp_types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    TextContent,
)
from rich.console import Console, Group
from rich.syntax import Syntax

from fast_agent.config import LoggerSettings, Settings, ShellSettings, ToolDisplaySettings
from fast_agent.constants import FAST_AGENT_SHELL_PROCESS_METADATA
from fast_agent.mcp.tool_result_metadata import update_tool_result_display_metadata
from fast_agent.types import PromptMessageExtended
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.history_actions import display_history_turn
from fast_agent.ui.progress_display import progress_display
from fast_agent.ui.streaming.display import StreamingMessageHandle


def _display(
    *,
    tool_display: ToolDisplaySettings | None = None,
    output_lines: int = 5,
) -> ConsoleDisplay:
    return ConsoleDisplay(
        Settings(
            logger=LoggerSettings(tool_display=tool_display or ToolDisplaySettings()),
            shell_execution=ShellSettings(output_display_lines=output_lines),
        )
    )


def test_compact_is_default_and_full_remains_available() -> None:
    settings = ToolDisplaySettings()

    assert settings.layout == "compact"
    assert settings.arguments == "auto"
    assert settings.results == "auto"
    assert not settings.show_successful_file_reads
    assert settings.stream_edit_previews == "primary"
    assert settings.aggregate_parallel


def test_compact_mcp_call_shows_arguments_and_hides_successful_result() -> None:
    display = _display()
    result = CallToolResult(
        content=[TextContent(type="text", text="large result body")],
        is_error=False,
    )

    with console.console.capture() as capture:
        display.show_tool_call(
            "huggingface__hf_fs",
            {"path": "datasets/example"},
            name="agent",
            source_label="MCP",
            server_name="huggingface",
            tool_call_id="call_abcdef0123456789",
        )
        display.show_tool_result(
            result,
            name="agent",
            tool_name="huggingface__hf_fs",
            source_label="MCP",
            server_name="huggingface",
            timing_ms=12_000,
            tool_call_id="call_abcdef0123456789",
        )

    rendered = " ".join(capture.get().split())
    assert "▎◀ agent tool (MCP) huggingface hf_fs · id: call_…456789" in rendered
    assert (
        "▎▶ agent tool (MCP) huggingface hf_fs · text only 17 chars · 12.0s · id: call_…456789"
    ) in rendered
    assert "datasets/example" in rendered
    assert "large result body" not in rendered


def test_compact_generic_arguments_use_six_unwrapped_rows_with_truncation() -> None:
    original_console = console.console
    output = io.StringIO()
    console.console = Console(
        file=output,
        force_terminal=True,
        color_system="standard",
        width=36,
    )
    try:
        _display().show_tool_call(
            "lookup",
            {
                "emoji": "日本語",
                "long_value": "x" * 100,
                "first": 1,
                "second": 2,
                "third": 3,
                "fourth": 4,
            },
        )
    finally:
        console.console = original_console

    lines = strip_ansi(output.getvalue()).splitlines()
    assert len(lines) == 7
    assert "日本語" in lines[2]
    assert '"long_value"' in lines[3]
    assert "x" * 100 not in output.getvalue()
    assert lines[-1].strip() == "…"


def test_compact_argument_policies_hide_all_bodies_or_show_all_arguments() -> None:
    hidden = _display(tool_display=ToolDisplaySettings(arguments="none"))
    shown = _display(tool_display=ToolDisplaySettings(arguments="all"))
    values = {f"key_{index}": index for index in range(8)}

    with console.console.capture() as capture:
        hidden.show_tool_call("lookup", {"query": "hidden"})
        hidden.show_tool_call(
            "Bash",
            {"command": "hidden shell command"},
            metadata={"variant": "shell", "command": "hidden shell command"},
        )
        shown.show_tool_call("lookup", values)

    rendered = capture.get()
    assert "hidden" not in rendered
    assert "key_0" in rendered
    assert "key_7" in rendered


def test_compact_auto_redacts_nested_credentials_without_hiding_similar_keys() -> None:
    automatic = _display()
    shown = _display(tool_display=ToolDisplaySettings(arguments="all"))
    arguments = {
        "api_key": "auto-secret",
        "token_count": 42,
        "nested": {"Authorization": "Bearer nested-secret"},
    }

    with console.console.capture() as capture:
        automatic.show_tool_call("lookup", arguments)
        shown.show_tool_call("lookup", arguments)

    automatic_output, shown_output = capture.get().split("▎◀ tool lookup")[1:]
    assert automatic_output.count("[REDACTED]") == 2
    assert "auto-secret" not in automatic_output
    assert "nested-secret" not in automatic_output
    assert "token_count" in automatic_output
    assert "42" in automatic_output
    assert "auto-secret" in shown_output
    assert "nested-secret" in shown_output


def test_compact_file_reads_hide_all_successes_and_show_error_summary() -> None:
    display = _display(output_lines=2)
    complete = CallToolResult(
        content=[TextContent(type="text", text="one\ntwo")],
        is_error=False,
    )
    partial = CallToolResult(
        content=[TextContent(type="text", text="one\ntwo\nthree\nfour\nfive")],
        is_error=False,
    )
    error = CallToolResult(
        content=[TextContent(type="text", text="Permission denied")],
        is_error=True,
    )
    for result in (complete, partial, error):
        update_tool_result_display_metadata(
            result,
            {
                "read_text_file_path": "llm/sampling_converter.py",
                "read_text_file_line": 90,
                "read_text_file_limit": 60,
            },
        )

    with console.console.capture() as capture:
        display.show_tool_call(
            "read_text_file",
            {"path": "llm/sampling_converter.py", "line": 90, "limit": 60},
            name="dev",
        )
        display.show_tool_result(complete, name="dev", tool_name="read_text_file")
        display.show_tool_result(partial, name="dev", tool_name="read_text_file")
        display.show_tool_result(error, name="dev", tool_name="read_text_file")

    rendered = capture.get()
    assert rendered.count("file read") == 1
    assert "sampling_converter.py" not in rendered
    assert "ERROR" in rendered
    assert "Permission denied" not in rendered
    assert "\none\n" not in rendered


def test_compact_shell_collapses_result_to_inverse_exit_summary() -> None:
    display = _display()
    result = CallToolResult(
        content=[TextContent(type="text", text="lint passed\nprocess exit code was 0")],
        is_error=False,
    )
    update_tool_result_display_metadata(result, {"output_line_count": 1})

    with console.console.capture() as capture:
        display.show_tool_call(
            "Bash",
            {"command": "uv run scripts/lint.py"},
            name="dev",
            metadata={
                "variant": "shell",
                "command": "uv run scripts/lint.py",
                "shell_name": "bash",
                "shell_path": "/bin/bash",
                "idle_yield_seconds": 10,
                "foreground_yield_seconds": 30,
            },
            tool_call_id="call_abcdef0123456789",
        )
        display.show_tool_result(
            result,
            name="dev",
            tool_name="Bash",
            tool_call_id="call_abcdef0123456789",
        )

    rendered = " ".join(capture.get().split())
    assert ("▎◀ dev bash (/bin/bash) | yield 10s idle / 30s total · id: call_…456789") in rendered
    assert "uv run scripts/lint.py" in rendered
    assert "lint passed" not in rendered
    assert "text only" not in rendered
    assert "▎▶ dev bash" not in rendered
    assert "exit code 0" in rendered
    assert "process exit code was 0" not in rendered

    command_lines = [
        line for line in capture.get().splitlines() if "uv run scripts/lint.py" in line
    ]
    assert [line.strip() for line in command_lines] == ["uv run scripts/lint.py"]


def test_compact_metadata_code_tool_uses_specialized_preview() -> None:
    display = _display()

    with console.console.capture() as capture:
        display.show_tool_call(
            "run_code",
            {"source": "print('hello')", "label": "example"},
            metadata={"variant": "code", "code_arg": "source", "language": "python"},
        )

    rendered = capture.get()
    assert "print('hello')" in rendered
    assert '"source"' not in rendered


def test_compact_background_shell_result_identifies_started_process() -> None:
    display = _display()
    result = CallToolResult(
        content=[
            TextContent(
                type="text",
                text=(
                    "effective_lifecycle: persistent\n"
                    "Managed background process is still running.\n"
                    "process_id: process-7\n"
                    "elapsed_seconds: 0.1\n"
                    "Use process to inspect it."
                ),
            )
        ],
        is_error=False,
        meta={
            FAST_AGENT_SHELL_PROCESS_METADATA: {
                "process_id": "process-7",
                "lifecycle": "persistent",
                "process_status": "running",
                "process_yield_reason": "background",
            }
        },
    )

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="Bash",
            timing_ms=71,
            tool_call_id="call_abcdef0123456789",
        )

    rendered = " ".join(capture.get().split())
    assert "▎▶ dev bash · started process-7 · 71ms · id: call_…456789" in rendered
    assert "text only" not in rendered
    assert "effective_lifecycle" not in rendered
    assert "Use process" not in rendered


def test_shell_heredoc_uses_language_block_for_static_output_file() -> None:
    display = _display()
    command = (
        "cat > example.py <<'PY'\n"
        "from pathlib import Path\n"
        "print(Path.cwd())\n"
        "PY\n"
        "python example.py"
    )

    content = display._tool_display._shell_tool_call_content(
        command=command,
        tool_args={"command": command},
        metadata={"shell_name": "bash", "shell_path": "/bin/bash"},
    )

    assert isinstance(content, Group)
    syntaxes = [renderable for renderable in content.renderables if isinstance(renderable, Syntax)]
    lexer_names: list[str] = []
    for syntax in syntaxes:
        lexer = syntax.lexer
        assert lexer is not None
        lexer_name = lexer.name
        assert isinstance(lexer_name, str)
        lexer_names.append(lexer_name)
    assert lexer_names == [
        "Bash",
        "Python",
        "Bash",
    ]


def test_shell_heredoc_uses_language_block_for_direct_interpreter() -> None:
    display = _display()
    command = "python - <<'PY'\nprint('hello')\nPY\nrm -rf /tmp/example"

    content = display._tool_display._shell_tool_call_content(
        command=command,
        tool_args={"command": command},
        metadata={"shell_name": "bash", "shell_path": "/bin/bash"},
    )

    assert isinstance(content, Group)
    syntaxes = [renderable for renderable in content.renderables if isinstance(renderable, Syntax)]
    assert [syntax._lexer for syntax in syntaxes] == ["bash", "python", "bash"]


def test_compact_completed_process_collapses_to_exit_summary() -> None:
    display = _display()
    result = CallToolResult(
        content=[
            TextContent(
                type="text",
                text=(
                    "background process started\n"
                    "background process completed\n"
                    "process_id: process-5\n"
                    "process exit code was 0\n"
                    "output_activity: 2 lines / 56 bytes since last poll"
                ),
            )
        ],
        is_error=False,
        meta={
            FAST_AGENT_SHELL_PROCESS_METADATA: {
                "process_id": "process-5",
                "process_status": "completed",
                "output_line_count": 2,
            }
        },
    )

    with console.console.capture() as capture:
        display.show_tool_result(
            result,
            name="dev",
            tool_name="Process",
            tool_call_id="call_abcdef0123456789",
        )

    rendered = " ".join(capture.get().split())
    assert "▎ exit code 0 2 lines id: call_…456789" in rendered
    assert "▎▶ dev process" not in rendered
    assert "background process started" not in rendered
    assert "output_activity" not in rendered


def test_full_layout_preserves_detailed_generic_bodies() -> None:
    display = _display(tool_display=ToolDisplaySettings(layout="full"))
    result = CallToolResult(
        content=[TextContent(type="text", text="result body")],
        is_error=False,
    )

    with console.console.capture() as capture:
        display.show_tool_call("lookup", {"query": "needle"}, name="dev")
        display.show_tool_result(result, name="dev", tool_name="lookup")

    rendered = capture.get()
    assert "needle" in rendered
    assert "result body" in rendered


def test_primary_edit_preview_streaming_is_on_only_for_default_agent() -> None:
    display = _display()
    progress_display.set_default_agent_name("main")
    try:
        with display.streaming_assistant_message(name="main") as main_handle:
            assert isinstance(main_handle, StreamingMessageHandle)
            assert main_handle._segment_assembler._stream_edit_previews
        with display.streaming_assistant_message(name="child") as child_handle:
            assert isinstance(child_handle, StreamingMessageHandle)
            assert not child_handle._segment_assembler._stream_edit_previews
    finally:
        progress_display.set_default_agent_name(None)


@pytest.mark.asyncio
async def test_history_review_forces_full_tool_bodies() -> None:
    config = Settings(logger=LoggerSettings(tool_display=ToolDisplaySettings(layout="compact")))
    text_tail = "history-text-tail"
    structured_tail = "structured-history-tail"
    turn = [
        PromptMessageExtended(
            role="assistant",
            content=[],
            tool_calls={
                "call_history_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="lookup",
                        arguments={"query": "history needle"},
                    ),
                )
            },
        ),
        PromptMessageExtended(
            role="user",
            content=[],
            tool_results={
                "call_history_1": CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"history text content body {'x' * 2000} {text_tail}",
                        )
                    ],
                    structured_content={
                        "records": [
                            {
                                "id": "structured-history-record",
                                "detail": f"{'y' * 2000} {structured_tail}",
                            }
                        ],
                    },
                    is_error=False,
                )
            },
        ),
    ]

    with console.console.capture() as capture:
        await display_history_turn("dev", turn, config=config)

    rendered = capture.get()
    assert "history needle" in rendered
    assert "history text content body" in rendered
    assert text_tail in rendered
    assert "■ structured content" in rendered
    assert rendered.count("structured-history-record") == 1
    assert structured_tail in rendered
