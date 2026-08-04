"""Render deterministic tool-display scenarios for visual comparison."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from mcp_types import CallToolResult, TextContent

from fast_agent.config import LoggerSettings, Settings, ShellSettings, ToolDisplaySettings
from fast_agent.mcp.tool_result_metadata import update_tool_result_display_metadata
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.tool_display import ToolCallDisplayRequest, ToolResultDisplayRequest

Scenario = Callable[[ConsoleDisplay], None]
Layout = Literal["compact", "full"]


def _result(text: str, *, error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        is_error=error,
    )


def _mcp(display: ConsoleDisplay) -> None:
    display.show_tool_call(
        "huggingface__hf_fs",
        {"path": "datasets/evalstate/tool-fixtures", "detail": True},
        name="agent",
        source_label="MCP",
        server_name="huggingface",
        tool_call_id="call_0123456789abcdef",
    )
    display.show_tool_result(
        _result('{"kind":"directory","entries":12}'),
        name="agent",
        tool_name="huggingface__hf_fs",
        source_label="MCP",
        server_name="huggingface",
        timing_ms=12_000,
        tool_call_id="call_0123456789abcdef",
    )


def _parallel(display: ConsoleDisplay) -> None:
    calls = [
        ToolCallDisplayRequest(
            tool_name="huggingface__hf_fs",
            tool_args={"path": f"datasets/example/{index}"},
            name="agent",
            source_label="MCP",
            server_name="huggingface",
            tool_call_id=f"call_parallel_{index:02d}",
        )
        for index in range(1, 6)
    ]
    results = [
        ToolResultDisplayRequest(
            result=_result(f"entry-{index}\n"),
            name="agent",
            tool_name="huggingface__hf_fs",
            source_label="MCP",
            server_name="huggingface",
            timing_ms=8_000 + index * 100,
            tool_call_id=f"call_parallel_{index:02d}",
        )
        for index in range(1, 6)
    ]
    display.show_parallel_tool_calls(calls)
    display.show_parallel_tool_results(results)


def _reads(display: ConsoleDisplay) -> None:
    complete = _result("line 90\nline 91")
    partial = _result("\n".join(f"line {line}" for line in range(90, 98)))
    error = _result("Permission denied: llm/private.py", error=True)
    for result, path in (
        (complete, "llm/sampling_converter.py"),
        (partial, "llm/sampling_converter.py"),
        (error, "llm/private.py"),
    ):
        update_tool_result_display_metadata(
            result,
            {
                "read_text_file_path": path,
                "read_text_file_line": 90,
                "read_text_file_limit": 60,
            },
        )
        display.show_tool_result(result, name="dev", tool_name="read_text_file")


def _shell(display: ConsoleDisplay) -> None:
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
        tool_call_id="call_shell_123456789",
    )
    result = _result("All checks passed!\nprocess exit code was 0")
    update_tool_result_display_metadata(result, {"output_line_count": 1})
    display.show_tool_result(
        result,
        name="dev",
        tool_name="Bash",
        timing_ms=2_140,
        tool_call_id="call_shell_123456789",
    )


def _edits(display: ConsoleDisplay) -> None:
    display.show_tool_call(
        "edit_file",
        {
            "path": "src/fast_agent/ui/tool_display.py",
            "old_string": 'layout = "full"\n',
            "new_string": 'layout = "compact"\n',
        },
        name="dev",
        source_label="Shell",
        tool_call_id="call_edit_123456789",
    )
    display.show_tool_result(
        _result("Success. Replaced 1 match."),
        name="dev",
        tool_name="edit_file",
        source_label="Shell",
        timing_ms=34,
        tool_call_id="call_edit_123456789",
    )


SCENARIOS: tuple[tuple[str, Scenario], ...] = (
    ("MCP call/result", _mcp),
    ("Parallel same-tool calls", _parallel),
    ("File reads", _reads),
    ("Shell lifecycle", _shell),
    ("Built-in edit", _edits),
)


def render(layout: Layout) -> str:
    settings = Settings(
        logger=LoggerSettings(tool_display=ToolDisplaySettings(layout=layout)),
        shell_execution=ShellSettings(output_display_lines=4),
    )
    display = ConsoleDisplay(settings)
    sections: list[str] = []
    for title, scenario in SCENARIOS:
        with console.console.capture() as capture:
            scenario(display)
        sections.extend((f"### {title}", "", "```text", capture.get().rstrip(), "```", ""))
    return "\n".join(sections).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        choices=("compact", "full", "compare"),
        default="compare",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    layouts: tuple[Layout, ...] = (
        ("compact", "full") if args.layout == "compare" else (args.layout,)
    )
    content = "\n\n".join(f"## {layout.title()}\n\n{render(layout)}" for layout in layouts)
    content = (
        "# Tool display gallery\n\n"
        "Generated with `COLUMNS=100 uv run scripts/tool_display_gallery.py --layout compare`.\n\n"
        f"{content}\n"
    )
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
