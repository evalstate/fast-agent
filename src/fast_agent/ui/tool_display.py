from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from rich.markup import escape as escape_markup
from rich.syntax import Syntax
from rich.text import Text

from fast_agent.config import Settings, ShellSettings
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.tool_result_metadata import get_tool_result_media_preview
from fast_agent.tools.apply_patch_tool import extract_apply_patch_input, is_apply_patch_tool_name
from fast_agent.ui import console
from fast_agent.ui.apply_patch_preview import (
    build_apply_patch_preview,
    build_apply_patch_preview_from_input,
    extract_non_command_args,
    format_apply_patch_preview,
    is_shell_execution_tool,
    shell_syntax_language,
    style_apply_patch_preview_text,
)
from fast_agent.ui.message_primitives import MESSAGE_CONFIGS, MessageType
from fast_agent.ui.shell_output_truncation import (
    format_shell_output_line_count,
    truncate_shell_output_lines,
)
from fast_agent.ui.tool_call_ids import format_tool_call_id
from fast_agent.utils.count_display import format_count
from fast_agent.utils.numeric import positive_int_or_none
from fast_agent.utils.path_display import fit_path_for_display
from fast_agent.utils.text import strip_casefold, strip_str_to_none
from fast_agent.utils.tool_names import (
    POLL_PROCESS_TOOL_NAME,
    SHELL_EXECUTION_TOOL_NAMES,
    is_read_text_file_tool_name,
    normalize_tool_name,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mcp.types import CallToolResult
    from rich.console import RenderableType

    from fast_agent.mcp.skybridge import SkybridgeServerConfig
    from fast_agent.ui.console_display import ConsoleDisplay


@dataclass(frozen=True, slots=True)
class ShellOutputExitCodeExtraction:
    lines: list[str]
    exit_code_line: str | None


@dataclass(frozen=True, slots=True)
class LimitedReadTextOutput:
    text: str
    omitted_line_count: int


@dataclass(frozen=True, slots=True)
class PreparedToolResultContent:
    display_content: object
    source_content: object
    truncate_content: bool
    read_omitted_line_count: int = 0


@dataclass(frozen=True, slots=True)
class SkybridgeResultDetails:
    is_skybridge_tool: bool
    resource_uri: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedReadTextFileResultDisplay:
    display_content: object
    render_markdown: bool | None = None
    additional_message: Text | None = None


@dataclass(frozen=True, slots=True)
class ToolResultDisplayMetadata:
    read_text_file_path: str | None = None
    read_text_file_line: int | None = None
    read_text_file_limit: int | None = None
    transport_channel: str | None = None
    output_line_count: int | None = None


@dataclass(slots=True)
class PreparedToolCallDisplay:
    content: object
    right_info: str
    bottom_items: list[str] | None
    highlight_indexes: list[int]
    max_item_length: int | None
    truncate_content: bool = True
    render_markdown: bool | None = None


_TRANSPORT_METADATA_LABELS: dict[str, str] = {
    "post-json": "HTTP (JSON-RPC)",
    "post-sse": "HTTP (SSE)",
    "get": "Legacy SSE",
    "resumption": "Resumption",
    "stdio": "STDIO",
}


class ToolDisplay:
    """Encapsulates rendering logic for tool calls and results."""

    _READ_TEXT_FILE_LANGUAGE_BY_EXTENSION: ClassVar[dict[str, str]] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
    }

    def __init__(self, display: "ConsoleDisplay") -> None:
        self._display = display

    @property
    def _markup(self) -> bool:
        return self._display._markup

    def _read_text_file_summary(self, tool_args: Mapping[str, Any]) -> str | None:
        path_value = tool_args.get("path")
        if not isinstance(path_value, str):
            return None

        stripped_path = path_value.strip()
        if not stripped_path:
            return None

        line = positive_int_or_none(tool_args.get("line"))
        limit = positive_int_or_none(tool_args.get("limit"))

        offset_suffix = f" (offset {line})." if line is not None else "."
        if limit is not None:
            prefix = f"The assistant is reading {format_shell_output_line_count(limit)} from "
        elif line is not None:
            prefix = "The assistant is reading from "
        else:
            prefix = "The assistant is reading a file from "

        max_width = max(16, console.console.size.width - len(prefix) - len(offset_suffix))
        display_path = fit_path_for_display(stripped_path, max_width)
        return f"{prefix}{display_path}{offset_suffix}"

    def _build_code_tool_call_syntax(
        self,
        tool_args: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> tuple[Syntax, list[str]]:
        code_arg = str(metadata.get("code_arg") or "code")
        language = str(metadata.get("language") or "text")
        raw_code = tool_args.get(code_arg)

        if isinstance(raw_code, str):
            code_text = raw_code.rstrip()
        elif raw_code is None:
            code_text = ""
        else:
            code_text = json.dumps(raw_code, ensure_ascii=False, indent=2).rstrip()

        footer_items: list[str] = []
        for key, value in tool_args.items():
            if key == code_arg:
                continue
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            footer_items.append(f"{key}: {rendered}")

        return (
            Syntax(
                code_text,
                language,
                theme=self._display.code_style,
                line_numbers=False,
                word_wrap=self._display.code_word_wrap,
            ),
            footer_items,
        )

    def _configured_output_line_limit(self) -> int | None:
        shell_settings = self._shell_settings()
        if shell_settings is None:
            return None
        return shell_settings.output_display_lines

    def _shell_settings(self) -> ShellSettings | None:
        config = self._display.config
        if isinstance(config, Settings):
            return config.shell_execution
        return None

    @staticmethod
    def _display_tool_name(tool_name: str) -> str:
        if tool_name.startswith("agent__"):
            return tool_name[7:]
        return tool_name

    @classmethod
    def _normalize_tool_footer_items(
        cls,
        bottom_items: list[str] | None,
        *,
        display_tool_name: str,
    ) -> list[str] | None:
        if not bottom_items:
            return bottom_items
        if len(bottom_items) != 1:
            return bottom_items
        only_item = cls._display_tool_name(bottom_items[0])
        if only_item == display_tool_name:
            return None
        return bottom_items

    @classmethod
    def _build_tool_right_info(cls, base_label: str | None, tool_call_id: str | None) -> str:
        parts: list[str] = []
        if base_label:
            parts.append(base_label)

        short_id = format_tool_call_id(tool_call_id)
        if short_id:
            parts.append(f"id: {short_id}")

        if not parts:
            return ""

        joined = " · ".join(parts)
        return f"[dim]{joined}[/dim]"

    def _shell_output_line_limit(self, tool_name: str | None) -> int | None:
        if not is_shell_execution_tool(tool_name):
            return None
        return self._configured_output_line_limit()

    def _read_text_file_output_line_limit(self, tool_name: str | None) -> int | None:
        if not is_read_text_file_tool_name(tool_name):
            return None
        return self._configured_output_line_limit()

    def _shell_show_bash(self, tool_name: str | None) -> bool:
        if not is_shell_execution_tool(tool_name):
            return True
        shell_settings = self._shell_settings()
        if shell_settings is None:
            return True
        return shell_settings.show_bash

    @staticmethod
    def _extract_exit_code_line(lines: list[str]) -> ShellOutputExitCodeExtraction:
        if not lines:
            return ShellOutputExitCodeExtraction(lines=lines, exit_code_line=None)
        index = len(lines) - 1
        while index >= 0 and not lines[index].strip():
            index -= 1
        if index < 0:
            return ShellOutputExitCodeExtraction(lines=lines, exit_code_line=None)
        candidate = lines[index].strip()
        if candidate.startswith(("[Exit code:", "process exit code was")):
            return ShellOutputExitCodeExtraction(lines=lines[:index], exit_code_line=candidate)
        return ShellOutputExitCodeExtraction(lines=lines, exit_code_line=None)

    @staticmethod
    def _parse_exit_code_value(exit_line: str | None) -> int | None:
        if not exit_line:
            return None

        bracket_match = re.search(r"\[Exit code:\s*(-?\d+)", exit_line)
        if bracket_match:
            return int(bracket_match.group(1))

        process_match = re.search(r"process exit code was\s+(-?\d+)", exit_line)
        if process_match:
            return int(process_match.group(1))

        return None

    def _shell_exit_detail(
        self,
        *,
        no_output: bool,
        tool_call_id: str | None,
        output_line_count: int | None,
    ) -> str | None:
        detail_parts: list[str] = []
        if output_line_count is not None and output_line_count > 0:
            detail_parts.append(format_shell_output_line_count(output_line_count))
        if no_output:
            detail_parts.append("(no output)")

        formatted_id = format_tool_call_id(tool_call_id)
        if formatted_id:
            detail_parts.append(f"id: {formatted_id}")

        if not detail_parts:
            return None
        return f" {' '.join(detail_parts)}"

    @staticmethod
    def _shell_output_line_count_from_content(content) -> int | None:
        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        if not content or len(content) != 1 or not is_text_content(content[0]):
            return None

        text = get_text(content[0]) or ""
        lines = cast("list[str]", text.splitlines())
        extraction = ToolDisplay._extract_exit_code_line(lines)
        return len(extraction.lines)

    def _build_shell_exit_additional_message(
        self,
        *,
        content,
        source_content,
        tool_name: str | None,
        tool_call_id: str | None,
        output_line_count: int | None = None,
    ):
        from mcp.types import TextContent

        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        if not tool_name:
            return content, None
        normalized_tool_name = normalize_tool_name(tool_name)
        if normalized_tool_name not in SHELL_EXECUTION_TOOL_NAMES:
            return content, None

        if not content or len(content) != 1 or not is_text_content(content[0]):
            return content, None

        text = get_text(content[0]) or ""
        lines = cast("list[str]", text.splitlines())
        extraction = self._extract_exit_code_line(lines)
        exit_code = self._parse_exit_code_value(extraction.exit_code_line)
        if exit_code is None:
            return content, None

        line_count = output_line_count
        if line_count is None:
            line_count = self._shell_output_line_count_from_content(source_content)
        if line_count is None:
            line_count = len(extraction.lines)

        no_output = not any(line.strip() for line in extraction.lines)
        detail = self._shell_exit_detail(
            no_output=no_output,
            tool_call_id=tool_call_id,
            output_line_count=line_count,
        )
        additional_message = self._display.style.shell_exit_line(
            exit_code,
            detail,
        )

        if not extraction.lines:
            return "", additional_message

        rendered_text = "\n".join(extraction.lines)
        return [TextContent(type="text", text=rendered_text)], additional_message

    def _limit_shell_output_text(self, text: str, line_limit: int) -> str:
        if line_limit < 0:
            return text
        lines = text.splitlines()
        if not lines:
            return text if line_limit != 0 else ""
        if line_limit == 0:
            extraction = self._extract_exit_code_line(lines)
            return extraction.exit_code_line or ""

        extraction = self._extract_exit_code_line(lines)
        if line_limit >= len(extraction.lines):
            return text

        output_lines, _ = truncate_shell_output_lines(extraction.lines, line_limit)
        if extraction.exit_code_line:
            output_lines.append(extraction.exit_code_line)
        return "\n".join(output_lines)

    def _limit_shell_output_content(self, content, line_limit: int):
        from mcp.types import TextContent

        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        if not content or len(content) != 1 or not is_text_content(content[0]):
            return content
        text = get_text(content[0]) or ""
        limited = self._limit_shell_output_text(text, line_limit)
        if limited == text:
            return content
        return [TextContent(type="text", text=limited)]

    def _limit_read_text_output_text(self, text: str, line_limit: int) -> LimitedReadTextOutput:
        if line_limit < 0:
            return LimitedReadTextOutput(text=text, omitted_line_count=0)

        lines = text.splitlines()
        if line_limit == 0:
            if not lines:
                return LimitedReadTextOutput(text=text, omitted_line_count=0)
            return LimitedReadTextOutput(text="", omitted_line_count=len(lines))

        if len(lines) <= line_limit + 2:
            return LimitedReadTextOutput(text=text, omitted_line_count=0)

        start_index = 0
        while start_index < len(lines) and not lines[start_index].strip():
            start_index += 1
        if start_index >= len(lines):
            start_index = 0

        visible_lines = lines[start_index : start_index + line_limit]
        omitted_line_count = len(lines) - len(visible_lines)
        return LimitedReadTextOutput(
            text="\n".join(visible_lines),
            omitted_line_count=omitted_line_count,
        )

    def _limit_read_text_output_content(self, content, line_limit: int):
        from mcp.types import TextContent

        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        if not content or len(content) != 1 or not is_text_content(content[0]):
            return content, 0
        text = get_text(content[0]) or ""
        limited = self._limit_read_text_output_text(text, line_limit)
        if limited.text == text and limited.omitted_line_count == 0:
            return content, 0
        return [TextContent(type="text", text=limited.text)], limited.omitted_line_count

    @staticmethod
    def _longest_backtick_run(text: str) -> int:
        matches = re.findall(r"`+", text)
        if not matches:
            return 0
        return max(len(match) for match in matches)

    @classmethod
    def _read_text_file_language_from_path(cls, path_value: object) -> str | None:
        if not isinstance(path_value, str):
            return None
        suffix = strip_casefold(Path(path_value).suffix)
        if not suffix:
            return None
        return cls._READ_TEXT_FILE_LANGUAGE_BY_EXTENSION.get(suffix)

    def _format_read_text_file_content_as_markdown(
        self,
        content,
        *,
        path_value: object,
    ) -> str | None:
        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        if not content or len(content) != 1 or not is_text_content(content[0]):
            return None

        text = get_text(content[0])
        if text is None:
            return None

        fence_length = max(3, self._longest_backtick_run(text) + 1)
        fence = "`" * fence_length
        language = self._read_text_file_language_from_path(path_value)
        opening_fence = f"{fence}{language}" if language else fence
        return f"{opening_fence}\n{text}\n{fence}"

    def _read_text_file_header_status(
        self,
        path_value: object,
        *,
        line_value: object = None,
        limit_value: object = None,
    ) -> str:
        if isinstance(path_value, str):
            stripped = path_value.strip()
            base_status = fit_path_for_display(stripped, 42) if stripped else "preview"
        else:
            base_status = "preview"

        line = positive_int_or_none(line_value)
        limit = positive_int_or_none(limit_value)

        details: list[str] = []
        if line is not None:
            details.append(f"offset {line}")
        if limit is not None:
            details.append(format_shell_output_line_count(limit))

        if not details:
            return base_status
        return f"{base_status} ({', '.join(details)})"

    @staticmethod
    def _read_text_file_more_lines_message(omitted_line_count: int) -> Text | None:
        if omitted_line_count <= 0:
            return None
        return Text(
            f"(+{format_count(omitted_line_count, 'more line')})",
            style="dim italic",
        )

    @staticmethod
    def _read_text_file_line_count_from_content(content) -> int | None:
        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        if not content or len(content) != 1 or not is_text_content(content[0]):
            return None
        text = get_text(content[0])
        if text is None:
            return None
        return len(text.splitlines())

    @staticmethod
    def _combine_additional_messages(
        *messages: Text | None,
    ) -> Text | None:
        present_messages = [message for message in messages if message is not None]
        if not present_messages:
            return None
        if len(present_messages) == 1:
            return present_messages[0]
        combined = Text()
        for index, message in enumerate(present_messages):
            if index:
                combined.append("\n")
            combined.append_text(message)
        return combined

    def _prepare_tool_result_content(
        self,
        *,
        content,
        structured_content: object = None,
        tool_name: str | None,
        truncate_content: bool,
    ) -> PreparedToolResultContent:
        source_content = content
        display_content = self._structured_tool_result_display_content(
            content=content,
            structured_content=structured_content,
        )
        display_content = self._compact_managed_process_result(
            display_content,
            tool_name=tool_name,
        )
        if not truncate_content:
            return PreparedToolResultContent(
                display_content=display_content,
                source_content=source_content,
                truncate_content=truncate_content,
            )

        show_bash_output = self._shell_show_bash(tool_name)
        if not show_bash_output:
            display_content = self._compact_managed_process_result(
                self._limit_shell_output_content(content, 0),
                tool_name=tool_name,
            )
            return PreparedToolResultContent(
                display_content=display_content,
                source_content=source_content,
                truncate_content=False,
            )

        shell_line_limit = self._shell_output_line_limit(tool_name)
        if shell_line_limit is not None:
            display_content = self._compact_managed_process_result(
                self._limit_shell_output_content(content, shell_line_limit),
                tool_name=tool_name,
            )
            return PreparedToolResultContent(
                display_content=display_content,
                source_content=source_content,
                truncate_content=False,
            )

        read_line_limit = self._read_text_file_output_line_limit(tool_name)
        if read_line_limit is None:
            return PreparedToolResultContent(
                display_content=display_content,
                source_content=source_content,
                truncate_content=truncate_content,
            )

        display_content, read_omitted_line_count = self._limit_read_text_output_content(
            display_content,
            read_line_limit,
        )
        return PreparedToolResultContent(
            display_content=display_content,
            source_content=source_content,
            truncate_content=False,
            read_omitted_line_count=read_omitted_line_count,
        )

    @staticmethod
    def _compact_managed_process_result(content, *, tool_name: str | None):
        """Hide model-oriented process metadata in favor of one lifecycle line."""
        from mcp.types import TextContent

        if normalize_tool_name(tool_name) not in SHELL_EXECUTION_TOOL_NAMES:
            return content
        if (
            not isinstance(content, list)
            or len(content) != 1
            or not isinstance(content[0], TextContent)
        ):
            return content

        lines = content[0].text.splitlines()
        process_line = next(
            (line for line in lines if line.startswith("process_id: ")),
            None,
        )
        if process_line is None:
            return content
        process_id = process_line.partition(":")[2].strip()

        outcome_line = next(
            (line for line in lines if line.startswith("outcome: ")),
            None,
        )
        if outcome_line is not None:
            outcome = outcome_line.partition(":")[2].strip().replace("_", " ")
            return [TextContent(type="text", text=f"▶ {process_id} {outcome}")]

        running_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.startswith("Process is still running")
            ),
            None,
        )
        if running_index is not None:
            metadata = {
                key.strip(): value.strip()
                for line in lines[running_index + 1 :]
                if ":" in line
                for key, value in [line.split(":", 1)]
            }
            reason = lines[running_index]
            detail_parts = ["running"]
            if "background" in reason:
                detail_parts.append("background")
            elif "no-output" in reason:
                detail_parts.append("idle yield")
            elif "foreground" in reason:
                detail_parts.append("foreground yield")
            if elapsed := metadata.get("elapsed_seconds"):
                detail_parts.append(f"{elapsed}s")
            if pid := metadata.get("os_pid"):
                detail_parts.append(f"pid {pid}")
            compact_line = f"▶ {process_id} {' • '.join(detail_parts)}"
            output_lines = lines[:running_index]
            return [
                TextContent(
                    type="text",
                    text="\n".join([*output_lines, compact_line]),
                )
            ]

        return [
            TextContent(
                type="text",
                text="\n".join(line for line in lines if line != process_line),
            )
        ]

    @staticmethod
    def _is_quiet_running_process_result(
        result: "CallToolResult",
        *,
        tool_name: str | None,
    ) -> bool:
        """Return whether a poll result contains liveness metadata but no new output."""
        from mcp.types import TextContent

        if normalize_tool_name(tool_name) != POLL_PROCESS_TOOL_NAME:
            return False
        content = result.content
        if (
            not isinstance(content, list)
            or len(content) != 1
            or not isinstance(content[0], TextContent)
        ):
            return False
        lines = content[0].text.splitlines()
        return bool(lines) and lines[0].startswith("Process is still running")

    @staticmethod
    def _structured_tool_result_display_content(
        *,
        content,
        structured_content: object = None,
    ):
        from mcp.types import TextContent

        from fast_agent.mcp.helpers.content_helpers import is_text_content

        if not (
            isinstance(structured_content, (dict, list))
            and isinstance(content, list)
            and len(content) > 1
            and all(is_text_content(item) for item in content)
        ):
            return content

        return [
            TextContent(
                type="text",
                text=json.dumps(structured_content, ensure_ascii=False, indent=2),
            )
        ]

    @staticmethod
    def _resolve_skybridge_result_details(
        *,
        has_structured: bool,
        tool_name: str | None,
        skybridge_config: "SkybridgeServerConfig | None",
    ) -> SkybridgeResultDetails:
        if not has_structured or not tool_name or skybridge_config is None:
            return SkybridgeResultDetails(is_skybridge_tool=False)

        for tool_cfg in skybridge_config.tools:
            if tool_cfg.tool_name == tool_name and tool_cfg.is_valid:
                resource_uri = (
                    str(tool_cfg.resource_uri) if tool_cfg.resource_uri is not None else None
                )
                return SkybridgeResultDetails(
                    is_skybridge_tool=True,
                    resource_uri=resource_uri,
                )

        return SkybridgeResultDetails(is_skybridge_tool=False)

    def _default_tool_result_status(self, result: "CallToolResult") -> str:
        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        content = self._structured_tool_result_display_content(
            content=result.content,
            structured_content=getattr(result, "structuredContent", None),
        )
        if result.isError:
            return "ERROR"

        if not content:
            return "No Content"

        if len(content) == 1 and is_text_content(content[0]):
            text_content = get_text(content[0])
            char_count = len(text_content) if text_content else 0
            return f"text only {format_count(char_count, 'char')}"

        text_count = sum(1 for item in content if is_text_content(item))
        if text_count == len(content):
            return format_count(len(content), "Text Block", "Text Blocks")

        return format_count(len(content), "Content Block", "Content Blocks")

    @staticmethod
    def _optional_string_attribute(result: "CallToolResult", name: str) -> str | None:
        value = getattr(result, name, None)
        return strip_str_to_none(value)

    @staticmethod
    def _optional_int_attribute(result: "CallToolResult", name: str) -> int | None:
        return positive_int_or_none(getattr(result, name, None))

    @staticmethod
    def _optional_nonnegative_int_attribute(
        result: "CallToolResult",
        name: str,
    ) -> int | None:
        value = getattr(result, name, None)
        if type(value) is not int or value < 0:
            return None
        return value

    @classmethod
    def _tool_result_display_metadata(
        cls,
        result: "CallToolResult",
    ) -> ToolResultDisplayMetadata:
        return ToolResultDisplayMetadata(
            read_text_file_path=cls._optional_string_attribute(result, "read_text_file_path"),
            read_text_file_line=cls._optional_int_attribute(result, "read_text_file_line"),
            read_text_file_limit=cls._optional_int_attribute(result, "read_text_file_limit"),
            transport_channel=cls._optional_string_attribute(result, "transport_channel"),
            output_line_count=cls._optional_nonnegative_int_attribute(
                result,
                "output_line_count",
            ),
        )

    def _tool_result_status(
        self,
        result: "CallToolResult",
        *,
        tool_name: str | None,
        metadata: ToolResultDisplayMetadata,
    ) -> str:
        if is_read_text_file_tool_name(tool_name) and not result.isError:
            return self._read_text_file_header_status(
                metadata.read_text_file_path,
                line_value=metadata.read_text_file_line,
                limit_value=metadata.read_text_file_limit,
            )

        return self._default_tool_result_status(result)

    @staticmethod
    def _transport_metadata_label(channel: str) -> str:
        return _TRANSPORT_METADATA_LABELS.get(channel, channel.upper())

    def _build_tool_result_bottom_metadata(
        self,
        *,
        result: "CallToolResult",
        metadata: ToolResultDisplayMetadata,
        timing_ms: float | None,
        has_structured: bool,
    ) -> list[str] | None:
        bottom_metadata_items: list[str] = []

        if metadata.transport_channel:
            bottom_metadata_items.append(self._transport_metadata_label(metadata.transport_channel))

        if timing_ms is not None:
            timing_seconds = timing_ms / 1000.0
            bottom_metadata_items.append(self._display._format_elapsed(timing_seconds))

        if has_structured:
            structured_label = "Structured ■"
            if self._has_structured_text_content_mismatch(result):
                structured_label += " (TextContent mismatch)"
            bottom_metadata_items.append(structured_label)

        return bottom_metadata_items or None

    @staticmethod
    def _has_structured_text_content_mismatch(result: "CallToolResult") -> bool:
        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        structured_content = getattr(result, "structuredContent", None)
        content = getattr(result, "content", None)
        if not (
            isinstance(structured_content, (dict, list))
            and isinstance(content, list)
            and len(content) > 1
            and all(is_text_content(item) for item in content)
        ):
            return False

        parsed_blocks: list[object] = []
        for item in content:
            text = get_text(item)
            if text is None:
                return False
            try:
                parsed_blocks.append(json.loads(text))
            except json.JSONDecodeError:
                return False

        if structured_content == parsed_blocks:
            return False

        if isinstance(structured_content, dict):
            return all(value != parsed_blocks for value in structured_content.values())

        return True

    def _prepare_read_text_file_result_display(
        self,
        *,
        result: "CallToolResult",
        metadata: ToolResultDisplayMetadata,
        tool_name: str | None,
        source_content,
        display_content,
        read_omitted_line_count: int,
    ) -> PreparedReadTextFileResultDisplay:
        if not is_read_text_file_tool_name(tool_name) or result.isError:
            return PreparedReadTextFileResultDisplay(display_content=display_content)

        render_markdown: bool | None = None
        source_line_count = self._read_text_file_line_count_from_content(source_content)
        no_lines_returned = source_line_count == 0 or not source_content
        read_no_lines_message: Text | None = None

        if no_lines_returned and read_omitted_line_count == 0:
            display_content = ""
            render_markdown = False
            read_no_lines_message = Text("(No lines returned)", style="dim italic")
        else:
            markdown_content = self._format_read_text_file_content_as_markdown(
                display_content,
                path_value=metadata.read_text_file_path,
            )
            if markdown_content is not None:
                display_content = markdown_content
                render_markdown = True

        read_more_lines_message = self._read_text_file_more_lines_message(read_omitted_line_count)
        read_additional_message = self._combine_additional_messages(
            read_more_lines_message,
            read_no_lines_message,
        )
        return PreparedReadTextFileResultDisplay(
            display_content=display_content,
            render_markdown=render_markdown,
            additional_message=read_additional_message,
        )

    def _render_tool_result_footer(
        self,
        *,
        highlight_color: str,
        bottom_metadata_items: list[str] | None,
    ) -> None:
        line = self._display.style.bottom_metadata_line(
            bottom_metadata_items,
            [],
            highlight_color,
            None,
            console.console.size.width,
        )
        if line is None:
            return

        console.console.print(line, markup=self._markup)
        console.console.print()

    def _render_skybridge_structured_content(
        self,
        *,
        structured_content: object,
        resource_uri: str | None,
    ) -> None:
        resource_label = f"app resource: {resource_uri}" if resource_uri else "app resource"
        resource_text = Text(resource_label, style="magenta")
        line = self._display.style.metadata_line(resource_text)
        console.console.print(line, markup=self._markup)
        console.console.print()

        json_str = json.dumps(structured_content, indent=2)
        syntax_obj = Syntax(
            json_str,
            "json",
            theme=self._display.code_style,
            background_color="default",
            word_wrap=self._display.code_word_wrap,
        )
        console.console.print(syntax_obj, markup=self._markup)

    def _render_structured_tool_result(
        self,
        *,
        result: "CallToolResult",
        name: str | None,
        display_content,
        truncate_content: bool,
        right_info: str,
        bottom_metadata_items: list[str] | None,
        structured_content: object,
        is_skybridge_tool: bool,
        skybridge_resource_uri: str | None,
        show_hook_indicator: bool,
        post_content: RenderableType | None = None,
    ) -> None:
        config_map = MESSAGE_CONFIGS[MessageType.TOOL_RESULT]
        block_color = "red" if result.isError else config_map["block_color"]
        arrow = config_map["arrow"]
        arrow_style = config_map["arrow_style"]

        left = self._display.build_header_left(
            block_color=block_color,
            arrow=arrow,
            arrow_style=arrow_style,
            name=name,
            is_error=result.isError,
            show_hook_indicator=show_hook_indicator,
        )

        self._display._create_combined_separator_status(left, right_info)
        self._display._display_content(
            display_content,
            truncate_content,
            result.isError,
            MessageType.TOOL_RESULT,
            check_markdown_markers=False,
        )
        if post_content:
            console.console.print()
            console.console.print(post_content, markup=self._markup)
        console.console.print()

        if is_skybridge_tool:
            self._render_skybridge_structured_content(
                structured_content=structured_content,
                resource_uri=skybridge_resource_uri,
            )
            return

        self._render_tool_result_footer(
            highlight_color=config_map["highlight_color"],
            bottom_metadata_items=bottom_metadata_items,
        )

    def show_tool_result(
        self,
        result: "CallToolResult",
        *,
        name: str | None = None,
        tool_name: str | None = None,
        skybridge_config: "SkybridgeServerConfig | None" = None,
        timing_ms: float | None = None,
        tool_call_id: str | None = None,
        type_label: str = "tool result",
        truncate_content: bool = True,
        show_hook_indicator: bool = False,
    ) -> None:
        """Display a tool result in the console."""
        logger = get_logger(__name__)
        if not self._display.show_tools_enabled:
            return
        if self._is_quiet_running_process_result(result, tool_name=tool_name):
            return

        try:
            metadata = self._tool_result_display_metadata(result)
            structured_content = getattr(result, "structuredContent", None)
            has_structured = structured_content is not None
            prepared_content = self._prepare_tool_result_content(
                content=result.content,
                structured_content=structured_content,
                tool_name=tool_name,
                truncate_content=truncate_content,
            )
            display_content = prepared_content.display_content
            source_content = prepared_content.source_content
            truncate_content = prepared_content.truncate_content

            skybridge_details = self._resolve_skybridge_result_details(
                has_structured=has_structured,
                tool_name=tool_name,
                skybridge_config=skybridge_config,
            )
            status = self._tool_result_status(result, tool_name=tool_name, metadata=metadata)
            bottom_metadata = self._build_tool_result_bottom_metadata(
                result=result,
                metadata=metadata,
                timing_ms=timing_ms,
                has_structured=has_structured,
            )
            display_type_label = type_label
            if is_read_text_file_tool_name(tool_name) and type_label == "tool result":
                display_type_label = "file read"
            right_info = self._build_tool_right_info(
                f"{display_type_label} - {status}",
                tool_call_id,
            )

            display_content, shell_exit_additional_message = (
                self._build_shell_exit_additional_message(
                    content=display_content,
                    source_content=source_content,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    output_line_count=metadata.output_line_count,
                )
            )

            read_text_display = self._prepare_read_text_file_result_display(
                result=result,
                metadata=metadata,
                tool_name=tool_name,
                source_content=source_content,
                display_content=display_content,
                read_omitted_line_count=prepared_content.read_omitted_line_count,
            )
            display_content = read_text_display.display_content
            additional_message = self._combine_additional_messages(
                shell_exit_additional_message,
                read_text_display.additional_message,
            )
            post_content: RenderableType | None = None
            image_content = get_tool_result_media_preview(result) or result.content
            if image_content:
                from fast_agent.ui.terminal_images import render_tool_result_images_for_settings

                post_content = render_tool_result_images_for_settings(
                    self._display.terminal_image_settings,
                    image_content,
                )

            if has_structured:
                self._render_structured_tool_result(
                    result=result,
                    name=name,
                    display_content=display_content,
                    truncate_content=truncate_content,
                    right_info=right_info,
                    bottom_metadata_items=bottom_metadata,
                    structured_content=structured_content,
                    is_skybridge_tool=skybridge_details.is_skybridge_tool,
                    skybridge_resource_uri=skybridge_details.resource_uri,
                    show_hook_indicator=show_hook_indicator,
                    post_content=post_content,
                )
            else:
                self._display.display_message(
                    content=display_content,
                    message_type=MessageType.TOOL_RESULT,
                    name=name,
                    right_info=right_info,
                    bottom_metadata=bottom_metadata,
                    is_error=result.isError,
                    truncate_content=truncate_content,
                    additional_message=additional_message,
                    post_content=post_content,
                    render_markdown=read_text_display.render_markdown,
                    show_hook_indicator=show_hook_indicator,
                )
        except Exception:
            logger.exception(
                "Tool result display failed",
                tool_name=tool_name,
                agent_name=name,
                is_error=result.isError,
            )

    def _shell_tool_call_content(
        self,
        *,
        command: object,
        tool_args: dict[str, Any],
        metadata: dict[str, Any],
    ) -> object:
        command_text = Text()
        if not isinstance(command, str) or not command:
            command_text.append("$ ", style="magenta")
            command_text.append("(no shell command provided)", style="dim")
            return command_text

        preview = build_apply_patch_preview(
            command,
            max_lines=self._display.apply_patch_preview_max_lines,
        )
        if preview is not None:
            command_text.append("$ ", style="magenta")
            command_text.append("apply_patch (preview)", style="white")
            command_text.append("\n")
            command_text.append_text(
                style_apply_patch_preview_text(
                    format_apply_patch_preview(
                        preview,
                        other_args=extract_non_command_args(tool_args),
                    ),
                    default_style="dim",
                )
            )
            return command_text

        shell_language = shell_syntax_language(
            metadata.get("shell_name"),
            shell_path=cast("str | None", metadata.get("shell_path")),
        )
        return Syntax(
            command.rstrip(),
            shell_language,
            theme=self._display.code_style,
            line_numbers=False,
            word_wrap=self._display.code_word_wrap,
        )

    @staticmethod
    def _shell_tool_call_bottom_items(metadata: dict[str, Any]) -> list[str]:
        bottom_items: list[str] = []
        shell_path = metadata.get("shell_path")
        if shell_path:
            bottom_items.append(str(shell_path))

        working_dir_display = metadata.get("working_dir_display") or metadata.get("working_dir")
        if working_dir_display:
            bottom_items.append(f"cwd: {working_dir_display}")

        idle_yield_seconds = metadata.get("idle_yield_seconds")
        foreground_yield_seconds = metadata.get("foreground_yield_seconds")
        if idle_yield_seconds and foreground_yield_seconds:
            bottom_items.append(
                f"yield: {idle_yield_seconds}s idle / {foreground_yield_seconds}s total"
            )
        return bottom_items

    def _shell_tool_call_right_info(
        self,
        metadata: dict[str, Any],
        tool_call_id: str | None,
    ) -> str:
        shell_name = metadata.get("shell_name") or "shell"
        shell_path = metadata.get("shell_path")
        right_parts: list[str] = []
        if shell_path and shell_path != shell_name:
            right_parts.append(f"{shell_name} ({shell_path})")
        elif shell_name:
            right_parts.append(shell_name)

        if metadata.get("background"):
            right_parts.append("background")
        else:
            idle_yield_seconds = metadata.get("idle_yield_seconds")
            foreground_yield_seconds = metadata.get("foreground_yield_seconds")
            if idle_yield_seconds and foreground_yield_seconds:
                right_parts.append(
                    f"yield {idle_yield_seconds}s idle / {foreground_yield_seconds}s total"
                )
            elif idle_yield_seconds:
                right_parts.append(f"idle yield {idle_yield_seconds}s")
            elif timeout_seconds := metadata.get("timeout_seconds"):
                right_parts.append(f"timeout {timeout_seconds}s")

        base_label = " | ".join(right_parts) if right_parts else None
        return self._build_tool_right_info(base_label, tool_call_id)

    def _prepare_shell_tool_call_display(
        self,
        *,
        tool_args: dict[str, Any],
        metadata: dict[str, Any],
        tool_call_id: str | None,
    ) -> PreparedToolCallDisplay:
        command = metadata.get("command") or tool_args.get("command")
        return PreparedToolCallDisplay(
            content=self._shell_tool_call_content(
                command=command,
                tool_args=tool_args,
                metadata=metadata,
            ),
            right_info=self._shell_tool_call_right_info(metadata, tool_call_id),
            bottom_items=self._shell_tool_call_bottom_items(metadata),
            highlight_indexes=[],
            max_item_length=50,
            truncate_content=False,
            render_markdown=False,
        )

    def _prepare_shell_process_tool_call_display(
        self,
        *,
        metadata: dict[str, Any],
        tool_call_id: str | None,
    ) -> PreparedToolCallDisplay:
        action = str(metadata.get("action") or "process")
        process_id = str(metadata.get("process_id") or "process")
        content = Text()
        content.append(action, style="bold")
        content.append(" ")
        content.append(process_id, style="cyan")

        right_label = action
        if action == "poll":
            wait_sec = metadata.get("wait_sec")
            if type(wait_sec) is int and wait_sec > 0:
                right_label = f"wait up to {wait_sec}s"
            else:
                right_label = "non-blocking"

        return PreparedToolCallDisplay(
            content=content,
            right_info=self._build_tool_right_info(right_label, tool_call_id),
            bottom_items=None,
            highlight_indexes=[],
            max_item_length=None,
            truncate_content=False,
            render_markdown=False,
        )

    def _apply_patch_tool_call_content(self, tool_args: dict[str, Any]) -> Text:
        patch_input = extract_apply_patch_input(tool_args)
        preview = (
            build_apply_patch_preview_from_input(
                patch_input,
                max_lines=self._display.apply_patch_preview_max_lines,
            )
            if patch_input is not None
            else None
        )

        patch_text = Text()
        if preview is not None:
            patch_text.append("apply_patch (preview)", style="white")
            patch_text.append("\n")
            patch_text.append_text(
                style_apply_patch_preview_text(
                    format_apply_patch_preview(
                        preview,
                        other_args={
                            key: value for key, value in tool_args.items() if key != "input"
                        },
                    ),
                    default_style="dim",
                )
            )
        elif patch_input is not None:
            patch_text.append(patch_input, style="white")
        else:
            patch_text.append("(no apply_patch input provided)", style="dim")
        return patch_text

    def _prepare_tool_call_display(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        bottom_items: list[str] | None,
        highlight_indexes: list[int],
        max_item_length: int | None,
        metadata: dict[str, Any],
        tool_call_id: str | None,
        type_label: str,
    ) -> PreparedToolCallDisplay:
        display_tool_name = self._display_tool_name(tool_name)
        normalized_bottom_items = self._normalize_tool_footer_items(
            bottom_items,
            display_tool_name=display_tool_name,
        )
        right_info = self._build_tool_right_info(
            f"{type_label} - {display_tool_name}",
            tool_call_id,
        )

        if metadata.get("variant") == "shell":
            return self._prepare_shell_tool_call_display(
                tool_args=tool_args,
                metadata=metadata,
                tool_call_id=tool_call_id,
            )

        if metadata.get("variant") == "shell_process":
            return self._prepare_shell_process_tool_call_display(
                metadata=metadata,
                tool_call_id=tool_call_id,
            )

        if metadata.get("variant") == "code":
            content, footer_items = self._build_code_tool_call_syntax(tool_args, metadata)
            return PreparedToolCallDisplay(
                content=content,
                right_info=right_info,
                bottom_items=[*(normalized_bottom_items or []), *footer_items],
                highlight_indexes=highlight_indexes,
                max_item_length=max(max_item_length or 0, 50) or None,
                truncate_content=False,
                render_markdown=False,
            )

        if is_apply_patch_tool_name(tool_name):
            return PreparedToolCallDisplay(
                content=self._apply_patch_tool_call_content(tool_args),
                right_info=right_info,
                bottom_items=normalized_bottom_items,
                highlight_indexes=highlight_indexes,
                max_item_length=max_item_length,
                truncate_content=False,
                render_markdown=False,
            )

        content: object = tool_args
        truncate_content = True
        if is_read_text_file_tool_name(tool_name):
            read_summary = self._read_text_file_summary(tool_args)
            if read_summary:
                content = Text(read_summary, style="dim")
                truncate_content = False

        return PreparedToolCallDisplay(
            content=content,
            right_info=right_info,
            bottom_items=normalized_bottom_items,
            highlight_indexes=highlight_indexes,
            max_item_length=max_item_length,
            truncate_content=truncate_content,
        )

    def show_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None,
        *,
        bottom_items: list[str] | None = None,
        highlight_indexes: list[int] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        type_label: str = "tool call",
        show_hook_indicator: bool = False,
    ) -> None:
        """Display a tool call header and body."""
        logger = get_logger(__name__)
        if not self._display.show_tools_enabled:
            return

        try:
            tool_args = tool_args or {}
            metadata = metadata or {}
            if (
                metadata.get("variant") == "shell_process"
                and metadata.get("action") == "poll"
            ):
                if self._display.logger_settings.progress_display:
                    return
                elapsed = metadata.get("elapsed_seconds")
                wait_sec = metadata.get("wait_sec")
                from fast_agent.ui.progress_display import progress_display

                self._display.show_managed_process_poll(
                    name=None if progress_display.is_default_agent_name(name) else name,
                    process_id=str(metadata.get("process_id") or "process"),
                    command=(
                        command
                        if isinstance(
                            command := metadata.get("command_summary"),
                            str,
                        )
                        else None
                    ),
                    elapsed_seconds=(
                        float(elapsed)
                        if isinstance(elapsed, (int, float))
                        and not isinstance(elapsed, bool)
                        else None
                    ),
                    wait_sec=wait_sec if type(wait_sec) is int else None,
                    has_observed_output=(
                        observed if isinstance(
                            observed := metadata.get("has_observed_output"), bool
                        )
                        else None
                    ),
                    seconds_since_last_output=(
                        float(since_output)
                        if isinstance(
                            since_output := metadata.get("seconds_since_last_output"),
                            (int, float),
                        )
                        and not isinstance(since_output, bool)
                        else None
                    ),
                    tool_call_id=tool_call_id,
                )
                return
            pre_content: Text | None = None
            prepared = self._prepare_tool_call_display(
                tool_name=tool_name,
                tool_args=tool_args,
                bottom_items=bottom_items,
                highlight_indexes=highlight_indexes or [],
                max_item_length=max_item_length,
                metadata=metadata,
                tool_call_id=tool_call_id,
                type_label=type_label,
            )

            self._display.display_message(
                content=prepared.content,
                message_type=MessageType.TOOL_CALL,
                name=name,
                pre_content=pre_content,
                right_info=prepared.right_info,
                bottom_metadata=prepared.bottom_items,
                highlight_indexes=prepared.highlight_indexes,
                max_item_length=prepared.max_item_length,
                truncate_content=prepared.truncate_content,
                render_markdown=prepared.render_markdown,
                show_hook_indicator=show_hook_indicator,
            )
        except Exception:
            logger.exception(
                "Tool call display failed",
                tool_name=tool_name,
                agent_name=name,
            )

    async def show_tool_update(self, updated_server: str, *, agent_name: str | None = None) -> None:
        """Show a background tool update notification."""
        if not self._display.show_tools_enabled:
            return

        try:
            from prompt_toolkit.application.current import get_app

            app = get_app()
            from fast_agent.ui import notification_tracker

            notification_tracker.add_tool_update(updated_server)
            app.invalidate()
        except Exception:
            if agent_name:
                left = (
                    "[magenta]▎[/magenta][dim magenta]▶[/dim magenta] "
                    f"[magenta]{escape_markup(agent_name)}[/magenta]"
                )
            else:
                left = "[magenta]▎[/magenta][dim magenta]▶[/dim magenta]"

            right = f"[dim]{escape_markup(updated_server)}[/dim]"
            self._display._create_combined_separator_status(left, right)

            message = f"Updating tools for server {escape_markup(updated_server)}"
            console.console.print(message, style="dim", markup=self._markup)

            console.console.print()
            line = self._display.style.tool_update_line()
            console.console.print(line, markup=self._markup)
            console.console.print()

    @staticmethod
    def _has_skybridge_signal(config: "SkybridgeServerConfig", resources: list[Any]) -> bool:
        return bool(config.enabled or resources or config.tools or config.warnings)

    @staticmethod
    def _skybridge_resource_counts(resources: list[Any]) -> dict[str, int]:
        return {
            "valid_resource_count": sum(
                1 for resource in resources if resource.is_valid_app_resource
            ),
            "mcp_app_resource_count": sum(1 for resource in resources if resource.is_mcp_app),
            "skybridge_resource_count": sum(1 for resource in resources if resource.is_skybridge),
        }

    @staticmethod
    def _active_skybridge_tools(config: "SkybridgeServerConfig") -> list[dict[str, Any]]:
        return [
            {
                "name": tool.display_name,
                "template": str(tool.template_uri) if tool.template_uri else None,
                "kind": tool.kind,
            }
            for tool in config.tools
            if tool.is_valid
        ]

    @staticmethod
    def _skybridge_tool_counts(config: "SkybridgeServerConfig") -> dict[str, int]:
        return {
            "mcp_app_tool_count": sum(
                1 for tool in config.tools if tool.is_valid and tool.kind == "mcp_app"
            ),
            "skybridge_tool_count": sum(
                1 for tool in config.tools if tool.is_valid and tool.kind == "skybridge"
            ),
        }

    @classmethod
    def _skybridge_server_row(
        cls,
        server_name: str,
        config: "SkybridgeServerConfig",
        resources: list[Any],
    ) -> dict[str, Any]:
        return {
            "server_name": server_name,
            "config": config,
            "resources": resources,
            **cls._skybridge_resource_counts(resources),
            **cls._skybridge_tool_counts(config),
            "total_resource_count": len(resources),
            "active_tools": cls._active_skybridge_tools(config),
            "enabled": config.enabled,
        }

    @staticmethod
    def _add_skybridge_warning(
        *,
        warnings: list[str],
        warning_seen: set[str],
        server_name: str,
        warning: str,
    ) -> None:
        message = warning.strip()
        if not message:
            return
        if not message.startswith(server_name):
            message = f"{server_name} {message}"
        if message not in warning_seen:
            warnings.append(message)
            warning_seen.add(message)

    @classmethod
    def summarize_skybridge_configs(
        cls,
        configs: Mapping[str, "SkybridgeServerConfig"] | None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Convert Skybridge configs into display-ready structures."""
        server_rows: list[dict[str, Any]] = []
        warnings: list[str] = []
        warning_seen: set[str] = set()

        if not configs:
            return server_rows, warnings

        for server_name in sorted(configs.keys()):
            config = configs.get(server_name)
            if not config:
                continue
            resources = list(config.ui_resources or [])
            if not cls._has_skybridge_signal(config, resources):
                continue

            server_rows.append(cls._skybridge_server_row(server_name, config, resources))

            for warning in config.warnings:
                cls._add_skybridge_warning(
                    warnings=warnings,
                    warning_seen=warning_seen,
                    server_name=server_name,
                    warning=warning,
                )

        return server_rows, warnings

    def show_skybridge_summary(
        self,
        agent_name: str,
        configs: Mapping[str, "SkybridgeServerConfig"] | None,
    ) -> None:
        """Display aggregated Skybridge status."""
        del agent_name
        server_rows, warnings = self.summarize_skybridge_configs(configs)

        if not server_rows and not warnings:
            return

        heading = "[dim]Interactive MCP app integrations detected:[/dim]"
        console.console.print()
        console.console.print(heading, markup=self._markup)

        if not server_rows:
            console.console.print("[dim]  ● none detected[/dim]", markup=self._markup)
        else:
            for row in server_rows:
                server_name = escape_markup(row["server_name"])
                tool_infos = row["active_tools"]
                enabled = row["enabled"]

                segments: list[str] = []
                if row["mcp_app_tool_count"] or row["mcp_app_resource_count"]:
                    segments.append(
                        "[cyan]MCP Apps[/cyan][dim]: "
                        f"{format_count(row['mcp_app_tool_count'], 'tool')}, "
                        f"{format_count(row['mcp_app_resource_count'], 'resource')}[/dim]"
                    )
                if row["skybridge_tool_count"] or row["skybridge_resource_count"]:
                    segments.append(
                        "[cyan]OpenAI Apps SDK[/cyan][dim]: "
                        f"{format_count(row['skybridge_tool_count'], 'tool')}, "
                        f"{format_count(row['skybridge_resource_count'], 'resource')}[/dim]"
                    )
                integration_segment = (
                    "[dim]; [/dim]".join(segments)
                    if segments
                    else "[dim]no active app integrations[/dim]"
                )
                name_style = "cyan" if enabled else "yellow"
                status_suffix = "" if enabled else "[dim] (issues detected)[/dim]"

                console.console.print(
                    f"[dim]  ● [/dim][{name_style}]{server_name}[/{name_style}]{status_suffix}"
                    f"[dim] — [/dim]{integration_segment}",
                    markup=self._markup,
                )

                if tool_infos:
                    for tool in tool_infos:
                        template_info = (
                            f" [dim]({escape_markup(tool['template'])})[/dim]"
                            if tool["template"]
                            else ""
                        )
                        console.console.print(
                            f"[dim]     · [/dim]{escape_markup(tool['name'])}{template_info}",
                            markup=self._markup,
                        )
                else:
                    console.console.print("[dim]     · no active tools[/dim]", markup=self._markup)

        if warnings:
            console.console.print()
            console.console.print(
                "[yellow]Warnings[/yellow]",
                markup=self._markup,
            )
            for warning in warnings:
                console.console.print(Text(f"- {warning}", style="yellow"))
