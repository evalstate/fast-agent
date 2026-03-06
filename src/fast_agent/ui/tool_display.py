from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, cast

from rich.syntax import Syntax
from rich.text import Text

from fast_agent.core.logging.logger import get_logger
from fast_agent.ui import console
from fast_agent.ui.apply_patch_preview import (
    build_apply_patch_preview,
    extract_non_command_args,
    format_apply_patch_preview,
    is_shell_execution_tool,
    style_apply_patch_preview_text,
)
from fast_agent.ui.message_primitives import MESSAGE_CONFIGS, MessageType
from fast_agent.ui.shell_output_truncation import (
    format_shell_output_line_count,
    truncate_shell_output_lines,
)

if TYPE_CHECKING:
    from mcp.types import CallToolResult

    from fast_agent.mcp.skybridge import SkybridgeServerConfig
    from fast_agent.ui.console_display import ConsoleDisplay


class ToolDisplay:
    """Encapsulates rendering logic for tool calls and results."""

    _TOOL_CALL_ID_MAX_LENGTH = 12
    _TOOL_CALL_ID_PREFIX_LENGTH = 5
    _TOOL_CALL_ID_SUFFIX_LENGTH = 6
    _TOOL_CALL_ID_ELLIPSIS = "…"
    _PATH_ELLIPSIS = "…"
    _READ_TEXT_FILE_LANGUAGE_BY_EXTENSION: dict[str, str] = {
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

    @staticmethod
    def _normalize_tool_name(tool_name: str) -> str:
        normalized = tool_name.lower()
        for sep in ("/", ".", ":"):
            if sep in normalized:
                normalized = normalized.rsplit(sep, 1)[-1]
        return normalized

    @classmethod
    def _is_read_text_file_tool(cls, tool_name: str | None) -> bool:
        if not tool_name:
            return False
        normalized = cls._normalize_tool_name(tool_name)
        if normalized == "read_text_file":
            return True
        return tool_name.lower().endswith("__read_text_file")

    @classmethod
    def _left_truncate_with_ellipsis(cls, text: str, max_length: int) -> str:
        if max_length <= 0:
            return ""
        if len(text) <= max_length:
            return text
        if max_length == 1:
            return cls._PATH_ELLIPSIS
        return f"{cls._PATH_ELLIPSIS}{text[-(max_length - 1) :]}"

    @staticmethod
    def _format_parent_current_path(path_text: str) -> str:
        normalized = os.path.normpath(path_text)
        path = Path(normalized)
        current = path.name or normalized
        parent = path.parent.name
        if parent:
            return f"{parent}/{current}"
        return current

    @classmethod
    def _fit_path_for_display(cls, path_text: str, max_length: int) -> str:
        if max_length <= 0:
            return ""

        compact = cls._format_parent_current_path(path_text)
        if len(compact) <= max_length:
            return compact

        current = Path(path_text).name or path_text
        if len(current) <= max_length:
            return current

        return cls._left_truncate_with_ellipsis(current, max_length)

    def _read_text_file_summary(self, tool_args: Mapping[str, Any]) -> str | None:
        path_value = tool_args.get("path")
        if not isinstance(path_value, str):
            return None

        stripped_path = path_value.strip()
        if not stripped_path:
            return None

        line_value = tool_args.get("line")
        limit_value = tool_args.get("limit")
        line = line_value if isinstance(line_value, int) and line_value >= 1 else None
        limit = limit_value if isinstance(limit_value, int) and limit_value >= 1 else None

        offset_suffix = f" (offset {line})." if line is not None else "."
        if limit is not None:
            line_noun = "line" if limit == 1 else "lines"
            prefix = f"The assistant is reading {limit} {line_noun} from "
        elif line is not None:
            prefix = "The assistant is reading from "
        else:
            prefix = "The assistant is reading a file from "

        max_width = max(16, console.console.size.width - len(prefix) - len(offset_suffix))
        display_path = self._fit_path_for_display(stripped_path, max_width)
        return f"{prefix}{display_path}{offset_suffix}"

    def _configured_output_line_limit(self) -> int | None:
        config = self._display.config
        if not config:
            return None
        shell_config = getattr(config, "shell_execution", None)
        if not shell_config:
            return None
        return getattr(shell_config, "output_display_lines", None)

    @staticmethod
    def _display_tool_name(tool_name: str) -> str:
        if tool_name.startswith("agent__"):
            return tool_name[7:]
        return tool_name

    @classmethod
    def _format_tool_call_id(cls, tool_call_id: str | None) -> str | None:
        if not tool_call_id:
            return None
        if len(tool_call_id) <= cls._TOOL_CALL_ID_MAX_LENGTH:
            return tool_call_id
        return (
            f"{tool_call_id[: cls._TOOL_CALL_ID_PREFIX_LENGTH]}"
            f"{cls._TOOL_CALL_ID_ELLIPSIS}"
            f"{tool_call_id[-cls._TOOL_CALL_ID_SUFFIX_LENGTH :]}"
        )

    @classmethod
    def _build_tool_right_info(cls, base_label: str | None, tool_call_id: str | None) -> str:
        parts: list[str] = []
        if base_label:
            parts.append(base_label)

        short_id = cls._format_tool_call_id(tool_call_id)
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
        if not self._is_read_text_file_tool(tool_name):
            return None
        return self._configured_output_line_limit()

    def _shell_show_bash(self, tool_name: str | None) -> bool:
        if not is_shell_execution_tool(tool_name):
            return True
        config = self._display.config
        if not config:
            return True
        shell_config = getattr(config, "shell_execution", None)
        if not shell_config:
            return True
        return bool(getattr(shell_config, "show_bash", True))

    @staticmethod
    def _extract_exit_code_line(lines: list[str]) -> tuple[list[str], str | None]:
        if not lines:
            return lines, None
        index = len(lines) - 1
        while index >= 0 and not lines[index].strip():
            index -= 1
        if index < 0:
            return lines, None
        candidate = lines[index].strip()
        if candidate.startswith("[Exit code:") or candidate.startswith("process exit code was"):
            return lines[:index], candidate
        return lines, None

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

        formatted_id = self._format_tool_call_id(tool_call_id)
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
        lines_without_exit, _ = ToolDisplay._extract_exit_code_line(lines)
        return len(lines_without_exit)

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
        normalized_tool_name = self._normalize_tool_name(tool_name)
        if normalized_tool_name not in {"execute", "bash", "shell"}:
            return content, None

        if not content or len(content) != 1 or not is_text_content(content[0]):
            return content, None

        text = get_text(content[0]) or ""
        lines = cast("list[str]", text.splitlines())
        lines_without_exit, exit_line = self._extract_exit_code_line(lines)
        exit_code = self._parse_exit_code_value(exit_line)
        if exit_code is None:
            return content, None

        line_count = output_line_count
        if line_count is None:
            line_count = self._shell_output_line_count_from_content(source_content)
        if line_count is None:
            line_count = len(lines_without_exit)

        no_output = not any(line.strip() for line in lines_without_exit)
        detail = self._shell_exit_detail(
            no_output=no_output,
            tool_call_id=tool_call_id,
            output_line_count=line_count,
        )
        additional_message = self._display.style.shell_exit_line(
            exit_code,
            console.console.size.width,
            detail,
        )

        if not lines_without_exit:
            return "", additional_message

        rendered_text = "\n".join(lines_without_exit)
        return [TextContent(type="text", text=rendered_text)], additional_message

    def _limit_shell_output_text(self, text: str, line_limit: int) -> str:
        if line_limit < 0:
            return text
        lines = text.splitlines()
        if not lines:
            return text if line_limit != 0 else ""
        if line_limit == 0:
            _, exit_line = self._extract_exit_code_line(lines)
            return exit_line or ""

        lines_without_exit, exit_line = self._extract_exit_code_line(lines)
        if line_limit >= len(lines_without_exit):
            return text

        output_lines, _ = truncate_shell_output_lines(lines_without_exit, line_limit)
        if exit_line:
            output_lines.append(exit_line)
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

    def _limit_read_text_output_text(self, text: str, line_limit: int) -> tuple[str, int]:
        if line_limit < 0:
            return text, 0

        lines = text.splitlines()
        if line_limit == 0:
            if not lines:
                return text, 0
            return "", len(lines)

        if len(lines) <= line_limit + 2:
            return text, 0

        start_index = 0
        while start_index < len(lines) and not lines[start_index].strip():
            start_index += 1
        if start_index >= len(lines):
            start_index = 0

        visible_lines = lines[start_index : start_index + line_limit]
        omitted_line_count = len(lines) - len(visible_lines)
        return "\n".join(visible_lines), omitted_line_count

    def _limit_read_text_output_content(self, content, line_limit: int):
        from mcp.types import TextContent

        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        if not content or len(content) != 1 or not is_text_content(content[0]):
            return content, 0
        text = get_text(content[0]) or ""
        limited, omitted_line_count = self._limit_read_text_output_text(text, line_limit)
        if limited == text and omitted_line_count == 0:
            return content, 0
        return [TextContent(type="text", text=limited)], omitted_line_count

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
        suffix = Path(path_value).suffix.lower()
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
            base_status = self._fit_path_for_display(stripped, 42) if stripped else "preview"
        else:
            base_status = "preview"

        line = (
            line_value
            if isinstance(line_value, int) and not isinstance(line_value, bool) and line_value >= 1
            else None
        )
        limit = (
            limit_value
            if isinstance(limit_value, int) and not isinstance(limit_value, bool) and limit_value >= 1
            else None
        )

        details: list[str] = []
        if line is not None:
            details.append(f"offset {line}")
        if limit is not None:
            line_noun = "line" if limit == 1 else "lines"
            details.append(f"{limit} {line_noun}")

        if not details:
            return base_status
        return f"{base_status} ({', '.join(details)})"

    @staticmethod
    def _read_text_file_more_lines_message(omitted_line_count: int) -> Text | None:
        if omitted_line_count <= 0:
            return None
        noun = "line" if omitted_line_count == 1 else "lines"
        return Text(f"(+{omitted_line_count} more {noun})", style="dim italic")

    @staticmethod
    def _read_text_file_no_lines_message() -> Text:
        return Text("(No lines returned)", style="dim italic")

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
        first: Text | None,
        second: Text | None,
    ) -> Text | None:
        if first is None:
            return second
        if second is None:
            return first
        return Text.assemble(first, "\n", second)

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
        config = self._display.config
        if config and not config.logger.show_tools:
            return

        try:
            from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

            style = self._display.style

            content = result.content
            structured_content = getattr(result, "structuredContent", None)
            has_structured = structured_content is not None
            source_content = content
            display_content = content
            read_omitted_line_count = 0
            if truncate_content:
                show_bash_output = self._shell_show_bash(tool_name)
                if not show_bash_output:
                    display_content = self._limit_shell_output_content(content, 0)
                    truncate_content = False
                else:
                    line_limit = self._shell_output_line_limit(tool_name)
                    if line_limit is not None:
                        display_content = self._limit_shell_output_content(content, line_limit)
                        truncate_content = False
                    else:
                        read_line_limit = self._read_text_file_output_line_limit(tool_name)
                        if read_line_limit is not None:
                            display_content, read_omitted_line_count = (
                                self._limit_read_text_output_content(
                                content,
                                read_line_limit,
                            )
                            )
                            truncate_content = False

            is_skybridge_tool = False
            skybridge_resource_uri: str | None = None
            if has_structured and tool_name and skybridge_config:
                for tool_cfg in skybridge_config.tools:
                    if tool_cfg.tool_name == tool_name and tool_cfg.is_valid:
                        is_skybridge_tool = True
                        skybridge_resource_uri = (
                            str(tool_cfg.resource_uri)
                            if tool_cfg.resource_uri is not None
                            else None
                        )
                        break

            if result.isError:
                status = "ERROR"
            else:
                if not content:
                    status = "No Content"
                elif len(content) == 1 and is_text_content(content[0]):
                    text_content = get_text(content[0])
                    char_count = len(text_content) if text_content else 0
                    status = f"text only {char_count} chars"
                else:
                    text_count = sum(1 for item in content if is_text_content(item))
                    if text_count == len(content):
                        status = (
                            f"{len(content)} Text Blocks" if len(content) > 1 else "1 Text Block"
                        )
                    else:
                        status = (
                            f"{len(content)} Content Blocks"
                            if len(content) > 1
                            else "1 Content Block"
                        )

            if self._is_read_text_file_tool(tool_name) and not result.isError:
                status = self._read_text_file_header_status(
                    getattr(result, "read_text_file_path", None),
                    line_value=getattr(result, "read_text_file_line", None),
                    limit_value=getattr(result, "read_text_file_limit", None),
                )

            channel = getattr(result, "transport_channel", None)
            bottom_metadata_items: list[str] = []
            if channel:
                if channel == "post-json":
                    transport_info = "HTTP (JSON-RPC)"
                elif channel == "post-sse":
                    transport_info = "HTTP (SSE)"
                elif channel == "get":
                    transport_info = "Legacy SSE"
                elif channel == "resumption":
                    transport_info = "Resumption"
                elif channel == "stdio":
                    transport_info = "STDIO"
                else:
                    transport_info = channel.upper()

                bottom_metadata_items.append(transport_info)

            # Use timing from FAST_AGENT_TOOL_TIMING (passed as parameter)
            if timing_ms is not None:
                # Convert ms to seconds for display
                timing_seconds = timing_ms / 1000.0
                bottom_metadata_items.append(self._display._format_elapsed(timing_seconds))

            if has_structured:
                bottom_metadata_items.append("Structured ■")

            bottom_metadata = bottom_metadata_items or None
            display_type_label = type_label
            if self._is_read_text_file_tool(tool_name) and type_label == "tool result":
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
                    output_line_count=getattr(result, "output_line_count", None),
                )
            )

            render_markdown: bool | None = None
            read_more_lines_message: Text | None = None
            read_no_lines_message: Text | None = None
            if self._is_read_text_file_tool(tool_name) and not result.isError:
                source_line_count = self._read_text_file_line_count_from_content(source_content)
                no_lines_returned = source_line_count == 0 or not source_content

                if no_lines_returned and read_omitted_line_count == 0:
                    display_content = ""
                    render_markdown = False
                    read_no_lines_message = self._read_text_file_no_lines_message()
                else:
                    markdown_content = self._format_read_text_file_content_as_markdown(
                        display_content,
                        path_value=getattr(result, "read_text_file_path", None),
                    )
                    if markdown_content is not None:
                        display_content = markdown_content
                        render_markdown = True
                read_more_lines_message = self._read_text_file_more_lines_message(
                    read_omitted_line_count
                )

            additional_message = self._combine_additional_messages(
                shell_exit_additional_message,
                read_more_lines_message,
            )
            additional_message = self._combine_additional_messages(
                additional_message,
                read_no_lines_message,
            )

            if has_structured:
                config_map = MESSAGE_CONFIGS[MessageType.TOOL_RESULT]
                block_color = "red" if result.isError else config_map["block_color"]
                arrow = config_map["arrow"]
                arrow_style = config_map["arrow_style"]

                # Use build_header_left for consistency with hook indicator
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
                console.console.print()
                total_width = console.console.size.width

                if is_skybridge_tool:
                    resource_label = (
                        f"skybridge resource: {skybridge_resource_uri}"
                        if skybridge_resource_uri
                        else "skybridge resource"
                    )
                    resource_text = Text(resource_label, style="magenta")
                    line = style.metadata_line(resource_text, total_width)
                    console.console.print(line, markup=self._markup)
                    console.console.print()

                    json_str = json.dumps(structured_content, indent=2)
                    syntax_obj = Syntax(
                        json_str,
                        "json",
                        theme=self._display.code_style,
                        background_color="default",
                    )
                    console.console.print(syntax_obj, markup=self._markup)
                else:
                    line = style.bottom_metadata_line(
                        bottom_metadata_items,
                        None,
                        config_map["highlight_color"],
                        None,
                        total_width,
                    )
                    if line is not None:
                        console.console.print(line, markup=self._markup)
                        console.console.print()
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
                    render_markdown=render_markdown,
                    show_hook_indicator=show_hook_indicator,
                )
        except Exception:
            logger.exception(
                "Tool result display failed",
                tool_name=tool_name,
                agent_name=name,
                is_error=result.isError,
            )

    def show_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None,
        *,
        bottom_items: list[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        type_label: str = "tool call",
        show_hook_indicator: bool = False,
    ) -> None:
        """Display a tool call header and body."""
        logger = get_logger(__name__)
        config = self._display.config
        if config and not config.logger.show_tools:
            return

        try:
            tool_args = tool_args or {}
            metadata = metadata or {}

            display_tool_name = self._display_tool_name(tool_name)
            right_info = self._build_tool_right_info(
                f"{type_label} - {display_tool_name}",
                tool_call_id,
            )
            content: Any = tool_args
            pre_content: Text | None = None
            truncate_content = True
            render_markdown: bool | None = None

            if metadata.get("variant") == "shell":
                bottom_items = list()
                highlight_index = None
                max_item_length = 50
                command = metadata.get("command") or tool_args.get("command")

                command_text = Text()
                if command and isinstance(command, str):
                    preview = build_apply_patch_preview(command)
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
                                default_style="white",
                            )
                        )
                    else:
                        command_text.append("$ ", style="magenta")
                        command_text.append(command, style="white")
                else:
                    command_text.append("$ ", style="magenta")
                    command_text.append("(no shell command provided)", style="dim")

                content = command_text
                render_markdown = False

                shell_name = metadata.get("shell_name") or "shell"
                shell_path = metadata.get("shell_path")
                if shell_path:
                    bottom_items.append(str(shell_path))

                right_parts: list[str] = []
                if shell_path and shell_path != shell_name:
                    right_parts.append(f"{shell_name} ({shell_path})")
                elif shell_name:
                    right_parts.append(shell_name)

                base_label = " | ".join(right_parts) if right_parts else None
                right_info = self._build_tool_right_info(base_label, tool_call_id)
                truncate_content = False

                working_dir_display = metadata.get("working_dir_display") or metadata.get(
                    "working_dir"
                )
                if working_dir_display:
                    bottom_items.append(f"cwd: {working_dir_display}")

                timeout_seconds = metadata.get("timeout_seconds")
                warning_interval = metadata.get("warning_interval_seconds")

                if timeout_seconds and warning_interval:
                    bottom_items.append(
                        f"timeout: {timeout_seconds}s, warning every {warning_interval}s"
                    )
            elif self._is_read_text_file_tool(tool_name):
                read_summary = self._read_text_file_summary(tool_args)
                if read_summary:
                    content = Text(read_summary, style="dim")
                    truncate_content = False

            self._display.display_message(
                content=content,
                message_type=MessageType.TOOL_CALL,
                name=name,
                pre_content=pre_content,
                right_info=right_info,
                bottom_metadata=bottom_items,
                highlight_index=highlight_index,
                max_item_length=max_item_length,
                truncate_content=truncate_content,
                render_markdown=render_markdown,
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
        config = self._display.config
        if config and not config.logger.show_tools:
            return

        try:
            from prompt_toolkit.application.current import get_app

            app = get_app()
            from fast_agent.ui import notification_tracker

            notification_tracker.add_tool_update(updated_server)
            app.invalidate()
        except Exception:  # noqa: BLE001
            if agent_name:
                left = f"[magenta]▎[/magenta][dim magenta]▶[/dim magenta] [magenta]{agent_name}[/magenta]"
            else:
                left = "[magenta]▎[/magenta][dim magenta]▶[/dim magenta]"

            right = f"[dim]{updated_server}[/dim]"
            self._display._create_combined_separator_status(left, right)

            message = f"Updating tools for server {updated_server}"
            console.console.print(message, style="dim", markup=self._markup)

            console.console.print()
            line = self._display.style.tool_update_line(console.console.size.width)
            console.console.print(line, markup=self._markup)
            console.console.print()

    @staticmethod
    def summarize_skybridge_configs(
        configs: Mapping[str, "SkybridgeServerConfig"] | None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Convert Skybridge configs into display-ready structures."""
        server_rows: list[dict[str, Any]] = []
        warnings: list[str] = []
        warning_seen: set[str] = set()

        if not configs:
            return server_rows, warnings

        def add_warning(message: str) -> None:
            formatted = message.strip()
            if not formatted:
                return
            if formatted not in warning_seen:
                warnings.append(formatted)
                warning_seen.add(formatted)

        for server_name in sorted(configs.keys()):
            config = configs.get(server_name)
            if not config:
                continue
            resources = list(config.ui_resources or [])
            has_skybridge_signal = bool(
                config.enabled or resources or config.tools or config.warnings
            )
            if not has_skybridge_signal:
                continue

            valid_resource_count = sum(1 for resource in resources if resource.is_skybridge)

            server_rows.append(
                {
                    "server_name": server_name,
                    "config": config,
                    "resources": resources,
                    "valid_resource_count": valid_resource_count,
                    "total_resource_count": len(resources),
                    "active_tools": [
                        {
                            "name": tool.display_name,
                            "template": str(tool.template_uri) if tool.template_uri else None,
                        }
                        for tool in config.tools
                        if tool.is_valid
                    ],
                    "enabled": config.enabled,
                }
            )

            for warning in config.warnings:
                message = warning.strip()
                if not message:
                    continue
                if not message.startswith(server_name):
                    message = f"{server_name} {message}"
                add_warning(message)

        return server_rows, warnings

    def show_skybridge_summary(
        self,
        agent_name: str,
        configs: Mapping[str, "SkybridgeServerConfig"] | None,
    ) -> None:
        """Display aggregated Skybridge status."""
        server_rows, warnings = self.summarize_skybridge_configs(configs)

        if not server_rows and not warnings:
            return

        heading = "[dim]OpenAI Apps SDK ([/dim][cyan]skybridge[/cyan][dim]) detected:[/dim]"
        console.console.print()
        console.console.print(heading, markup=self._markup)

        if not server_rows:
            console.console.print("[dim]  ● none detected[/dim]", markup=self._markup)
        else:
            for row in server_rows:
                server_name = row["server_name"]
                resource_count = row["valid_resource_count"]
                tool_infos = row["active_tools"]
                enabled = row["enabled"]

                tool_count = len(tool_infos)
                tool_word = "tool" if tool_count == 1 else "tools"
                resource_word = (
                    "skybridge resource" if resource_count == 1 else "skybridge resources"
                )
                tool_segment = f"[cyan]{tool_count}[/cyan][dim] {tool_word}[/dim]"
                resource_segment = f"[cyan]{resource_count}[/cyan][dim] {resource_word}[/dim]"
                name_style = "cyan" if enabled else "yellow"
                status_suffix = "" if enabled else "[dim] (issues detected)[/dim]"

                console.console.print(
                    f"[dim]  ● [/dim][{name_style}]{server_name}[/{name_style}]{status_suffix}"
                    f"[dim] — [/dim]{tool_segment}[dim], [/dim]{resource_segment}",
                    markup=self._markup,
                )

                if tool_infos:
                    for tool in tool_infos:
                        template_info = (
                            f" [dim]({tool['template']})[/dim]" if tool["template"] else ""
                        )
                        console.console.print(
                            f"[dim]     · [/dim]{tool['name']}{template_info}", markup=self._markup
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
                console.console.print(f"[yellow]- {warning}[/yellow]", markup=self._markup)
