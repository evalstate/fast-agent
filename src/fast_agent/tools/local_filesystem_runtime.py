"""Local filesystem runtime for shell-enabled agents.

Provides ACP-compatible ``read_text_file`` / ``write_text_file`` tool
implementations for non-ACP environments plus local edit tools such as
``apply_patch`` and ``edit_file``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, TextContent, Tool

from fast_agent.patch.engine import apply_patch as run_apply_patch
from fast_agent.patch.errors import ApplyPatchError
from fast_agent.tools.apply_patch_tool import (
    build_apply_patch_tool,
    extract_apply_patch_input,
)
from fast_agent.tools.edit_file_engine import (
    edit_file as run_edit_file,
)
from fast_agent.tools.edit_file_engine import (
    serialize_edit_file_result,
)
from fast_agent.tools.edit_file_tool import (
    build_edit_file_tool,
    extract_edit_file_input,
)
from fast_agent.tools.filesystem_tool_definitions import (
    build_read_text_file_tool,
    build_write_text_file_tool,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
    from fast_agent.types import RequestParams


class LocalFilesystemRuntime:
    """Expose local filesystem tools with ACP-compatible signatures."""

    def __init__(
        self,
        logger,
        working_directory: Path | None = None,
        *,
        enable_read: bool = True,
        enable_write: bool = True,
        enable_apply_patch: bool = False,
        enable_edit_file: bool = False,
        tool_handler_resolver: "Callable[[RequestParams | None], ToolExecutionHandler | None]"
        | None = None,
    ) -> None:
        self._logger = logger
        self._working_directory = working_directory
        self._enable_read = enable_read
        self._enable_write = enable_write
        self._enable_apply_patch = enable_apply_patch
        self._enable_edit_file = enable_edit_file
        self._tool_handler_resolver = tool_handler_resolver

        self._read_tool = build_read_text_file_tool()
        self._write_tool = build_write_text_file_tool()
        self._apply_patch_tool = build_apply_patch_tool()
        self._edit_file_tool = build_edit_file_tool()

    @property
    def tools(self) -> list[Tool]:
        """Return locally supported filesystem tools."""
        tools: list[Tool] = []
        if self._enable_read:
            tools.append(self._read_tool)
        if self._enable_write:
            tools.append(self._write_tool)
        if self._enable_apply_patch:
            tools.append(self._apply_patch_tool)
        if self._enable_edit_file:
            tools.append(self._edit_file_tool)
        return tools

    def set_enabled_tools(
        self,
        *,
        enable_read: bool,
        enable_write: bool,
        enable_apply_patch: bool,
        enable_edit_file: bool | None = None,
    ) -> None:
        """Update enabled filesystem tool flags."""
        self._enable_read = enable_read
        self._enable_write = enable_write
        self._enable_apply_patch = enable_apply_patch
        if enable_edit_file is not None:
            self._enable_edit_file = enable_edit_file

    def set_working_directory(self, working_directory: Path | None) -> None:
        """Update the base directory used for relative file paths."""
        self._working_directory = working_directory

    def set_tool_handler_resolver(
        self,
        resolver: "Callable[[RequestParams | None], ToolExecutionHandler | None] | None",
    ) -> None:
        """Update the per-request tool handler resolver used for local telemetry."""
        self._tool_handler_resolver = resolver

    def _base_directory(self) -> Path:
        if self._working_directory is None:
            return Path.cwd()
        if self._working_directory.is_absolute():
            return self._working_directory.resolve()
        return (Path.cwd() / self._working_directory).resolve()

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self._base_directory() / candidate).resolve()

    @staticmethod
    def _coerce_positive_int(value: Any, field: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"Error: '{field}' argument must be an integer greater than or equal to 1"
            )
        return value

    def has_tool(self, tool_name: str) -> bool:
        if tool_name == "read_text_file":
            return self._enable_read
        if tool_name == "write_text_file":
            return self._enable_write
        if tool_name == "apply_patch":
            return self._enable_apply_patch
        if tool_name == "edit_file":
            return self._enable_edit_file
        return False

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params: "RequestParams | None" = None,
    ) -> CallToolResult:
        if name == "read_text_file" and self._enable_read:
            return await self._call_with_tracking(
                "read_text_file",
                arguments,
                tool_use_id,
                request_params,
                self.read_text_file,
            )
        if name == "write_text_file" and self._enable_write:
            return await self._call_with_tracking(
                "write_text_file",
                arguments,
                tool_use_id,
                request_params,
                self.write_text_file,
            )
        if name == "apply_patch" and self._enable_apply_patch:
            return await self._call_with_tracking(
                "apply_patch",
                arguments,
                tool_use_id,
                request_params,
                self.apply_patch,
            )
        if name == "edit_file" and self._enable_edit_file:
            return await self._call_with_tracking(
                "edit_file",
                arguments,
                tool_use_id,
                request_params,
                self.edit_file,
            )

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Error: unsupported filesystem tool '{name}'",
                )
            ],
            isError=True,
        )

    async def _call_with_tracking(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        tool_use_id: str | None,
        request_params: "RequestParams | None",
        method: "Callable[[dict[str, Any] | None, str | None], Awaitable[CallToolResult]]",
    ) -> CallToolResult:
        tool_handler = (
            self._tool_handler_resolver(request_params)
            if self._tool_handler_resolver is not None
            else None
        )
        tool_call_id: str | None = None
        if tool_handler is not None:
            try:
                tool_call_id = await tool_handler.on_tool_start(
                    tool_name,
                    "local",
                    arguments,
                    tool_use_id,
                )
            except Exception:
                tool_call_id = None

        result = await method(arguments, tool_use_id)

        if tool_handler is not None and tool_call_id is not None:
            error_text: str | None = None
            if result.isError:
                error_text = self._extract_error_text(result, tool_name)
            try:
                await tool_handler.on_tool_complete(
                    tool_call_id,
                    not result.isError,
                    result.content if not result.isError else None,
                    error_text,
                )
            except Exception:
                pass

        return result

    @staticmethod
    def _extract_error_text(result: CallToolResult, tool_name: str) -> str:
        content = result.content
        if (
            isinstance(content, list)
            and content
            and isinstance(content[0], TextContent)
            and isinstance(content[0].text, str)
        ):
            return content[0].text
        return f"{tool_name} failed"

    async def read_text_file(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        """Read a local text file, optionally slicing by line and limit."""
        del tool_use_id

        if not isinstance(arguments, dict):
            return CallToolResult(
                content=[TextContent(type="text", text="Error: arguments must be a dict")],
                isError=True,
            )

        path_value = arguments.get("path")
        if not path_value or not isinstance(path_value, str):
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="Error: 'path' argument is required and must be a string",
                    )
                ],
                isError=True,
            )

        try:
            line = self._coerce_positive_int(arguments.get("line"), "line")
            limit = self._coerce_positive_int(arguments.get("limit"), "limit")
        except ValueError as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc))],
                isError=True,
            )

        resolved_path = self._resolve_path(path_value.strip())

        try:
            content = resolved_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            self._logger.error(f"Error reading file: {exc}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error reading file: {exc}")],
                isError=True,
            )

        if line is not None or limit is not None:
            lines = content.splitlines()
            start_index = (line - 1) if line is not None else 0
            end_index = start_index + limit if limit is not None else None
            content = "\n".join(lines[start_index:end_index])

        self._logger.debug(f"Read local file: {resolved_path} ({len(content)} chars)")
        return CallToolResult(
            content=[TextContent(type="text", text=content)],
            isError=False,
        )

    async def write_text_file(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        """Write a local text file, creating parent directories as needed."""
        del tool_use_id

        if not isinstance(arguments, dict):
            return CallToolResult(
                content=[TextContent(type="text", text="Error: arguments must be a dict")],
                isError=True,
            )

        path_value = arguments.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="Error: 'path' argument is required and must be a string",
                    )
                ],
                isError=True,
            )

        content_value = arguments.get("content")
        if not isinstance(content_value, str):
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="Error: 'content' argument is required and must be a string",
                    )
                ],
                isError=True,
            )

        resolved_path = self._resolve_path(path_value.strip())
        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(content_value, encoding="utf-8", errors="replace")
        except Exception as exc:
            self._logger.error(f"Error writing file: {exc}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error writing file: {exc}")],
                isError=True,
            )

        self._logger.debug(f"Wrote local file: {resolved_path} ({len(content_value)} chars)")
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"Successfully wrote {len(content_value)} characters to {path_value.strip()}"
                    ),
                )
            ],
            isError=False,
        )

    async def apply_patch(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        """Apply a patch using the local apply_patch engine."""
        del tool_use_id

        if not isinstance(arguments, dict):
            return CallToolResult(
                content=[TextContent(type="text", text="Error: arguments must be a dict")],
                isError=True,
            )

        patch_text = extract_apply_patch_input(arguments)
        if patch_text is None:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="Error: 'input' argument is required and must be a string",
                    )
                ],
                isError=True,
            )

        stdout = io.StringIO()
        stderr = io.StringIO()
        base_directory = self._base_directory()
        try:
            run_apply_patch(patch_text, stdout, stderr, base_directory=base_directory)
        except ApplyPatchError as exc:
            self._logger.error(f"Error applying patch: {exc}")
            error_text = stderr.getvalue().strip() or str(exc)
            return CallToolResult(
                content=[TextContent(type="text", text=error_text)],
                isError=True,
            )

        output = stdout.getvalue().strip()
        if not output:
            output = "Success. Updated the requested files."
        self._logger.debug("Applied local patch", base_directory=str(base_directory))
        return CallToolResult(
            content=[TextContent(type="text", text=output)],
            isError=False,
        )

    async def edit_file(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        """Edit a local file using exact string replacement semantics."""
        del tool_use_id

        edit_input = extract_edit_file_input(arguments)
        if edit_input is None:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            "Error: 'path', 'old_string', and 'new_string' arguments are required "
                            "and must be strings; 'replace_all' must be a boolean when provided"
                        ),
                    )
                ],
                isError=True,
            )

        path_value, old_string, new_string, replace_all = edit_input
        resolved_path = self._resolve_path(path_value)
        result_payload = run_edit_file(
            resolved_path,
            display_path=path_value,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
        )
        structured_payload = serialize_edit_file_result(result_payload)
        payload_text = json.dumps(structured_payload, ensure_ascii=False, indent=2)
        is_error = structured_payload["success"] is False
        return CallToolResult(
            content=[TextContent(type="text", text=payload_text)],
            structuredContent=structured_payload,
            isError=is_error,
        )

    def metadata(self) -> dict[str, Any]:
        """Expose runtime metadata for tool displays and diagnostics."""
        tools: list[str] = []
        if self._enable_read:
            tools.append("read_text_file")
        if self._enable_write:
            tools.append("write_text_file")
        if self._enable_apply_patch:
            tools.append("apply_patch")
        if self._enable_edit_file:
            tools.append("edit_file")

        return {
            "type": "local_filesystem",
            "tools": tools,
            "working_directory": str(self._base_directory()),
        }
