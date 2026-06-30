"""Filesystem tools backed by a session-owned filesystem."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, TextContent, Tool

from fast_agent.patch.errors import ApplyPatchError, InvalidHunkError, InvalidPatchError
from fast_agent.patch.parser import parse_patch
from fast_agent.tools.apply_patch_tool import (
    APPLY_PATCH_TOOL_NAME,
    build_apply_patch_tool,
    extract_apply_patch_input,
)
from fast_agent.tools.edit_file_engine import edit_file as run_edit_file
from fast_agent.tools.edit_file_engine import (
    serialize_edit_file_result,
)
from fast_agent.tools.edit_file_tool import (
    EDIT_FILE_TOOL_NAME,
    build_edit_file_tool,
    extract_edit_file_input,
)
from fast_agent.tools.filesystem_tool_args import (
    parse_read_text_file_arguments,
    parse_write_text_file_arguments,
)
from fast_agent.tools.filesystem_tool_definitions import (
    READ_TEXT_FILE_TOOL_NAME,
    WRITE_TEXT_FILE_TOOL_NAME,
    build_read_text_file_tool,
    build_write_text_file_tool,
)
from fast_agent.tools.filesystem_tool_specs import FilesystemToolSpec, enabled_tool_specs
from fast_agent.tools.session_patch import apply_patch_to_session_filesystem
from fast_agent.tools.tool_sources import SHELL_TOOL_SOURCE, set_tool_source

if TYPE_CHECKING:
    from fast_agent.tools.session_environment import SessionFilesystem
    from fast_agent.types import RequestParams


def _text_result(message: str, *, is_error: bool) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=is_error,
    )


class SessionFilesystemRuntime:
    """Expose existing LLM filesystem tools against a session filesystem."""

    def __init__(
        self,
        filesystem: "SessionFilesystem",
        *,
        enable_read: bool = True,
        enable_write: bool = True,
        enable_apply_patch: bool = False,
        enable_edit_file: bool = False,
    ) -> None:
        self._filesystem = filesystem
        self._enable_read = enable_read
        self._enable_write = enable_write
        self._enable_apply_patch = enable_apply_patch
        self._enable_edit_file = enable_edit_file
        self._read_tool = set_tool_source(build_read_text_file_tool(), SHELL_TOOL_SOURCE)
        self._write_tool = set_tool_source(build_write_text_file_tool(), SHELL_TOOL_SOURCE)
        self._apply_patch_tool = set_tool_source(build_apply_patch_tool(), SHELL_TOOL_SOURCE)
        self._edit_file_tool = set_tool_source(build_edit_file_tool(), SHELL_TOOL_SOURCE)
        self._tool_specs: tuple[FilesystemToolSpec, ...] = (
            FilesystemToolSpec(
                name=READ_TEXT_FILE_TOOL_NAME,
                enabled=lambda: self._enable_read,
                tool=lambda: self._read_tool,
                handler=self.read_text_file,
            ),
            FilesystemToolSpec(
                name=WRITE_TEXT_FILE_TOOL_NAME,
                enabled=lambda: self._enable_write,
                tool=lambda: self._write_tool,
                handler=self.write_text_file,
            ),
            FilesystemToolSpec(
                name=APPLY_PATCH_TOOL_NAME,
                enabled=lambda: self._enable_apply_patch,
                tool=lambda: self._apply_patch_tool,
                handler=self.apply_patch,
            ),
            FilesystemToolSpec(
                name=EDIT_FILE_TOOL_NAME,
                enabled=lambda: self._enable_edit_file,
                tool=lambda: self._edit_file_tool,
                handler=self.edit_file,
            ),
        )

    @property
    def tools(self) -> list[Tool]:
        return [spec.tool() for spec in enabled_tool_specs(self._tool_specs)]

    def set_enabled_tools(
        self,
        *,
        enable_read: bool,
        enable_write: bool,
        enable_apply_patch: bool,
        enable_edit_file: bool | None = None,
    ) -> None:
        self._enable_read = enable_read
        self._enable_write = enable_write
        self._enable_apply_patch = enable_apply_patch
        if enable_edit_file is not None:
            self._enable_edit_file = enable_edit_file

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params: "RequestParams | None" = None,
    ) -> CallToolResult:
        del request_params
        for spec in self._tool_specs:
            if spec.name == name and spec.enabled():
                return await spec.handler(arguments, tool_use_id)
        return _text_result(f"Error: unsupported filesystem tool '{name}'", is_error=True)

    async def read_text_file(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        del tool_use_id
        try:
            parsed = parse_read_text_file_arguments(arguments)
            content = await self._filesystem.read_text(parsed.path)
        except Exception as exc:
            return _text_result(f"Error reading file: {exc}", is_error=True)

        lines = content.splitlines()
        start_index = (parsed.line - 1) if parsed.line is not None else 0
        if start_index < 0:
            start_index = 0
        selected = lines[start_index:]
        if parsed.limit is not None:
            selected = selected[: parsed.limit]
        return _text_result("\n".join(selected), is_error=False)

    async def write_text_file(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        del tool_use_id
        try:
            parsed = parse_write_text_file_arguments(arguments)
            await self._filesystem.write_text(parsed.path, parsed.content)
        except Exception as exc:
            return _text_result(f"Error writing file: {exc}", is_error=True)
        return _text_result(f"Successfully wrote file: {parsed.path}", is_error=False)

    async def edit_file(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        del tool_use_id
        edit_input = extract_edit_file_input(arguments)
        if edit_input is None:
            return _text_result(
                "Error: 'path', 'old_string', and 'new_string' arguments are required",
                is_error=True,
            )

        with tempfile.TemporaryDirectory(prefix="fast-agent-session-edit-") as temp_dir:
            local_path = Path(temp_dir) / "target.txt"
            try:
                local_path.write_text(await self._filesystem.read_text(edit_input.path), encoding="utf-8")
            except Exception as exc:
                return _text_result(f"Error reading file: {exc}", is_error=True)

            result = run_edit_file(
                local_path,
                display_path=edit_input.path,
                old_string=edit_input.old_string,
                new_string=edit_input.new_string,
                replace_all=edit_input.replace_all,
            )
            if result["success"] is True:
                await self._filesystem.write_text(
                    edit_input.path,
                    local_path.read_text(encoding="utf-8"),
                )
            return _text_result(
                _format_jsonish(serialize_edit_file_result(result)),
                is_error=result["success"] is False,
            )

    async def apply_patch(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        del tool_use_id
        patch_text = extract_apply_patch_input(arguments)
        if patch_text is None:
            return _text_result("Error: 'input' argument is required and must be a string", is_error=True)

        try:
            parsed = parse_patch(patch_text)
        except InvalidPatchError as exc:
            return _text_result(f"Invalid patch: {exc}", is_error=True)
        except InvalidHunkError as exc:
            return _text_result(
                f"Invalid patch hunk on line {exc.line_number}: {exc.message}",
                is_error=True,
            )

        try:
            output = await apply_patch_to_session_filesystem(self._filesystem, parsed.hunks)
        except ApplyPatchError as exc:
            return _text_result(str(exc), is_error=True)
        except Exception as exc:
            return _text_result(f"Error applying patch: {exc}", is_error=True)
        return _text_result(output, is_error=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "session_filesystem",
            "cwd": self._filesystem.cwd,
            "tools": [tool.name for tool in self.tools],
        }


def _format_jsonish(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2)
