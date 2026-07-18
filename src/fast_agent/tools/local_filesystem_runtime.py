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

from mcp.types import CallToolResult, TextContent

from fast_agent.mcp.mime_utils import is_image_mime_type
from fast_agent.mcp.tool_result_metadata import set_tool_result_media_preview
from fast_agent.patch.engine import apply_patch as run_apply_patch
from fast_agent.patch.errors import ApplyPatchError
from fast_agent.tools.apply_patch_tool import extract_apply_patch_input
from fast_agent.tools.attach_media import (
    DEFAULT_ATTACH_MEDIA_MAX_BYTES,
    attach_media_staging_message,
    build_attach_media,
    parse_attach_media_arguments,
)
from fast_agent.tools.edit_file_engine import (
    edit_file as run_edit_file,
)
from fast_agent.tools.edit_file_engine import (
    serialize_edit_file_result,
)
from fast_agent.tools.edit_file_tool import extract_edit_file_input
from fast_agent.tools.filesystem_runtime_base import FilesystemRuntimeBase, text_result
from fast_agent.tools.filesystem_tool_args import (
    is_permission_error,
    parse_read_text_file_arguments,
    parse_write_text_file_arguments,
    permission_denied_message,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_agent.llm.model_info import ModelInfo
    from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
    from fast_agent.types import RequestParams


class LocalFilesystemRuntime(FilesystemRuntimeBase):
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
        enable_attach_media: str | None = "auto",
        attach_media_max_bytes: int = DEFAULT_ATTACH_MEDIA_MAX_BYTES,
        model_info: "ModelInfo | None" = None,
        tool_handler_resolver: "Callable[[RequestParams | None], ToolExecutionHandler | None]"
        | None = None,
    ) -> None:
        self._logger = logger
        self._working_directory = working_directory
        super().__init__(
            tracking_source="local",
            enable_read=enable_read,
            enable_write=enable_write,
            enable_apply_patch=enable_apply_patch,
            enable_edit_file=enable_edit_file,
            enable_attach_media=enable_attach_media,
            attach_media_max_bytes=attach_media_max_bytes,
            model_info=model_info,
            tool_handler_resolver=tool_handler_resolver,
        )

    def set_working_directory(self, working_directory: Path | None) -> None:
        """Update the base directory used for relative file paths."""
        self._working_directory = working_directory

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

    async def read_text_file(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        """Read a local text file, optionally slicing by line and limit."""
        del tool_use_id

        try:
            parsed = parse_read_text_file_arguments(arguments)
        except ValueError as exc:
            return text_result(str(exc), is_error=True)

        resolved_path = self._resolve_path(parsed.path)

        try:
            content = resolved_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            self._logger.exception("Error reading file")
            if is_permission_error(exc):
                return text_result(permission_denied_message(parsed.path), is_error=True)
            return text_result(f"Error reading file: {exc}", is_error=True)

        if parsed.line is not None or parsed.limit is not None:
            lines = content.splitlines()
            start_index = (parsed.line - 1) if parsed.line is not None else 0
            end_index = start_index + parsed.limit if parsed.limit is not None else None
            content = "\n".join(lines[start_index:end_index])

        self._logger.debug(f"Read local file: {resolved_path} ({len(content)} chars)")
        return text_result(content, is_error=False)

    async def write_text_file(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        """Write a local text file, creating parent directories as needed."""
        del tool_use_id

        try:
            parsed = parse_write_text_file_arguments(arguments)
        except ValueError as exc:
            return text_result(str(exc), is_error=True)

        resolved_path = self._resolve_path(parsed.path)
        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(parsed.content, encoding="utf-8", errors="replace")
        except OSError as exc:
            self._logger.exception("Error writing file")
            if is_permission_error(exc):
                return text_result(permission_denied_message(parsed.path), is_error=True)
            return text_result(f"Error writing file: {exc}", is_error=True)

        self._logger.debug(f"Wrote local file: {resolved_path} ({len(parsed.content)} chars)")
        return text_result(
            f"Successfully wrote {len(parsed.content)} characters to {parsed.path}",
            is_error=False,
        )

    async def attach_media(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        """Stage a local file or provider-fetchable URI as model input."""
        del tool_use_id

        try:
            parsed_args = parse_attach_media_arguments(arguments)
            attached = build_attach_media(
                parsed_args.source,
                base_directory=self._base_directory(),
                mime_type=parsed_args.mime_type,
                name=parsed_args.name,
                description=parsed_args.description,
                model_info=self._model_info,
                max_bytes=self._attach_media_max_bytes,
            )
        except Exception as exc:
            self._logger.exception("Error attaching resource")
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc))],
                isError=True,
            )

        self._pending_media_attachments.append(attached.block)
        result = CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=attach_media_staging_message(attached),
                )
            ],
            isError=False,
        )
        if is_image_mime_type(attached.mime_type):
            set_tool_result_media_preview(result, [attached.block])
        return result

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

        resolved_path = self._resolve_path(edit_input.path)
        result_payload = run_edit_file(
            resolved_path,
            display_path=edit_input.path,
            old_string=edit_input.old_string,
            new_string=edit_input.new_string,
            replace_all=edit_input.replace_all,
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
        return {
            "type": "local_filesystem",
            "tools": [tool.name for tool in self.tools],
            "working_directory": str(self._base_directory()),
        }
