"""Filesystem tools backed by an environment-owned filesystem."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from fast_agent.mcp.mime_utils import is_image_mime_type
from fast_agent.mcp.tool_result_metadata import set_tool_result_media_preview
from fast_agent.patch.errors import ApplyPatchError, InvalidHunkError, InvalidPatchError
from fast_agent.patch.parser import parse_patch
from fast_agent.tools.apply_patch_tool import extract_apply_patch_input
from fast_agent.tools.attach_media import (
    DEFAULT_ATTACH_MEDIA_MAX_BYTES,
    attach_media_staging_message,
    attachment_uri,
    build_attach_media_from_bytes,
    build_attach_media_link,
    is_provider_fetchable_uri,
    parse_attach_media_arguments,
)
from fast_agent.tools.edit_file_engine import edit_file as run_edit_file
from fast_agent.tools.edit_file_engine import (
    serialize_edit_file_result,
)
from fast_agent.tools.edit_file_tool import extract_edit_file_input
from fast_agent.tools.environment_patch import apply_patch_to_environment_filesystem
from fast_agent.tools.filesystem_runtime_base import FilesystemRuntimeBase, text_result
from fast_agent.tools.filesystem_tool_args import (
    parse_read_text_file_arguments,
    parse_write_text_file_arguments,
)

from .execution_environment import EnvironmentFilesystemWithBytes

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.types import CallToolResult

    from fast_agent.llm.model_info import ModelInfo
    from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
    from fast_agent.tools.execution_environment import EnvironmentFilesystem
    from fast_agent.types import RequestParams


class EnvironmentFilesystemRuntime(FilesystemRuntimeBase):
    """Expose existing LLM filesystem tools against an environment filesystem."""

    def __init__(
        self,
        filesystem: "EnvironmentFilesystem",
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
        self._filesystem = filesystem
        super().__init__(
            tracking_source="environment",
            enable_read=enable_read,
            enable_write=enable_write,
            enable_apply_patch=enable_apply_patch,
            enable_edit_file=enable_edit_file,
            enable_attach_media=enable_attach_media,
            attach_media_max_bytes=attach_media_max_bytes,
            model_info=model_info,
            tool_handler_resolver=tool_handler_resolver,
        )

    def _attach_media_enabled(self) -> bool:
        if not isinstance(self._filesystem, EnvironmentFilesystemWithBytes):
            return False
        return super()._attach_media_enabled()

    async def read_text_file(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        del tool_use_id
        try:
            parsed = parse_read_text_file_arguments(arguments)
            content = await self._filesystem.read_text(parsed.path)
        except Exception as exc:
            return text_result(f"Error reading file: {exc}", is_error=True)

        if parsed.line is not None or parsed.limit is not None:
            lines = content.splitlines()
            start_index = (parsed.line - 1) if parsed.line is not None else 0
            selected = lines[start_index:]
            if parsed.limit is not None:
                selected = selected[: parsed.limit]
            content = "\n".join(selected)
        return text_result(content, is_error=False)

    async def write_text_file(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        del tool_use_id
        try:
            parsed = parse_write_text_file_arguments(arguments)
            await self._filesystem.write_text(parsed.path, parsed.content)
        except Exception as exc:
            return text_result(f"Error writing file: {exc}", is_error=True)
        return text_result(f"Successfully wrote file: {parsed.path}", is_error=False)

    async def attach_media(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        del tool_use_id
        if not isinstance(self._filesystem, EnvironmentFilesystemWithBytes):
            return text_result(
                "Error: attach_media is not available for this environment filesystem",
                is_error=True,
            )

        try:
            parsed_args = parse_attach_media_arguments(arguments)
            source = parsed_args.source
            if is_provider_fetchable_uri(source):
                attached = build_attach_media_link(
                    source,
                    mime_type=parsed_args.mime_type,
                    name=parsed_args.name,
                    description=parsed_args.description,
                    model_info=self._model_info,
                )
            else:
                environment_path = _environment_path_from_source(source)
                resolved_path = self._filesystem.resolve_path(environment_path)
                data = await self._filesystem.read_bytes(environment_path)
                attached = build_attach_media_from_bytes(
                    source=attachment_uri(resolved_path),
                    data=data,
                    mime_type=parsed_args.mime_type,
                    name=parsed_args.name,
                    model_info=self._model_info,
                    max_bytes=self._attach_media_max_bytes,
                )
        except Exception as exc:
            return text_result(str(exc), is_error=True)

        self._pending_media_attachments.append(attached.block)
        result = text_result(
            attach_media_staging_message(attached),
            is_error=False,
        )
        if is_image_mime_type(attached.mime_type):
            set_tool_result_media_preview(result, [attached.block])
        return result

    async def edit_file(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        del tool_use_id
        edit_input = extract_edit_file_input(arguments)
        if edit_input is None:
            return text_result(
                "Error: 'path', 'old_string', and 'new_string' arguments are required",
                is_error=True,
            )

        with tempfile.TemporaryDirectory(prefix="fast-agent-environment-edit-") as temp_dir:
            local_path = Path(temp_dir) / "target.txt"
            try:
                local_path.write_text(
                    await self._filesystem.read_text(edit_input.path), encoding="utf-8"
                )
            except Exception as exc:
                return text_result(f"Error reading file: {exc}", is_error=True)

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
            return text_result(
                _format_jsonish(serialize_edit_file_result(result)),
                is_error=result["success"] is False,
            )

    async def apply_patch(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        del tool_use_id
        patch_text = extract_apply_patch_input(arguments)
        if patch_text is None:
            return text_result(
                "Error: 'input' argument is required and must be a string", is_error=True
            )

        try:
            parsed = parse_patch(patch_text)
        except InvalidPatchError as exc:
            return text_result(f"Invalid patch: {exc}", is_error=True)
        except InvalidHunkError as exc:
            return text_result(
                f"Invalid patch hunk on line {exc.line_number}: {exc.message}",
                is_error=True,
            )

        try:
            output = await apply_patch_to_environment_filesystem(self._filesystem, parsed.hunks)
        except ApplyPatchError as exc:
            return text_result(str(exc), is_error=True)
        except Exception as exc:
            return text_result(f"Error applying patch: {exc}", is_error=True)
        return text_result(output, is_error=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "environment_filesystem",
            "cwd": self._filesystem.cwd,
            "tools": [tool.name for tool in self.tools],
        }


def _environment_path_from_source(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme == "file":
        return unquote(parsed.path)
    if parsed.scheme:
        if parsed.scheme == "internal":
            raise ValueError(
                "Error: attach_media does not read internal resources; use get_resource for "
                "internal:// or MCP resource URIs"
            )
        raise ValueError(
            f"Error: unsupported attachment URI scheme '{parsed.scheme}'; use get_resource for "
            "internal:// or MCP resource URIs"
        )
    return source


def _format_jsonish(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2)
