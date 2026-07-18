"""Filesystem tools backed by an environment-owned filesystem."""

from __future__ import annotations

import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from mcp.types import CallToolResult, ContentBlock, TextContent, Tool

from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.mime_utils import is_image_mime_type
from fast_agent.mcp.tool_result_metadata import set_tool_result_media_preview
from fast_agent.patch.errors import ApplyPatchError, InvalidHunkError, InvalidPatchError
from fast_agent.patch.parser import parse_patch
from fast_agent.tools.apply_patch_tool import (
    APPLY_PATCH_TOOL_NAME,
    build_apply_patch_tool,
    extract_apply_patch_input,
)
from fast_agent.tools.attach_media import (
    DEFAULT_ATTACH_MEDIA_MAX_BYTES,
    attach_media_staging_message,
    attachment_uri,
    build_attach_media_from_bytes,
    build_attach_media_link,
    is_provider_fetchable_uri,
    model_supports_attach_media,
    normalize_attach_media_max_bytes,
    parse_attach_media_arguments,
    supported_attach_media_mime_types,
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
from fast_agent.tools.environment_patch import apply_patch_to_environment_filesystem
from fast_agent.tools.filesystem_tool_args import (
    parse_read_text_file_arguments,
    parse_write_text_file_arguments,
)
from fast_agent.tools.filesystem_tool_definitions import (
    ATTACH_MEDIA_TOOL_NAME,
    READ_TEXT_FILE_TOOL_NAME,
    WRITE_TEXT_FILE_TOOL_NAME,
    build_attach_media_tool,
    build_read_text_file_tool,
    build_write_text_file_tool,
)
from fast_agent.tools.filesystem_tool_specs import (
    FilesystemToolSpec,
    enabled_tool_spec,
    enabled_tool_specs,
)
from fast_agent.tools.tool_sources import SHELL_TOOL_SOURCE, set_tool_source
from fast_agent.utils.text import strip_casefold

from .execution_environment import EnvironmentFilesystemWithBytes

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fast_agent.llm.model_info import ModelInfo
    from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
    from fast_agent.tools.execution_environment import EnvironmentFilesystem
    from fast_agent.types import RequestParams


def _text_result(message: str, *, is_error: bool) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=is_error,
    )


class EnvironmentFilesystemRuntime:
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
        self._enable_read = enable_read
        self._enable_write = enable_write
        self._enable_apply_patch = enable_apply_patch
        self._enable_edit_file = enable_edit_file
        self._enable_attach_media = enable_attach_media
        if self._enable_attach_media is None:
            self._enable_attach_media = "auto"
        self._attach_media_max_bytes = normalize_attach_media_max_bytes(attach_media_max_bytes)
        self._model_info = model_info
        self._tool_handler_resolver = tool_handler_resolver
        self._read_tool = set_tool_source(build_read_text_file_tool(), SHELL_TOOL_SOURCE)
        self._write_tool = set_tool_source(build_write_text_file_tool(), SHELL_TOOL_SOURCE)
        self._apply_patch_tool = set_tool_source(build_apply_patch_tool(), SHELL_TOOL_SOURCE)
        self._edit_file_tool = set_tool_source(build_edit_file_tool(), SHELL_TOOL_SOURCE)
        self._attach_media_tool = self._build_attach_media_tool()
        self._pending_media_attachments: list[ContentBlock] = []
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
            FilesystemToolSpec(
                name=ATTACH_MEDIA_TOOL_NAME,
                enabled=self._attach_media_enabled,
                tool=lambda: self._attach_media_tool,
                handler=self.attach_media,
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
        enable_attach_media: str | None = None,
    ) -> None:
        self._enable_read = enable_read
        self._enable_write = enable_write
        self._enable_apply_patch = enable_apply_patch
        if enable_edit_file is not None:
            self._enable_edit_file = enable_edit_file
        if enable_attach_media is not None:
            self._enable_attach_media = enable_attach_media

    def set_model_info(self, model_info: "ModelInfo | None") -> None:
        self._model_info = model_info
        self._attach_media_tool = self._build_attach_media_tool()

    def set_tool_handler_resolver(
        self,
        resolver: "Callable[[RequestParams | None], ToolExecutionHandler | None] | None",
    ) -> None:
        self._tool_handler_resolver = resolver

    def _build_attach_media_tool(self) -> Tool:
        return set_tool_source(
            build_attach_media_tool(
                supported_attach_media_mime_types(self._model_info),
                is_google=self._model_uses_google_media_payloads(),
                max_bytes=self._attach_media_max_bytes,
            ),
            SHELL_TOOL_SOURCE,
        )

    def _model_uses_google_media_payloads(self) -> bool:
        if self._model_info is None:
            return False
        return self._model_info.provider is Provider.GOOGLE or "gemini" in strip_casefold(
            self._model_info.name or ""
        )

    def _attach_media_enabled(self) -> bool:
        if not isinstance(self._filesystem, EnvironmentFilesystemWithBytes):
            return False
        if self._enable_attach_media == "off":
            return False
        if self._enable_attach_media == "on":
            return True
        return model_supports_attach_media(self._model_info)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params: "RequestParams | None" = None,
    ) -> CallToolResult:
        spec = enabled_tool_spec(self._tool_specs, name)
        if spec is not None:
            return await self._call_with_tracking(
                spec.name,
                arguments,
                tool_use_id,
                request_params,
                spec.handler,
            )
        return _text_result(f"Error: unsupported filesystem tool '{name}'", is_error=True)

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
                    "environment",
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
            with suppress(Exception):
                await tool_handler.on_tool_complete(
                    tool_call_id,
                    not result.isError,
                    result.content if not result.isError else None,
                    error_text,
                )

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
        del tool_use_id
        try:
            parsed = parse_read_text_file_arguments(arguments)
            content = await self._filesystem.read_text(parsed.path)
        except Exception as exc:
            return _text_result(f"Error reading file: {exc}", is_error=True)

        if parsed.line is not None or parsed.limit is not None:
            lines = content.splitlines()
            start_index = (parsed.line - 1) if parsed.line is not None else 0
            selected = lines[start_index:]
            if parsed.limit is not None:
                selected = selected[: parsed.limit]
            content = "\n".join(selected)
        return _text_result(content, is_error=False)

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

    def consume_pending_media_attachments(self) -> list[ContentBlock]:
        pending = self._pending_media_attachments
        self._pending_media_attachments = []
        return pending

    async def attach_media(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        del tool_use_id
        if not isinstance(self._filesystem, EnvironmentFilesystemWithBytes):
            return _text_result(
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
            return _text_result(str(exc), is_error=True)

        self._pending_media_attachments.append(attached.block)
        result = _text_result(
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
            return _text_result(
                "Error: 'path', 'old_string', and 'new_string' arguments are required",
                is_error=True,
            )

        with tempfile.TemporaryDirectory(prefix="fast-agent-environment-edit-") as temp_dir:
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
                try:
                    await self._filesystem.write_text(
                        edit_input.path,
                        local_path.read_text(encoding="utf-8"),
                    )
                except Exception as exc:
                    return _text_result(f"Error writing file: {exc}", is_error=True)
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
            output = await apply_patch_to_environment_filesystem(self._filesystem, parsed.hunks)
        except ApplyPatchError as exc:
            return _text_result(str(exc), is_error=True)
        except Exception as exc:
            return _text_result(f"Error applying patch: {exc}", is_error=True)
        return _text_result(output, is_error=False)

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
