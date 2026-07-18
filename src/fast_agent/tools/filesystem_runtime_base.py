"""Shared orchestration for filesystem tool runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, ContentBlock, TextContent, Tool

from fast_agent.llm.provider_types import Provider
from fast_agent.tools.apply_patch_tool import APPLY_PATCH_TOOL_NAME, build_apply_patch_tool
from fast_agent.tools.attach_media import (
    DEFAULT_ATTACH_MEDIA_MAX_BYTES,
    model_supports_attach_media,
    normalize_attach_media_max_bytes,
    supported_attach_media_mime_types,
)
from fast_agent.tools.edit_file_tool import EDIT_FILE_TOOL_NAME, build_edit_file_tool
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

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fast_agent.llm.model_info import ModelInfo
    from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
    from fast_agent.types import RequestParams


def text_result(message: str, *, is_error: bool) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=is_error,
    )


class FilesystemRuntimeBase(ABC):
    """Common tool registration, capability state, and execution tracking."""

    def __init__(
        self,
        *,
        tracking_source: str,
        enable_read: bool = True,
        enable_write: bool = True,
        enable_apply_patch: bool = False,
        enable_edit_file: bool = False,
        enable_attach_media: str | None = "auto",
        attach_media_max_bytes: int = DEFAULT_ATTACH_MEDIA_MAX_BYTES,
        model_info: ModelInfo | None = None,
        tool_handler_resolver: Callable[[RequestParams | None], ToolExecutionHandler | None]
        | None = None,
    ) -> None:
        self._tracking_source = tracking_source
        self._enable_read = enable_read
        self._enable_write = enable_write
        self._enable_apply_patch = enable_apply_patch
        self._enable_edit_file = enable_edit_file
        self._enable_attach_media = enable_attach_media or "auto"
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

    def set_model_info(self, model_info: ModelInfo | None) -> None:
        self._model_info = model_info
        self._attach_media_tool = self._build_attach_media_tool()

    def set_tool_handler_resolver(
        self,
        resolver: Callable[[RequestParams | None], ToolExecutionHandler | None] | None,
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
        request_params: RequestParams | None = None,
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
        return text_result(f"Error: unsupported filesystem tool '{name}'", is_error=True)

    async def _call_with_tracking(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        tool_use_id: str | None,
        request_params: RequestParams | None,
        method: Callable[
            [dict[str, Any] | None, str | None],
            Awaitable[CallToolResult],
        ],
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
                    self._tracking_source,
                    arguments,
                    tool_use_id,
                )
            except Exception:
                tool_call_id = None

        result = await method(arguments, tool_use_id)

        if tool_handler is not None and tool_call_id is not None:
            error_text = self._extract_error_text(result, tool_name) if result.isError else None
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

    def consume_pending_media_attachments(self) -> list[ContentBlock]:
        pending = self._pending_media_attachments
        self._pending_media_attachments = []
        return pending

    @abstractmethod
    async def read_text_file(
        self,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> CallToolResult: ...

    @abstractmethod
    async def write_text_file(
        self,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> CallToolResult: ...

    @abstractmethod
    async def attach_media(
        self,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> CallToolResult: ...

    @abstractmethod
    async def apply_patch(
        self,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> CallToolResult: ...

    @abstractmethod
    async def edit_file(
        self,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> CallToolResult: ...
