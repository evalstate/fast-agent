"""Explicit rejection of ACP capabilities the harness does not advertise."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Never

from acp.exceptions import RequestError

if TYPE_CHECKING:
    from acp.schema import (
        NesRepository,
        NesSuggestContext,
        Position,
        Range,
        TextDocumentContentChangeEvent,
        WorkspaceFolder,
    )


class UnsupportedACPMethods:
    """Keep unsupported ACP requests distinct from successful no-op responses."""

    async def delete_session(self, session_id: str, **kwargs: Any) -> Never:
        raise RequestError.method_not_found("session/delete")

    async def list_providers(self, **kwargs: Any) -> Never:
        raise RequestError.method_not_found("providers/list")

    async def set_provider(
        self,
        provider_id: str,
        api_type: str,
        base_url: str,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Never:
        raise RequestError.method_not_found("providers/set")

    async def disable_provider(self, provider_id: str, **kwargs: Any) -> Never:
        raise RequestError.method_not_found("providers/disable")

    async def logout(self, **kwargs: Any) -> Never:
        raise RequestError.method_not_found("logout")

    async def mcp_message(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> Never:
        raise RequestError.method_not_found("mcp/message")

    async def notify_mcp(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """This capability is not advertised; ignore unsolicited notifications."""

    async def start_nes(
        self,
        workspace_uri: str | None = None,
        workspace_folders: list[WorkspaceFolder] | None = None,
        repository: NesRepository | None = None,
        **kwargs: Any,
    ) -> Never:
        raise RequestError.method_not_found("nes/start")

    async def suggest_nes(
        self,
        session_id: str,
        uri: str,
        version: int,
        position: Position,
        trigger_kind: Literal["automatic", "diagnostic", "manual"],
        selection: Range | None = None,
        context: NesSuggestContext | None = None,
        **kwargs: Any,
    ) -> Never:
        raise RequestError.method_not_found("nes/suggest")

    async def close_nes(self, session_id: str, **kwargs: Any) -> Never:
        raise RequestError.method_not_found("nes/close")

    async def accept_nes(self, session_id: str, id: str, **kwargs: Any) -> None:
        """This capability is not advertised; ignore unsolicited notifications."""

    async def reject_nes(
        self,
        session_id: str,
        id: str,
        reason: Literal["rejected", "ignored", "replaced", "cancelled"] | None = None,
        **kwargs: Any,
    ) -> None:
        """This capability is not advertised; ignore unsolicited notifications."""

    async def did_open(
        self, session_id: str, uri: str, language_id: str, version: int, text: str, **kwargs: Any
    ) -> None:
        """This capability is not advertised; ignore unsolicited notifications."""

    async def did_change(
        self,
        session_id: str,
        uri: str,
        version: int,
        content_changes: list[TextDocumentContentChangeEvent],
        **kwargs: Any,
    ) -> None:
        """This capability is not advertised; ignore unsolicited notifications."""

    async def did_close(self, session_id: str, uri: str, **kwargs: Any) -> None:
        """This capability is not advertised; ignore unsolicited notifications."""

    async def did_save(self, session_id: str, uri: str, **kwargs: Any) -> None:
        """This capability is not advertised; ignore unsolicited notifications."""

    async def did_focus(
        self,
        session_id: str,
        uri: str,
        version: int,
        position: Position,
        visible_range: Range,
        **kwargs: Any,
    ) -> None:
        """This capability is not advertised; ignore unsolicited notifications."""
