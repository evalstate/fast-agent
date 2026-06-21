"""Protocol-neutral agent request/response envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from mcp.types import TextContent

from fast_agent.mcp.helpers.content_helpers import normalize_to_extended_list
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.llm.request_params import RequestParams


class ProgressReporter(Protocol):
    """Minimal progress sink implemented by protocol adapters."""

    async def report(
        self,
        message: str,
        *,
        progress: float | None = None,
        total: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentAuth:
    """Request-scoped authenticated identity."""

    token: str | None = None
    scheme: str = "bearer"
    provider: str | None = None
    subject: str | None = None
    client_id: str | None = None
    scopes: tuple[str, ...] = ()
    claims: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def bearer(
        cls,
        token: str,
        *,
        provider: str | None = None,
        subject: str | None = None,
        client_id: str | None = None,
        scopes: tuple[str, ...] = (),
        claims: Mapping[str, Any] | None = None,
    ) -> AgentAuth:
        return cls(
            token=token,
            provider=provider,
            subject=subject,
            client_id=client_id,
            scopes=scopes,
            claims=claims or {},
        )

    @classmethod
    def huggingface(
        cls,
        token: str,
        *,
        subject: str | None = None,
        client_id: str | None = None,
        scopes: tuple[str, ...] = (),
        claims: Mapping[str, Any] | None = None,
    ) -> AgentAuth:
        return cls.bearer(
            token,
            provider="huggingface",
            subject=subject,
            client_id=client_id,
            scopes=scopes,
            claims=claims,
        )


@dataclass(slots=True)
class AgentRequest:
    """An invocation request around a fast-agent prompt message."""

    message: PromptMessageExtended
    agent: str | None = None
    session_id: str | None = None
    auth: AgentAuth | None = None
    params: RequestParams | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    progress: ProgressReporter | None = None

    @classmethod
    def text(
        cls,
        text: str,
        *,
        agent: str | None = None,
        session_id: str | None = None,
        auth: AgentAuth | None = None,
        params: RequestParams | None = None,
        metadata: Mapping[str, Any] | None = None,
        progress: ProgressReporter | None = None,
    ) -> AgentRequest:
        return cls(
            message=PromptMessageExtended(
                role="user", content=[TextContent(type="text", text=text)]
            ),
            agent=agent,
            session_id=session_id,
            auth=auth,
            params=params,
            metadata=dict(metadata or {}),
            progress=progress,
        )

    @classmethod
    def from_message(
        cls,
        message: str | PromptMessageExtended,
        *,
        agent: str | None = None,
        session_id: str | None = None,
        auth: AgentAuth | None = None,
        params: RequestParams | None = None,
        metadata: Mapping[str, Any] | None = None,
        progress: ProgressReporter | None = None,
    ) -> AgentRequest:
        normalized = normalize_to_extended_list(message)
        if len(normalized) != 1:
            raise ValueError("AgentRequest.from_message expects exactly one message")
        return cls(
            message=normalized[0],
            agent=agent,
            session_id=session_id,
            auth=auth,
            params=params,
            metadata=dict(metadata or {}),
            progress=progress,
        )

    async def report(
        self,
        message: str,
        *,
        progress: float | None = None,
        total: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self.progress is not None:
            await self.progress.report(message, progress=progress, total=total, metadata=metadata)


@dataclass(slots=True)
class AgentResponse:
    """Protocol-neutral agent response around a prompt message."""

    message: PromptMessageExtended
    kind: str = "message"
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[Any, ...] = ()

    @classmethod
    def text(
        cls,
        text: str,
        *,
        kind: str = "message",
        metadata: Mapping[str, Any] | None = None,
        artifacts: tuple[Any, ...] = (),
    ) -> AgentResponse:
        return cls(
            message=PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text=text)],
            ),
            kind=kind,
            metadata=dict(metadata or {}),
            artifacts=artifacts,
        )

    def text_content(self) -> str:
        return self.message.all_text()

    def with_kind(self, kind: str) -> AgentResponse:
        return AgentResponse(
            message=self.message,
            kind=kind,
            metadata=dict(self.metadata),
            artifacts=self.artifacts,
        )
