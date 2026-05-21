"""Remote A2A agent implementation."""

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.types import Message, Part, Role, SendMessageRequest, TaskState
from google.protobuf.json_format import MessageToDict, ParseDict
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.llm_decorator import LlmDecorator
from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.mcp.hf_auth import add_hf_auth_header
from fast_agent.mcp.oauth_client import build_oauth_provider
from fast_agent.types import LlmStopReason, PromptMessageExtended, RequestParams
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.message_display_helpers import build_user_message_display
from fast_agent.ui.progress_display import progress_display

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from a2a.types import AgentCard
    from mcp import Tool

    from fast_agent.a2a.config import A2AAgentConfig
    from fast_agent.config import MCPServerSettings
    from fast_agent.context import Context

_TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_CANCELLED",
    "TASK_STATE_REJECTED",
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_AUTH_REQUIRED",
}

_ERROR_STATES = {
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_CANCELLED",
    "TASK_STATE_REJECTED",
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_AUTH_REQUIRED",
}

logger = get_logger(__name__)

SUPPORTED_A2A_HTTP_TRANSPORTS = ["JSONRPC", "HTTP+JSON"]
_INPUT_REQUIRED_STATE = "TASK_STATE_INPUT_REQUIRED"


@dataclass(frozen=True)
class A2ADiagnostics:
    url: str
    transport: str | None
    remote_name: str | None
    context_id: str
    current_task_id: str | None
    last_task_state: str | None
    selected_transport_class: str | None


class A2ARemoteAgent(LlmDecorator):
    """A fast-agent AgentProtocol adapter for a remote A2A agent."""

    def __init__(
        self,
        config: AgentConfig,
        a2a_config: A2AAgentConfig,
        context: Context | None = None,
    ) -> None:
        super().__init__(config=config, context=context)
        self.a2a_config = a2a_config
        self.context_id = str(uuid.uuid4())
        self.current_task_id: str | None = None
        self.last_task_state: str | None = None
        self.remote_card: AgentCard | None = None
        self.display = ConsoleDisplay(config=context.config if context else None)
        self._client: Any | None = None
        self._httpx_client: httpx.AsyncClient | None = None
        self._stream_listeners: list[Callable[[StreamChunk], None]] = []

    @property
    def agent_type(self) -> AgentType:
        return AgentType.A2A

    async def initialize(self) -> None:
        await super().initialize()
        headers = add_hf_auth_header(self.a2a_config.url, self.a2a_config.headers)
        self._httpx_client = httpx.AsyncClient(
            headers=headers or None,
            timeout=self.a2a_config.request_timeout_seconds,
        )
        client_config = ClientConfig(
            streaming=self.a2a_config.streaming,
            polling=self.a2a_config.polling,
            httpx_client=self._httpx_client,
            accepted_output_modes=list(self.a2a_config.accepted_output_modes),
        )
        if self.a2a_config.transport:
            client_config.supported_protocol_bindings = [self.a2a_config.transport]
        else:
            client_config.supported_protocol_bindings = list(SUPPORTED_A2A_HTTP_TRANSPORTS)

        resolver = A2ACardResolver(
            self._httpx_client,
            self.a2a_config.url,
            self.a2a_config.relative_card_path or "/.well-known/agent-card.json",
        )
        self.remote_card = await resolver.get_agent_card()
        oauth_provider = self._build_oauth_provider_for_card(self.remote_card)
        if oauth_provider is not None:
            await self._httpx_client.aclose()
            self._httpx_client = httpx.AsyncClient(
                auth=oauth_provider,
                headers=headers or None,
                timeout=self.a2a_config.request_timeout_seconds,
            )
            client_config.httpx_client = self._httpx_client
        self._client = await create_client(
            self.remote_card,
            client_config=client_config,
        )

    async def shutdown(self) -> None:
        client = self._client
        if client is not None:
            await client.close()
            self._client = None
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None
        await super().shutdown()

    def add_stream_listener(self, listener: Callable[[StreamChunk], None]) -> Callable[[], None]:
        self._stream_listeners.append(listener)

        def remove_listener() -> None:
            try:
                self._stream_listeners.remove(listener)
            except ValueError:
                return

        return remove_listener

    def _clone_constructor_kwargs(self) -> dict[str, Any]:
        """Carry remote A2A connection configuration into detached clones."""
        return {"a2a_config": self.a2a_config}

    def _build_oauth_provider_for_card(self, card: AgentCard) -> Any | None:
        auth_config = self.a2a_config.auth
        if auth_config is not None and not auth_config.oauth:
            return None
        if auth_config is None and not _card_advertises_oauth(card):
            return None
        if self.a2a_config.headers:
            return None
        return build_oauth_provider(
            cast(
                "MCPServerSettings",
                SimpleNamespace(
                    name=self.config.name,
                    transport="http",
                    url=self.a2a_config.url,
                    auth=auth_config,
                ),
            )
        )

    def reset_a2a_state(self) -> None:
        self.context_id = str(uuid.uuid4())
        self.current_task_id = None
        self.last_task_state = None

    def diagnostics(self) -> A2ADiagnostics:
        return A2ADiagnostics(
            url=self.a2a_config.url,
            transport=self.a2a_config.transport,
            remote_name=self.remote_card.name if self.remote_card else None,
            context_id=self.context_id,
            current_task_id=self.current_task_id,
            last_task_state=self.last_task_state,
            selected_transport_class=self._selected_transport_class(),
        )

    async def generate_impl(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        del tools
        if self._client is None:
            raise RuntimeError("A2A remote agent is not initialized")

        use_history = request_params.use_history if request_params else self.config.use_history
        self._prepare_turn_state(use_history=use_history)
        self._timestamp_messages(messages)
        self._display_user_messages(messages)
        user_text = _latest_text(messages)
        request = SendMessageRequest(
            message=Message(
                role=Role.ROLE_USER,
                message_id=str(uuid.uuid4()),
                context_id=self.context_id,
                task_id=self.current_task_id,
                parts=_parts_from_messages(messages) or [Part(text=user_text)],
            )
        )

        self._log_a2a_progress(ProgressAction.SENDING, details=self._transport_label())
        result = await self._consume_events(self._client.send_message(request))
        self._log_a2a_progress(ProgressAction.READY, details=result.state or "completed")
        response_text = result.text or result.status_text or _state_message(result.state)
        if result.state in _ERROR_STATES:
            response_text = f"A2A task {result.state}: {response_text}"
        stop_reason = (
            LlmStopReason.PAUSE
            if result.state == _INPUT_REQUIRED_STATE
            else LlmStopReason.END_TURN
        )
        assistant_message = PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text=response_text)],
            stop_reason=stop_reason,
        )
        progress_display.pause(cancel_deferred_on_noop=True)
        await self.display.show_assistant_message(
            assistant_message,
            name=self.name,
            model="A2A",
            bottom_items=[self._transport_label()],
        )
        console.console.print()
        if use_history:
            self._persist_history(messages, assistant_message)
        return assistant_message

    def _prepare_turn_state(self, *, use_history: bool) -> None:
        if use_history:
            return
        if self.last_task_state == _INPUT_REQUIRED_STATE and self.current_task_id:
            return
        self.reset_a2a_state()

    def _display_user_messages(self, messages: list[PromptMessageExtended]) -> None:
        display_messages = [message for message in messages if message.role == "user"]
        if not display_messages:
            return
        message_text, attachments = build_user_message_display(display_messages)
        self.display.show_user_message(
            message_text,
            chat_turn=0,
            name=self.name,
            attachments=attachments if attachments else None,
            part_count=len(display_messages) if len(display_messages) > 1 else None,
        )

    def _transport_label(self) -> str:
        return f"A2A · {self.a2a_config.transport}" if self.a2a_config.transport else "A2A"

    def _selected_transport_class(self) -> str | None:
        if self._client is None:
            return None
        transport = getattr(self._client, "_transport", None)
        if transport is None:
            return self._client.__class__.__name__
        return transport.__class__.__name__

    def _log_a2a_progress(self, action: ProgressAction, *, details: str = "") -> None:
        logger.debug(
            "A2A request progress",
            data={
                "progress_action": action,
                "agent_name": self.name,
                "target": self.remote_card.name if self.remote_card else self.name,
                "details": details,
            },
        )

    async def _consume_events(self, events: Any) -> "_A2AResult":
        message_chunks: list[str] = []
        artifact_order: list[str] = []
        artifact_texts: dict[str, str] = {}
        state: str | None = None
        status_text: str | None = None

        async for event in events:
            if event.HasField("message"):
                text = _parts_text(event.message.parts)
                _append_text(message_chunks, text)
                self._emit_stream(text)
                continue

            if event.HasField("task"):
                state = TaskState.Name(event.task.status.state)
                self._advance_task_state(
                    state=state,
                    task_id=event.task.id,
                    context_id=event.task.context_id,
                )
                self._log_a2a_progress(ProgressAction.UPDATED, details=state)
                for artifact in event.task.artifacts:
                    _replace_artifact_text(artifact_order, artifact_texts, artifact, _parts_text(artifact.parts))
                continue

            if event.HasField("status_update"):
                status = event.status_update.status
                state = TaskState.Name(status.state)
                self._advance_task_state(
                    state=state,
                    task_id=event.status_update.task_id,
                    context_id=event.status_update.context_id,
                )
                self._log_a2a_progress(ProgressAction.UPDATED, details=state)
                if status.HasField("message"):
                    status_text = _parts_text(status.message.parts) or status_text
                continue

            if event.HasField("artifact_update"):
                update = event.artifact_update
                artifact = update.artifact
                text = _parts_text(artifact.parts)
                if not text:
                    continue
                _apply_artifact_update(
                    artifact_order,
                    artifact_texts,
                    artifact,
                    text,
                    append=update.append,
                )
                self._log_a2a_progress(ProgressAction.STREAMING, details=artifact.name)
                self._emit_stream(text)

        return _A2AResult(
            text="\n".join(
                chunk
                for chunk in [
                    *message_chunks,
                    *(artifact_texts[artifact_id] for artifact_id in artifact_order),
                ]
                if chunk
            ),
            state=state,
            status_text=status_text,
        )

    def _emit_stream(self, text: str) -> None:
        if not text:
            return
        chunk = StreamChunk(text=text)
        for listener in list(self._stream_listeners):
            listener(chunk)

    def _advance_task_state(self, *, state: str, task_id: str, context_id: str) -> None:
        self.last_task_state = state
        self.context_id = context_id
        if state == _INPUT_REQUIRED_STATE:
            self.current_task_id = task_id
            return
        if state in _TERMINAL_STATES:
            self.current_task_id = None
            return
        self.current_task_id = task_id


@dataclass(frozen=True)
class _A2AResult:
    text: str
    state: str | None
    status_text: str | None



def _parts_from_messages(messages: Sequence[PromptMessageExtended]) -> list[Part]:
    parts: list[Part] = []
    for message in messages:
        if message.role != "user":
            continue
        for content in message.content:
            if isinstance(content, TextContent):
                if content.text:
                    parts.append(Part(text=content.text))
                continue
            if isinstance(content, ImageContent | AudioContent):
                parts.append(
                    Part(
                        raw=base64.b64decode(content.data),
                        media_type=content.mimeType,
                    )
                )
                continue
            if isinstance(content, ResourceLink):
                parts.append(
                    Part(
                        url=str(content.uri),
                        media_type=content.mimeType or "",
                        filename=content.name,
                    )
                )
                continue
            if isinstance(content, EmbeddedResource):
                resource = content.resource
                if isinstance(resource, BlobResourceContents):
                    parts.append(
                        Part(
                            raw=base64.b64decode(resource.blob),
                            media_type=resource.mimeType or "",
                            filename=_filename_from_uri(str(resource.uri)),
                        )
                    )
                    continue
                if isinstance(resource, TextResourceContents):
                    data_part = _json_data_part(resource.text, media_type=resource.mimeType)
                    if data_part is not None:
                        parts.append(data_part)
                        continue
                    parts.append(
                        Part(
                            text=resource.text,
                            media_type=resource.mimeType or "text/plain",
                            filename=_filename_from_uri(str(resource.uri)),
                        )
                    )
    return parts


def _filename_from_uri(uri: str) -> str:
    path = PurePosixPath(uri.split("?", 1)[0])
    return path.name or "attachment"


def _parts_text(parts: Sequence[Part]) -> str:
    rendered: list[str] = []
    for part in parts:
        text = _part_text(part)
        if text:
            rendered.append(text)
    return "\n".join(rendered)


def _part_text(part: Part) -> str:
    if part.HasField("text"):
        return part.text
    if part.HasField("url"):
        label = part.filename or part.url
        suffix = f" ({part.media_type})" if part.media_type else ""
        return f"[{label}]({part.url}){suffix}"
    if part.HasField("data"):
        data = MessageToDict(part).get("data", {})
        return f"```json\n{json.dumps(data, indent=2, sort_keys=True)}\n```"
    if part.HasField("raw"):
        label = part.filename or "attachment"
        suffix = f" {part.media_type}" if part.media_type else ""
        return f"[{label}: {len(part.raw)} bytes{suffix}]"
    return ""


def _card_advertises_oauth(card: AgentCard) -> bool:
    if not card.security_schemes or not card.security_requirements:
        return False
    required_scheme_names = {
        scheme_name
        for requirement in card.security_requirements
        for scheme_name in requirement.schemes
    }
    for scheme_name in required_scheme_names:
        scheme = card.security_schemes.get(scheme_name)
        if scheme is None:
            continue
        if scheme.HasField("oauth2_security_scheme") or scheme.HasField(
            "open_id_connect_security_scheme"
        ):
            return True
    return False


def _latest_text(messages: Sequence[PromptMessageExtended]) -> str:
    for message in reversed(messages):
        text = message.all_text()
        if text.strip():
            return text
    return ""


def _append_text(chunks: list[str], text: str) -> None:
    if not text:
        return
    chunks.append(text)


def _artifact_key(artifact: Any) -> str:
    artifact_id = artifact.artifact_id
    if artifact_id:
        return artifact_id
    if artifact.name:
        return artifact.name
    return str(id(artifact))


def _replace_artifact_text(
    artifact_order: list[str],
    artifact_texts: dict[str, str],
    artifact: Any,
    text: str,
) -> None:
    if not text:
        return
    key = _artifact_key(artifact)
    if key not in artifact_texts:
        artifact_order.append(key)
    artifact_texts[key] = text


def _apply_artifact_update(
    artifact_order: list[str],
    artifact_texts: dict[str, str],
    artifact: Any,
    text: str,
    *,
    append: bool,
) -> None:
    key = _artifact_key(artifact)
    if key not in artifact_texts:
        artifact_order.append(key)
        artifact_texts[key] = text
        return
    if append:
        artifact_texts[key] = f"{artifact_texts[key]}{text}"
        return
    artifact_texts[key] = text


def _state_message(state: str | None) -> str:
    if not state:
        return "A2A task completed without text output."
    if state == "TASK_STATE_COMPLETED":
        return "A2A task completed without text output."
    return "A2A task ended without text output."


def _json_data_part(text: str, *, media_type: str | None) -> Part | None:
    if media_type != "application/json":
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    part = Part(media_type=media_type)
    ParseDict(data, part.data)
    return part
