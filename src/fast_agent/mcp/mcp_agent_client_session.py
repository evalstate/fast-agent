"""
A derived client session for the MCP Agent framework.
It adds logging and supports sampling requests.
"""

import asyncio
import json
import sys
from contextlib import suppress
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from mcp import ClientSession, ServerNotification
from mcp.shared.context import RequestContext
from mcp.shared.message import MessageMetadata
from mcp.shared.session import (
    ProgressFnT,
    ReceiveResultT,
)
from mcp.types import (
    URL_ELICITATION_REQUIRED,
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ClientCapabilities,
    ClientRequest,
    EmptyResult,
    GetPromptRequest,
    GetPromptRequestParams,
    GetPromptResult,
    Implementation,
    InitializeRequest,
    InitializeRequestParams,
    InitializeResult,
    ListRootsResult,
    PingRequest,
    ProgressNotification,
    ReadResourceRequest,
    ReadResourceRequestParams,
    ReadResourceResult,
    RequestParams,
    Result,
    Root,
    SamplingCapability,
    SamplingToolsCapability,
    ToolListChangedNotification,
)
from pydantic import AnyUrl, BaseModel, FileUrl

from fast_agent.context_dependent import ContextDependent
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.server_config_helpers import get_server_config
from fast_agent.mcp.sampling import resolve_auto_sampling_enabled, sample
from fast_agent.mcp.url_elicitation_required import (
    URLElicitationDisplayItem,
    URLElicitationRequiredDisplayPayload,
    build_url_elicitation_required_display_payload,
)
from fast_agent.utils.env import env_flag
from fast_agent.utils.text import strip_casefold, strip_str_to_none, strip_to_none

if TYPE_CHECKING:
    from mcp.client.session import ListRootsFnT, SamplingFnT

    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.transport_tracking import TransportChannelMetrics

logger = get_logger(__name__)


@runtime_checkable
class _ModelDumpParams(Protocol):
    def model_dump(
        self,
        *,
        by_alias: bool = False,
        exclude_none: bool = False,
    ) -> dict[str, Any]: ...


def _progress_trace_enabled() -> bool:
    return env_flag("FAST_AGENT_TRACE_MCP_PROGRESS")


def _progress_trace(message: str) -> None:
    if not _progress_trace_enabled():
        return
    print(f"[mcp-progress-trace] {message}", file=sys.stderr, flush=True)


_SESSIONS_CAPABILITY_KEY = "sessions"
_EXPERIMENTAL_SESSION_TEST_CAPABILITY_KEY = "experimental/sessions"

_EXPERIMENTAL_SESSION_META_KEY = "io.modelcontextprotocol/session"
_URL_ELICITATION_RESULT_PREFIX = "fast-agent-url-elicitation-required:"


class _SessionMetadata(BaseModel):
    sessionId: str
    expiresAt: str | None = None
    state: str | None = None


class _SessionCreateRequest(BaseModel):
    method: Literal["sessions/create"] = "sessions/create"
    params: RequestParams | None = None


class _SessionCreateResult(Result):
    session: _SessionMetadata | None = None


class _SessionDeleteRequest(BaseModel):
    method: Literal["sessions/delete"] = "sessions/delete"
    params: RequestParams | None = None


class _SessionDeleteResult(Result):
    deleted: bool | None = None


async def list_roots(context: RequestContext[ClientSession, None]) -> ListRootsResult:
    """List roots callback that will be called by the MCP library."""

    if (server_config := get_server_config(context.session)) and server_config.roots:
        roots = [
            Root(
                uri=FileUrl(
                    root.server_uri_alias or root.uri,
                ),
                name=root.name,
            )
            for root in server_config.roots
        ]
        return ListRootsResult(roots=roots)

    return ListRootsResult(roots=[])


class MCPAgentClientSession(ClientSession, ContextDependent):
    """
    MCP Agent framework acts as a client to the servers providing tools/resources/prompts for the agent workloads.
    This is a simple client session for those server connections, and supports
        - handling sampling requests
        - notifications
        - MCP root configuration

    Developers can extend this class to add more custom functionality as needed
    """

    _pending_url_elicitations: list[URLElicitationDisplayItem]

    def __init__(self, read_stream, write_stream, read_timeout=None, **kwargs) -> None:
        custom_elicitation_handler = self._pop_fast_agent_kwargs(kwargs)
        self._initialize_session_state()

        fast_agent = self._client_implementation()
        list_roots_cb = self._make_list_roots_callback()
        sampling_cb = self._make_sampling_callback()
        sampling_caps = self._make_sampling_capabilities(sampling_cb)
        elicitation_handler = self._resolve_elicitation_handler(custom_elicitation_handler)
        self.effective_elicitation_mode = self._resolve_effective_elicitation_mode(
            elicitation_handler
        )
        self._discard_managed_client_kwargs(kwargs)

        super().__init__(
            read_stream,
            write_stream,
            read_timeout,
            **kwargs,
            list_roots_callback=list_roots_cb,
            sampling_callback=sampling_cb,
            sampling_capabilities=sampling_caps,
            client_info=fast_agent,
            elicitation_callback=elicitation_handler,
        )

    def _pop_fast_agent_kwargs(self, kwargs: dict[str, Any]) -> Any:
        self.session_server_name = kwargs.pop("server_name", None)
        self._tool_list_changed_callback = kwargs.pop("tool_list_changed_callback", None)
        self._aggregator = kwargs.pop("aggregator", None)
        self.server_config: MCPServerSettings | None = kwargs.pop("server_config", None)
        self.agent_model: str | None = kwargs.pop("agent_model", None)
        self.agent_name: str | None = kwargs.pop("agent_name", None)
        self.api_key: str | None = kwargs.pop("api_key", None)
        self._context = kwargs.pop("context", None)
        self._transport_metrics: TransportChannelMetrics | None = kwargs.pop(
            "transport_metrics", None
        )
        return kwargs.pop("elicitation_handler", None)

    def _initialize_session_state(self) -> None:
        self.effective_elicitation_mode: str | None = "none"
        self._offline_notified = False
        self._pending_url_elicitations = []
        self._experimental_session_supported = False
        self._experimental_session_features: tuple[str, ...] = ()
        self._experimental_session_cookie: dict[str, Any] | None = None

    def _client_implementation(self) -> Implementation:
        from importlib.metadata import version

        if self.server_config and self.server_config.implementation:
            return self.server_config.implementation
        fast_agent_version = version("fast-agent-mcp") or "dev"
        return Implementation(name="fast-agent-mcp", version=fast_agent_version)

    def _make_list_roots_callback(self) -> "ListRootsFnT | None":
        if self.server_config and self.server_config.roots:
            return cast("ListRootsFnT", list_roots)
        return None

    def _make_sampling_callback(self) -> "SamplingFnT | None":
        if (self.server_config and self.server_config.sampling) or self._should_enable_auto_sampling():
            return cast("SamplingFnT", sample)
        return None

    @staticmethod
    def _make_sampling_capabilities(
        sampling_cb: "SamplingFnT | None",
    ) -> SamplingCapability | None:
        if sampling_cb is None:
            return None
        return SamplingCapability(tools=SamplingToolsCapability())

    def _resolve_elicitation_handler(self, custom_elicitation_handler: Any) -> Any | None:
        if custom_elicitation_handler is not None:
            return custom_elicitation_handler

        elicitation_handler = self._resolve_configured_elicitation_handler()
        if elicitation_handler is not None or self.server_config:
            return elicitation_handler

        from fast_agent.mcp.elicitation_handlers import forms_elicitation_handler

        return forms_elicitation_handler

    def _resolve_configured_elicitation_handler(self) -> Any | None:
        try:
            from fast_agent.agents.agent_types import AgentConfig
            from fast_agent.context import get_current_context
            from fast_agent.mcp.elicitation_factory import resolve_elicitation_handler

            context = get_current_context()
            if not context or not context.config:
                return None
            agent_config = AgentConfig(
                name=self.agent_name or "unknown",
                model=self.agent_model or "unknown",
                elicitation_handler=None,
            )
            return resolve_elicitation_handler(agent_config, context.config, self.server_config)
        except Exception:
            return None

    def _resolve_effective_elicitation_mode(self, elicitation_handler: Any | None) -> str:
        if self.server_config and self.server_config.elicitation is not None:
            return self.server_config.elicitation.mode or "forms"
        if elicitation_handler is None:
            return "none"
        return self._global_elicitation_mode() or "forms"

    @staticmethod
    def _global_elicitation_mode() -> str | None:
        from fast_agent.context import get_current_context
        from fast_agent.mcp.elicitation_factory import resolve_global_elicitation_mode

        context = get_current_context()
        if not context or not context.config:
            return None
        return resolve_global_elicitation_mode(context.config)

    @staticmethod
    def _discard_managed_client_kwargs(kwargs: dict[str, Any]) -> None:
        for key in (
            "list_roots_callback",
            "sampling_callback",
            "sampling_capabilities",
            "client_info",
            "elicitation_callback",
        ):
            kwargs.pop(key, None)

    @property
    def experimental_session_supported(self) -> bool:
        return self._experimental_session_supported

    @property
    def experimental_session_features(self) -> tuple[str, ...]:
        return self._experimental_session_features

    @property
    def experimental_session_cookie(self) -> dict[str, Any] | None:
        if self._experimental_session_cookie is None:
            return None
        return dict(self._experimental_session_cookie)

    @property
    def experimental_session_id(self) -> str | None:
        cookie = self._experimental_session_cookie
        if not cookie:
            return None
        return self._extract_session_id(cookie)

    @property
    def experimental_session_title(self) -> str | None:
        cookie = self._experimental_session_cookie
        if not cookie:
            return None
        return self._extract_session_title(cookie)

    def set_experimental_session_cookie(self, cookie: dict[str, Any] | None) -> None:
        """Override the in-memory MCP session metadata for this connection."""
        if cookie is None:
            self._experimental_session_cookie = None
            return
        self._experimental_session_cookie = self._normalize_experimental_session_cookie(cookie)

    async def experimental_session_create(
        self,
        *,
        title: str | None = None,
        data: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Create a new experimental data-layer session and return SessionMetadata."""
        del title, data

        request = _SessionCreateRequest()

        result = await self.send_request(
            cast("ClientRequest", request),
            _SessionCreateResult,
        )

        if result.session is not None:
            self._experimental_session_cookie = self._normalize_experimental_session_cookie(
                result.session.model_dump(exclude_none=True)
            )

        return self.experimental_session_cookie

    async def experimental_session_list(self) -> list[dict[str, Any]]:
        """Return the currently active session metadata as a one-item snapshot."""
        cookie = self.experimental_session_cookie
        if cookie is None:
            return []
        return [cookie]

    async def experimental_session_delete(self, session_id: str | None = None) -> bool:
        """Delete an experimental data-layer session."""
        resolved_session_id = (
            session_id.strip()
            if isinstance(session_id, str) and session_id.strip()
            else self.experimental_session_id
        )

        request_params: RequestParams | None = None
        if resolved_session_id:
            request_params = RequestParams.model_validate(
                {
                    "_meta": {
                        _EXPERIMENTAL_SESSION_META_KEY: {
                            "sessionId": resolved_session_id,
                        }
                    }
                }
            )

        request = _SessionDeleteRequest(params=request_params)
        result = await self.send_request(
            cast("ClientRequest", request),
            _SessionDeleteResult,
        )

        deleted = result.deleted
        if isinstance(deleted, bool):
            if deleted and resolved_session_id == self.experimental_session_id:
                self._experimental_session_cookie = None
            return deleted

        if resolved_session_id == self.experimental_session_id:
            self._experimental_session_cookie = None
        return True

    async def initialize(self) -> InitializeResult:
        result = await super().initialize()
        capabilities = getattr(result, "capabilities", None)
        self._capture_experimental_session_capability(capabilities)
        await self._maybe_establish_experimental_session()
        return result

    def _should_enable_auto_sampling(self) -> bool:
        """Check if auto_sampling is enabled at the application level."""
        try:
            from fast_agent.context import get_current_context

            return resolve_auto_sampling_enabled(get_current_context())
        except Exception:
            return True

    def _capture_experimental_session_capability(
        self,
        capabilities: Any,
    ) -> None:
        self._experimental_session_supported = False
        self._experimental_session_features = ()

        if capabilities is None:
            return

        # Draft data-layer sessions capability.
        sessions_capability = self._extract_sessions_capability(capabilities)
        if isinstance(sessions_capability, dict):
            self._experimental_session_supported = True
            self._experimental_session_features = ("create", "delete")
        return

    async def _maybe_establish_experimental_session(self) -> None:
        if not self._experimental_session_supported:
            return
        if self._experimental_session_cookie is not None:
            return
        if "create" not in self._experimental_session_features:
            return

        try:
            await self.experimental_session_create()
        except Exception as exc:
            logger.debug(
                "Failed to establish experimental MCP session",
                server=self.session_server_name,
                error=str(exc),
            )

    def _build_advertised_experimental_session_capability(self) -> dict[str, object] | None:
        cfg = self.server_config
        if cfg is None or not cfg.experimental_session_advertise:
            return None

        return {}

    def _maybe_advertise_experimental_session_capability(
        self, request: ClientRequest
    ) -> ClientRequest:
        advertised = self._build_advertised_experimental_session_capability()
        if advertised is None:
            return request

        root = getattr(request, "root", None)
        if not isinstance(root, InitializeRequest):
            return request

        params = root.params
        if params is None:
            return request

        capabilities = params.capabilities
        existing_experimental = capabilities.experimental
        experimental_payload: dict[str, dict[str, object]] = {}
        if isinstance(existing_experimental, dict):
            for name, value in existing_experimental.items():
                if isinstance(name, str) and isinstance(value, dict):
                    experimental_payload[name] = dict(value)

        if _EXPERIMENTAL_SESSION_TEST_CAPABILITY_KEY not in experimental_payload:
            experimental_payload[_EXPERIMENTAL_SESSION_TEST_CAPABILITY_KEY] = dict(advertised)
            capabilities = ClientCapabilities(
                roots=capabilities.roots,
                sampling=capabilities.sampling,
                elicitation=capabilities.elicitation,
                experimental=experimental_payload,
                tasks=capabilities.tasks,
            )
            params = InitializeRequestParams(
                protocolVersion=params.protocolVersion,
                capabilities=capabilities,
                clientInfo=params.clientInfo,
            )
            return ClientRequest(InitializeRequest(params=params))

        return request

    def _merge_experimental_session_meta(
        self, metadata: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        merged: dict[str, Any] = dict(metadata) if metadata else {}
        session_meta = self._session_meta_for_request(self._experimental_session_cookie)
        if session_meta and _EXPERIMENTAL_SESSION_META_KEY not in merged:
            merged[_EXPERIMENTAL_SESSION_META_KEY] = session_meta
        return merged or None

    def _update_experimental_session_cookie(
        self,
        metadata: dict[str, Any] | None,
        *,
        expected_session_id: str | None = None,
    ) -> None:
        if not isinstance(metadata, dict):
            return

        value = metadata.get(_EXPERIMENTAL_SESSION_META_KEY)
        if value is None and _EXPERIMENTAL_SESSION_META_KEY not in metadata:
            return

        if value is None:
            self._experimental_session_cookie = None
            return

        if isinstance(value, dict):
            normalized = self._normalize_experimental_session_cookie(value)
            if normalized is None:
                return

            response_session_id = self._extract_session_id(normalized)
            if expected_session_id and response_session_id != expected_session_id:
                logger.warning(
                    "Ignoring experimental session metadata with mismatched sessionId",
                    server=self.session_server_name,
                    expected_session_id=expected_session_id,
                    response_session_id=response_session_id,
                )
                return

            current = self._experimental_session_cookie
            current_session_id = (
                self._extract_session_id(current) if isinstance(current, dict) else None
            )

            if (
                isinstance(current, dict)
                and current_session_id
                and response_session_id == current_session_id
            ):
                merged = dict(current)
                merged.update(normalized)
                self._experimental_session_cookie = merged
                return

            self._experimental_session_cookie = normalized

    @staticmethod
    def _extract_session_id(cookie: dict[str, Any]) -> str | None:
        raw = cookie.get("sessionId")
        if isinstance(raw, str) and raw:
            return raw
        return None

    @staticmethod
    def _extract_session_title(cookie: dict[str, Any]) -> str | None:
        title = cookie.get("title")
        if isinstance(title, str):
            direct_title = strip_to_none(title)
            if direct_title is not None:
                return direct_title

        data = cookie.get("data")
        if not isinstance(data, dict):
            return None

        nested_title = data.get("title") or data.get("label")
        return strip_str_to_none(nested_title)

    @classmethod
    def _session_meta_for_request(
        cls,
        cookie: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(cookie, dict):
            return None

        session_id = cls._extract_session_id(cookie)
        if not session_id:
            return None

        payload: dict[str, Any] = {"sessionId": session_id}

        state = cookie.get("state")
        if isinstance(state, str) and state:
            payload["state"] = state

        return payload

    @staticmethod
    def _extract_request_session_id(request: ClientRequest) -> str | None:
        root = getattr(request, "root", None)
        params = getattr(root, "params", None)
        if params is None:
            return None

        payload: dict[str, Any] | None = None
        if isinstance(params, dict):
            payload = params
        elif isinstance(params, _ModelDumpParams):
            dumped = params.model_dump(by_alias=True, exclude_none=False)
            if isinstance(dumped, dict):
                payload = dumped

        if not isinstance(payload, dict):
            return None

        meta = payload.get("_meta")
        if not isinstance(meta, dict):
            return None

        session_meta = meta.get(_EXPERIMENTAL_SESSION_META_KEY)
        if not isinstance(session_meta, dict):
            return None

        raw = session_meta.get("sessionId")
        if isinstance(raw, str) and raw:
            return raw
        return None

    @classmethod
    def _normalize_experimental_session_cookie(
        cls,
        cookie: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(cookie, dict):
            return None

        normalized = dict(cookie)
        session_id = cls._extract_session_id(normalized)
        if not session_id:
            return None
        normalized["sessionId"] = session_id

        return normalized

    @staticmethod
    def _extract_sessions_capability(capabilities: Any) -> dict[str, Any] | None:
        if isinstance(capabilities, dict):
            raw = capabilities.get(_SESSIONS_CAPABILITY_KEY)
            return raw if isinstance(raw, dict) else None

        direct = getattr(capabilities, _SESSIONS_CAPABILITY_KEY, None)
        if isinstance(direct, dict):
            return direct

        model_extra = getattr(capabilities, "model_extra", None)
        if isinstance(model_extra, dict):
            raw = model_extra.get(_SESSIONS_CAPABILITY_KEY)
            if isinstance(raw, dict):
                return raw

        return None

    async def send_request(
        self,
        request: ClientRequest,
        result_type: type[ReceiveResultT],
        request_read_timeout_seconds: timedelta | None = None,
        metadata: MessageMetadata | None = None,
        progress_callback: ProgressFnT | None = None,
    ) -> ReceiveResultT:
        request = self._maybe_advertise_experimental_session_capability(request)
        logger.debug("send_request: request=", data=request.model_dump())
        expected_session_id = self._extract_request_session_id(request)
        request_id = getattr(self, "_request_id", None)
        is_ping_request = self._is_ping_request(request)
        request_method = getattr(getattr(request, "root", None), "method", "unknown")

        self._trace_request_progress(
            "outbound-request",
            request_method=request_method,
            request_id=request_id,
            progress_callback=progress_callback,
            extra=f" progress_token={request_id!r}",
        )
        self._register_ping_request(is_ping_request, request_id)
        try:
            result = await super().send_request(
                # NOTE: request must be positional due to an upstream bug in
                # opentelemetry-instrumentation-mcp (seen in 0.52.1) where the
                # wrapper expects args[0] and can return None when request is
                # only provided via kwargs.
                # TODO: revert to keyword argument once upstream handles kwargs.
                request,
                result_type=result_type,
                request_read_timeout_seconds=request_read_timeout_seconds,
                metadata=metadata,
                progress_callback=progress_callback,
            )
            self._handle_successful_request(
                result,
                expected_session_id=expected_session_id,
                is_ping_request=is_ping_request,
                request_id=request_id,
                request_method=request_method,
                progress_callback=progress_callback,
            )
            return result
        except Exception as e:
            self._handle_failed_request(
                e,
                is_ping_request=is_ping_request,
                request_id=request_id,
                request_method=request_method,
                progress_callback=progress_callback,
            )
            raise

    def _trace_request_progress(
        self,
        event: str,
        *,
        request_method: str,
        request_id: Any,
        progress_callback: ProgressFnT | None,
        extra: str = "",
    ) -> None:
        if progress_callback is None or request_id is None:
            return
        _progress_trace(
            f"{event} "
            f"server={self.session_server_name or 'unknown'} "
            f"method={request_method} "
            f"request_id={request_id!r}"
            f"{extra}"
        )

    def _register_ping_request(self, is_ping_request: bool, request_id: Any) -> None:
        if is_ping_request and request_id is not None and self._transport_metrics is not None:
            self._transport_metrics.register_ping_request(request_id)

    def _discard_ping_request(self, is_ping_request: bool, request_id: Any) -> None:
        if is_ping_request and request_id is not None and self._transport_metrics is not None:
            self._transport_metrics.discard_ping_request(request_id)

    def _handle_successful_request(
        self,
        result: ReceiveResultT,
        *,
        expected_session_id: str | None,
        is_ping_request: bool,
        request_id: Any,
        request_method: str,
        progress_callback: ProgressFnT | None,
    ) -> None:
        logger.debug(
            "send_request: response=",
            data=result.model_dump() if result is not None else "no response returned",
        )
        result_meta = getattr(result, "meta", None)
        if isinstance(result_meta, dict):
            self._update_experimental_session_cookie(
                result_meta,
                expected_session_id=expected_session_id,
            )

        self._trace_request_progress(
            "request-complete",
            request_method=request_method,
            request_id=request_id,
            progress_callback=progress_callback,
        )

        self._attach_transport_channel(request_id, result)
        self._attach_url_elicitation_payload_from_result(
            result,
            request_method=request_method,
        )
        self._attach_pending_url_elicitation_payload_for_request(
            result,
            request_method=request_method,
        )
        self._discard_ping_request(is_ping_request, request_id)
        self._offline_notified = False

    def _handle_failed_request(
        self,
        exc: Exception,
        *,
        is_ping_request: bool,
        request_id: Any,
        request_method: str,
        progress_callback: ProgressFnT | None,
    ) -> None:
        from anyio import ClosedResourceError

        from fast_agent.core.exceptions import ServerSessionTerminatedError

        self._discard_pending_url_elicitation_payload()
        self._trace_request_progress(
            "request-error",
            request_method=request_method,
            request_id=request_id,
            progress_callback=progress_callback,
            extra=f" error={type(exc).__name__}: {exc}",
        )
        self._discard_ping_request(is_ping_request, request_id)

        if self._is_session_terminated_error(exc):
            raise ServerSessionTerminatedError(
                server_name=self.session_server_name or "unknown",
                details="Server returned 404 - session may have expired due to server restart",
            ) from exc

        if self._is_url_elicitation_required_error(exc):
            self._attach_url_elicitation_required_payload(exc, request_method)

        if isinstance(exc, ClosedResourceError):
            self._raise_connection_offline(exc)

        logger.error(f"send_request failed: {exc!s}")

    def _raise_connection_offline(self, exc: Exception) -> None:
        if not self._offline_notified:
            from fast_agent.ui import console

            console.console.print(
                f"[dim red]MCP server {self.session_server_name} offline[/dim red]"
            )
            self._offline_notified = True
        raise ConnectionError(f"MCP server {self.session_server_name} offline") from exc

    @staticmethod
    def _is_ping_request(request: ClientRequest) -> bool:
        root = getattr(request, "root", None)
        method = getattr(root, "method", None)
        if not isinstance(method, str):
            return False
        method_lower = strip_casefold(method)
        return method_lower == "ping" or method_lower.endswith(("/ping", ".ping"))

    def _is_session_terminated_error(self, exc: Exception) -> bool:
        """Check if exception is a session terminated error (code 32600 from 404)."""
        from mcp.shared.exceptions import McpError

        from fast_agent.core.exceptions import ServerSessionTerminatedError

        if isinstance(exc, McpError):
            error_data = getattr(exc, "error", None)
            if error_data:
                code = getattr(error_data, "code", None)
                if code == ServerSessionTerminatedError.SESSION_TERMINATED_CODE:
                    return True
        return False

    def _is_url_elicitation_required_error(self, exc: Exception) -> bool:
        """Check if exception is URL elicitation required error (-32042)."""
        from mcp.shared.exceptions import McpError

        if not isinstance(exc, McpError):
            return False

        error_data = getattr(exc, "error", None)
        if error_data is None:
            return False

        return getattr(error_data, "code", None) == URL_ELICITATION_REQUIRED

    def _attach_url_elicitation_required_payload(self, exc: Exception, request_method: str) -> None:
        """Attach parsed URL elicitation data to exception for deferred display."""
        from mcp.shared.exceptions import McpError

        if not isinstance(exc, McpError):
            return

        error_data = getattr(exc, "error", None)
        if error_data is None:
            return

        server_name = self.session_server_name or "unknown"
        payload = build_url_elicitation_required_display_payload(
            error_data.data,
            server_name=server_name,
            request_method=request_method,
        )
        exc_meta = cast("Any", exc)
        exc_meta._fast_agent_url_elicitation_required = payload

    def queue_url_elicitation_for_active_request(
        self,
        *,
        message: str,
        url: str,
        elicitation_id: str | None,
    ) -> bool:
        """Queue URL elicitation for the next successful request result."""
        self._pending_url_elicitations.append(
            URLElicitationDisplayItem(
                message=message,
                url=url,
                elicitation_id=elicitation_id or "",
            )
        )
        return True

    def _attach_pending_url_elicitation_payload_for_request(
        self,
        result: object,
        *,
        request_method: str,
    ) -> None:
        payload = self._consume_pending_url_elicitation_payload(request_method=request_method)
        if payload is None:
            return
        with suppress(Exception):
            result_meta = cast("Any", result)
            result_meta._fast_agent_url_elicitation_required = payload

    def _consume_pending_url_elicitation_payload(
        self,
        *,
        request_method: str,
    ) -> URLElicitationRequiredDisplayPayload | None:
        if not self._pending_url_elicitations:
            return None

        items = self._pending_url_elicitations
        self._pending_url_elicitations = []
        return URLElicitationRequiredDisplayPayload(
            server_name=self.session_server_name or "unknown",
            request_method=request_method,
            elicitations=items,
            issues=[],
        )

    def _discard_pending_url_elicitation_payload(self) -> None:
        self._pending_url_elicitations = []

    def _attach_url_elicitation_payload_from_result(
        self,
        result: object,
        *,
        request_method: str,
    ) -> None:
        if not isinstance(result, CallToolResult) or not result.isError or not result.content:
            return

        first_block = result.content[0]
        text = getattr(first_block, "text", None)
        if not isinstance(text, str) or _URL_ELICITATION_RESULT_PREFIX not in text:
            return

        _, _, raw_payload = text.partition(_URL_ELICITATION_RESULT_PREFIX)
        try:
            payload_data = json.loads(raw_payload.strip())
        except json.JSONDecodeError:
            return
        if not isinstance(payload_data, dict):
            return

        payload = build_url_elicitation_required_display_payload(
            payload_data,
            server_name=self.session_server_name or "unknown",
            request_method=request_method,
        )
        with suppress(Exception):
            result_meta = cast("Any", result)
            result_meta._fast_agent_url_elicitation_required = payload

    @staticmethod
    def get_url_elicitation_required_payload(
        exc: object,
    ) -> URLElicitationRequiredDisplayPayload | None:
        """Return deferred URL elicitation display payload when present."""
        payload = getattr(exc, "_fast_agent_url_elicitation_required", None)
        if isinstance(payload, URLElicitationRequiredDisplayPayload):
            return payload
        return None

    def _attach_transport_channel(self, request_id, result) -> None:
        if self._transport_metrics is None or request_id is None or result is None:
            return
        channel = self._transport_metrics.consume_response_channel(request_id)
        if not channel:
            return
        with suppress(Exception):
            result_meta = cast("Any", result)
            result_meta.transport_channel = channel

    async def _received_notification(self, notification: ServerNotification) -> None:
        """
        Can be overridden by subclasses to handle a notification without needing
        to listen on the message stream.
        """
        logger.debug(
            "_received_notification: notification=",
            data=notification.model_dump(),
        )

        # Call parent notification handler first
        await super()._received_notification(notification)

        # Then process our specific notification types
        match notification.root:
            case ToolListChangedNotification():
                # Simple notification handling - just call the callback if it exists
                if self._tool_list_changed_callback and self.session_server_name:
                    logger.info(
                        f"Tool list changed for server '{self.session_server_name}', triggering callback"
                    )
                    asyncio.create_task(
                        self._handle_tool_list_change_callback(self.session_server_name)
                    )
                else:
                    logger.debug(
                        f"Tool list changed for server '{self.session_server_name}' but no callback registered"
                    )

        # Forward non-progress server notifications to the aggregator callback.
        # Progress updates already flow through the request progress callback path.
        _cb = getattr(self._aggregator, "server_notification_callback", None) if self._aggregator else None
        if _cb and not isinstance(notification.root, ProgressNotification):
            asyncio.create_task(self._handle_server_notification(notification))

    async def _handle_server_notification(self, notification: ServerNotification) -> None:
        """Forward server notifications to the registered callback."""
        _cb = getattr(self._aggregator, "server_notification_callback", None) if self._aggregator else None
        if not _cb:
            return
        try:
            await _cb(
                self.session_server_name or "unknown",
                notification,
            )
        except Exception as e:
            logger.warning(
                f"Error in server notification callback for '{self.session_server_name}': {e}"
            )

    async def _handle_tool_list_change_callback(self, server_name: str) -> None:
        """
        Helper method to handle tool list change callback in a separate task
        to prevent blocking the notification handler
        """
        try:
            await self._tool_list_changed_callback(server_name)
        except Exception as e:
            logger.error(f"Error in tool list changed callback: {e}")

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
        progress_callback: ProgressFnT | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Call a tool with optional metadata and progress callback support.

        Always uses our overridden send_request to ensure session terminated errors
        are properly detected and converted to ServerSessionTerminatedError.
        """
        merged_meta = self._merge_experimental_session_meta(meta)
        request_meta = RequestParams.Meta(**merged_meta) if merged_meta is not None else None
        return await self.send_request(
            ClientRequest(
                CallToolRequest(
                    params=CallToolRequestParams(
                        name=name,
                        arguments=arguments,
                        _meta=request_meta,
                    )
                )
            ),
            CallToolResult,
            request_read_timeout_seconds=read_timeout_seconds,
            progress_callback=progress_callback,
        )

    async def ping(self, read_timeout_seconds: timedelta | None = None) -> EmptyResult:
        """Send a ping request to check server liveness."""
        request = PingRequest(method="ping")
        return await self.send_request(
            ClientRequest(request),
            EmptyResult,
            request_read_timeout_seconds=read_timeout_seconds,
        )

    async def read_resource(
        self,
        uri: AnyUrl | str,
        *,
        meta: dict[str, Any] | RequestParams.Meta | None = None,
    ) -> ReadResourceResult:
        """Read a resource with optional metadata support.

        Always uses our overridden send_request to ensure session terminated errors
        are properly detected and converted to ServerSessionTerminatedError.
        """
        # Convert str to AnyUrl if needed
        uri_obj: AnyUrl = uri if isinstance(uri, AnyUrl) else AnyUrl(uri)

        # Always create request ourselves to ensure we go through our send_request override
        params = ReadResourceRequestParams(uri=uri_obj)

        supplied_meta = meta.model_dump() if isinstance(meta, RequestParams.Meta) else meta
        merged_meta = self._merge_experimental_session_meta(supplied_meta)
        if merged_meta:
            params = ReadResourceRequestParams(uri=uri_obj, _meta=RequestParams.Meta(**merged_meta))

        request = ReadResourceRequest(method="resources/read", params=params)
        return await self.send_request(ClientRequest(request), ReadResourceResult)

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
        *,
        meta: dict[str, Any] | RequestParams.Meta | None = None,
    ) -> GetPromptResult:
        """Get a prompt with optional metadata support.

        Always uses our overridden send_request to ensure session terminated errors
        are properly detected and converted to ServerSessionTerminatedError.
        """
        # Always create request ourselves to ensure we go through our send_request override
        params = GetPromptRequestParams(name=name, arguments=arguments)

        supplied_meta = meta.model_dump() if isinstance(meta, RequestParams.Meta) else meta
        merged_meta = self._merge_experimental_session_meta(supplied_meta)
        if merged_meta:
            params = GetPromptRequestParams(
                name=name,
                arguments=arguments,
                _meta=RequestParams.Meta(**merged_meta),
            )

        request = GetPromptRequest(method="prompts/get", params=params)
        return await self.send_request(ClientRequest(request), GetPromptResult)
