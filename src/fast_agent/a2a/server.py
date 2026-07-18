"""Expose fast-agent agents through A2A HTTP transports."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import json
import os
from importlib.metadata import version as get_version
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.parse import quote

import uvicorn
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.request_handlers.response_helpers import agent_card_to_dict
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes, create_rest_routes
from a2a.server.routes.common import DefaultServerCallContextBuilder
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    HTTPAuthSecurityScheme,
    Message,
    Part,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from fastapi import FastAPI
from google.protobuf.json_format import MessageToDict
from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
)
from pydantic import AnyUrl
from starlette.responses import JSONResponse

from fast_agent.a2a.content import part_from_content
from fast_agent.a2a.task_api import (
    _current_task_returned_message,
    _current_task_started,
    _ensure_current_task_started,
    _reset_current_task,
    _return_message_parts,
    _set_current_task,
)
from fast_agent.a2a.task_api import return_artifact as a2a_return_artifact
from fast_agent.a2a.task_api import return_message as a2a_return_message
from fast_agent.a2a.task_api import start_task as a2a_start_task
from fast_agent.core.default_agent import agent_is_default, resolve_default_agent_name
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.auth.context import request_bearer_token
from fast_agent.mcp.auth.huggingface import HuggingFaceOAuthOrHubTokenVerifier
from fast_agent.tools.function_tool_loader import build_default_function_tool
from fast_agent.types import (
    AgentAuth,
    AgentRequest,
    AgentResponse,
    LlmStopReason,
    PromptMessageExtended,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from a2a.server.agent_execution.context import RequestContext
    from a2a.server.events.event_queue import EventQueue
    from a2a.server.tasks import PushNotificationConfigStore, PushNotificationSender
    from starlette.requests import Request
    from starlette.types import ASGIApp, Receive, Scope, Send

    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.interfaces import AgentProtocol


@runtime_checkable
class _FunctionToolAttachable(Protocol):
    def add_tool(self, tool: Any, *, replace: bool = True) -> None:
        """Attach a local function tool."""


@runtime_checkable
class _A2ADeferredTaskStartCapable(Protocol):
    a2a_defer_task_start: bool


logger = get_logger(__name__)

A2A_INPUT_MODES = ["text/plain", "application/json", "application/octet-stream", "image/*"]
A2A_OUTPUT_MODES = ["text/plain", "application/json", "application/octet-stream", "image/*"]
A2A_HF_BEARER_SCHEME = "hf_bearer"
DEFAULT_A2A_HOST = "127.0.0.1"


def _fast_agent_version() -> str:
    for package_name in ("fast-agent-mcp", "fast-agent"):
        with contextlib.suppress(Exception):
            return get_version(package_name)
    return "unknown"


def _get_a2a_oauth_provider() -> str | None:
    oauth_provider = os.environ.get("FAST_AGENT_SERVE_OAUTH", "").lower()
    if oauth_provider in {"hf", "huggingface"}:
        return "huggingface"
    if not oauth_provider:
        return None
    return oauth_provider


def _bearer_token_from_header(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    prefix = "bearer "
    if stripped.lower().startswith(prefix):
        token = stripped[len(prefix) :].strip()
        return token or None
    return None


def _bearer_token_from_call_context(context: RequestContext) -> str | None:
    saved_token = context.call_context.state.get("fast_agent_bearer_token")
    if isinstance(saved_token, str) and saved_token:
        return saved_token

    headers = context.call_context.state.get("headers")
    if not isinstance(headers, dict):
        return None
    authorization = headers.get("authorization") or headers.get("Authorization")
    token = _bearer_token_from_header(authorization if isinstance(authorization, str) else None)
    if token is not None:
        return token
    hf_authorization = headers.get("x-hf-authorization") or headers.get("X-HF-Authorization")
    return _bearer_token_from_header(
        hf_authorization if isinstance(hf_authorization, str) else None
    )


def _a2a_agent_auth(context: RequestContext) -> AgentAuth | None:
    token = _bearer_token_from_call_context(context)
    return AgentAuth.bearer(token) if token is not None else None


class A2ABearerAuthMiddleware:
    """Require bearer authentication for A2A action routes."""

    def __init__(self, app: ASGIApp, *, provider: str) -> None:
        self.app = app
        self.provider = provider
        self.token_verifier = HuggingFaceOAuthOrHubTokenVerifier()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        if not path.startswith("/a2a/"):
            await self.app(scope, receive, send)
            return

        headers = list(scope.get("headers", []))
        authorization = _header_value(headers, b"authorization")
        hf_authorization = _header_value(headers, b"x-hf-authorization")
        if authorization is None and hf_authorization is not None:
            authorization = hf_authorization
            headers.append((b"authorization", hf_authorization.encode("latin-1")))
            scope = dict(scope, headers=headers)

        token = _bearer_token_from_header(authorization)
        access_token = await self.token_verifier.verify_token(token or "")
        if token is None or access_token is None:
            response = JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={
                    "WWW-Authenticate": (
                        f'Bearer realm="fast-agent-a2a", '
                        f'error="invalid_token", provider="{self.provider}"'
                    )
                },
            )
            await response(scope, receive, send)
            return

        state = dict(scope.get("state") or {})
        state["fast_agent_bearer_token"] = token
        scope = dict(scope, state=state)
        await self.app(scope, receive, send)


def _header_value(headers: list[tuple[bytes, bytes]], name: bytes) -> str | None:
    for key, value in headers:
        if key.lower() == name:
            return value.decode("latin-1")
    return None


class A2AServerCallContextBuilder(DefaultServerCallContextBuilder):
    """Build A2A call context while preserving fast-agent request auth state."""

    def build(self, request: Request) -> Any:
        context = super().build(request)
        token = getattr(request.state, "fast_agent_bearer_token", None)
        if isinstance(token, str) and token:
            context.state["fast_agent_bearer_token"] = token
        return context


class FastAgentA2AExecutor(AgentExecutor):
    """A2A executor that routes tasks into fast-agent agents."""

    def __init__(
        self,
        primary_instance: AgentInstance,
        create_instance: Callable[[], Awaitable[AgentInstance]],
        dispose_instance: Callable[[AgentInstance], Awaitable[None]],
        *,
        primary_agent_name: str,
        instance_scope: str = "connection",
    ) -> None:
        self._primary_instance = primary_instance
        self._create_instance = create_instance
        self._dispose_instance = dispose_instance
        self._primary_agent_name = primary_agent_name
        self._instance_scope = instance_scope
        self._context_instances: dict[str, AgentInstance] = {}
        self._context_locks: dict[str, asyncio.Lock] = {}
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        running_task = self._running_tasks.get(task_id)
        if running_task is not None:
            running_task.cancel()
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context.context_id or "",
        )
        await updater.cancel()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.message or not context.task_id or not context.context_id:
            return

        task = asyncio.current_task()
        if task is not None:
            self._running_tasks[context.task_id] = task
        try:
            await self._execute(context, event_queue)
        finally:
            self._running_tasks.pop(context.task_id, None)

    async def _execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        assert context.message is not None
        assert context.task_id is not None
        assert context.context_id is not None

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )

        lock = await self._context_lock(self._lock_key(context))
        saved_bearer_token = request_bearer_token.set(_bearer_token_from_call_context(context))
        saved_a2a_task = _set_current_task(
            updater,
            event_queue=event_queue,
            request_message=context.message,
        )
        try:
            async with lock:
                instance: AgentInstance | None = None
                try:
                    instance = await self._acquire_instance(context.context_id)
                    session = _A2AInvokeSession(
                        executor=self,
                        instance=instance,
                        message=context.message,
                        updater=updater,
                    )
                    try:
                        response = await session.invoke(
                            AgentRequest.from_message(
                                _prompt_from_a2a_message(context.message),
                                agent=_requested_agent_name(context.message),
                                session_id=context.context_id,
                                auth=_a2a_agent_auth(context),
                                metadata={
                                    "transport": "a2a",
                                    "task_id": context.task_id,
                                    "context_id": context.context_id,
                                },
                            )
                        )
                    except ProviderKeyError as exc:
                        await _start_a2a_task()
                        await updater.requires_auth(
                            message=updater.new_agent_message(parts=[Part(text=exc.message)])
                        )
                        return
                    except asyncio.CancelledError:
                        await updater.cancel()
                        raise
                    except Exception as exc:
                        await _start_a2a_task()
                        await updater.failed(
                            message=updater.new_agent_message(parts=[Part(text=str(exc))])
                        )
                        return
                    finally:
                        if session.stream_context is not None:
                            await self._cleanup_streaming_context(session.stream_context)
                    returned_message = _current_task_returned_message()
                finally:
                    if instance is not None:
                        await self._release_instance(
                            context.context_id,
                            instance,
                        )

            if returned_message:
                return

            stream_context = session.stream_context
            streamed_text = stream_context.streamed_text() if stream_context is not None else ""
            response_message = response.message
            response_text = response_message.all_text()
            if response_message.stop_reason == LlmStopReason.PAUSE:
                await _start_a2a_task()
                await updater.requires_input(
                    message=updater.new_agent_message(
                        parts=_parts_from_prompt_message(response_message)
                    )
                )
                return

            if not _current_task_started():
                await _return_message_parts(_parts_from_prompt_message(response_message))
                return

            if streamed_text:
                if response_text and response_text != streamed_text:
                    assert stream_context is not None
                    await _start_a2a_task()
                    await updater.add_artifact(
                        parts=_parts_from_prompt_message(response_message),
                        artifact_id=stream_context.artifact_id,
                        name="response",
                        append=False,
                        last_chunk=True,
                    )
            else:
                await _start_a2a_task()
                await updater.add_artifact(
                    parts=_parts_from_prompt_message(response_message),
                    name="response",
                    append=False,
                    last_chunk=True,
                )
            await updater.complete()
        finally:
            _reset_current_task(saved_a2a_task)
            request_bearer_token.reset(saved_bearer_token)

    def _prepare_streaming_context(
        self,
        *,
        agent: AgentProtocol,
        updater: TaskUpdater,
    ) -> "_A2AStreamingContext":
        stream_context = _A2AStreamingContext(
            updater=updater,
            artifact_id=f"{updater.task_id}:response",
        )
        return stream_context

    async def _cleanup_streaming_context(self, stream_context: "_A2AStreamingContext") -> None:
        if stream_context.remove_listener is not None:
            stream_context.remove_listener()
        await stream_context.drain()
        if stream_context.tasks:
            await asyncio.gather(*stream_context.tasks, return_exceptions=True)

    def _lock_key(self, context: RequestContext) -> str:
        if self._instance_scope == "shared":
            return "__shared__"
        if self._instance_scope == "request":
            return context.task_id or context.context_id or "__request__"
        return context.context_id or "__context__"

    async def _context_lock(self, lock_key: str) -> asyncio.Lock:
        async with self._lock:
            lock = self._context_locks.get(lock_key)
            if lock is None:
                lock = asyncio.Lock()
                self._context_locks[lock_key] = lock
            return lock

    async def _acquire_instance(self, context_id: str) -> AgentInstance:
        if self._instance_scope == "shared":
            return self._primary_instance
        if self._instance_scope == "request":
            return await self._create_instance()
        instance = self._context_instances.get(context_id)
        if instance is not None:
            return instance
        instance = await self._create_instance()
        self._context_instances[context_id] = instance
        return instance

    async def _release_instance(self, context_id: str, instance: AgentInstance) -> None:
        del context_id
        if self._instance_scope == "request":
            await self._dispose_instance(instance)

    def _select_agent(
        self,
        instance: AgentInstance,
        message: Message,
        requested_agent: str | None = None,
    ) -> AgentProtocol:
        agent_name = requested_agent or _requested_agent_name(message)
        if agent_name and agent_name in instance.agents:
            return instance.agents[agent_name]
        if self._primary_agent_name in instance.agents:
            return instance.agents[self._primary_agent_name]
        return instance.app._agent(None)

    async def shutdown(self) -> None:
        for task in list(self._running_tasks.values()):
            task.cancel()
        for instance in list(self._context_instances.values()):
            await self._dispose_instance(instance)
        self._context_instances.clear()
        self._context_locks.clear()


class _A2AInvokeSession:
    """A2A protocol adapter over one acquired AgentInstance."""

    def __init__(
        self,
        *,
        executor: FastAgentA2AExecutor,
        instance: AgentInstance,
        message: Message,
        updater: TaskUpdater,
    ) -> None:
        self._executor = executor
        self._instance = instance
        self._message = message
        self._updater = updater
        self.stream_context: _A2AStreamingContext | None = None

    @property
    def agent_app(self):
        return self._instance.app

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        agent = self._executor._select_agent(self._instance, self._message, request.agent)
        _attach_a2a_task_tools(agent)
        if not _agent_defers_a2a_task_start(agent):
            await a2a_start_task()
        self.stream_context = self._executor._prepare_streaming_context(
            agent=agent,
            updater=self._updater,
        )
        message = await agent.generate(request.message, request_params=request.params)
        return AgentResponse(message=message, metadata=dict(request.metadata))


def _attach_a2a_task_tools(agent: AgentProtocol) -> None:
    if not isinstance(agent, _FunctionToolAttachable):
        return
    agent.add_tool(
        build_default_function_tool(
            _a2a_start_task_tool,
            name="start_task",
            description="Publish an A2A working status update for the current task.",
        )
    )
    agent.add_tool(
        build_default_function_tool(
            _a2a_return_artifact_tool,
            name="return_artifact",
            description="Publish a text artifact update for the current A2A task.",
        )
    )
    agent.add_tool(
        build_default_function_tool(
            _a2a_return_message_tool,
            name="return_message",
            description="Return a standalone A2A message without starting a task.",
        )
    )


def _agent_defers_a2a_task_start(agent: AgentProtocol) -> bool:
    if isinstance(agent, _FunctionToolAttachable):
        return True
    if isinstance(agent, _A2ADeferredTaskStartCapable):
        return agent.a2a_defer_task_start
    return False


async def _a2a_start_task_tool(message: str = "fast-agent is working") -> dict[str, str]:
    handle = await a2a_start_task(message)
    return {"task_id": handle.task_id, "context_id": handle.context_id}


async def _a2a_return_artifact_tool(
    text: str,
    name: str = "response",
    artifact_id: str | None = None,
    append: bool = False,
    last_chunk: bool = True,
) -> dict[str, str]:
    handle = await a2a_return_artifact(
        text,
        name=name,
        artifact_id=artifact_id,
        append=append,
        last_chunk=last_chunk,
    )
    return {"task_id": handle.task_id, "context_id": handle.context_id}


async def _a2a_return_message_tool(text: str) -> dict[str, str]:
    handle = await a2a_return_message(text)
    return {"task_id": handle.task_id, "context_id": handle.context_id}


async def _start_a2a_task() -> None:
    await _ensure_current_task_started()


class _A2AStreamingContext:
    def __init__(self, *, updater: TaskUpdater, artifact_id: str) -> None:
        self.updater = updater
        self.artifact_id = artifact_id
        self.remove_listener: Callable[[], None] | None = None
        self.tasks: list[asyncio.Task[None]] = []
        self._queue: asyncio.Queue[tuple[str, bool] | None] = asyncio.Queue()
        self._chunks: list[str] = []

    def start(self) -> None:
        self.tasks.append(asyncio.create_task(self._publish_chunks()))

    def record_chunk(self, text: str) -> None:
        append = bool(self._chunks)
        self._chunks.append(text)
        self._queue.put_nowait((text, append))

    def streamed_text(self) -> str:
        return "".join(self._chunks)

    async def _publish_chunks(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            text, append = item
            try:
                await _ensure_current_task_started()
                await self.updater.add_artifact(
                    parts=[Part(text=text)],
                    artifact_id=self.artifact_id,
                    name="response",
                    append=append,
                    last_chunk=False,
                )
            except Exception:
                logger.warning("Failed to publish A2A streaming artifact update", exc_info=True)
            finally:
                self._queue.task_done()

    async def drain(self) -> None:
        await self._queue.join()
        if self.tasks:
            self._queue.put_nowait(None)


class AgentA2AServer:
    """Expose fast-agent as an A2A server over JSON-RPC and HTTP+JSON."""

    def __init__(
        self,
        primary_instance: AgentInstance,
        create_instance: Callable[[], Awaitable[AgentInstance]],
        dispose_instance: Callable[[AgentInstance], Awaitable[None]],
        *,
        server_name: str = "fast-agent-a2a",
        server_description: str | None = None,
        host: str = DEFAULT_A2A_HOST,
        port: int = 8000,
        instance_scope: str = "connection",
        push_config_store: PushNotificationConfigStore | None = None,
        push_sender: PushNotificationSender | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._oauth_provider = _get_a2a_oauth_provider()
        self._primary_agent_name = _select_primary_agent(primary_instance)
        push_notifications_enabled = (
            push_config_store is not None and push_sender is not None
        )
        self.agent_card = _build_agent_card(
            primary_instance=primary_instance,
            server_name=server_name,
            server_description=server_description,
            host=host,
            port=port,
            auth_enabled=self._oauth_provider == "huggingface",
            push_notifications=push_notifications_enabled,
        )
        self.executor = FastAgentA2AExecutor(
            primary_instance=primary_instance,
            create_instance=create_instance,
            dispose_instance=dispose_instance,
            primary_agent_name=self._primary_agent_name,
            instance_scope=instance_scope,
        )
        self.request_handler = DefaultRequestHandler(
            agent_executor=self.executor,
            task_store=InMemoryTaskStore(),
            agent_card=self.agent_card,
            push_config_store=push_config_store,
            push_sender=push_sender,
        )

    def asgi_app(self) -> FastAPI:
        app = FastAPI(title=self.agent_card.name)
        context_builder = A2AServerCallContextBuilder()
        app.routes.extend(_agent_card_routes(self.agent_card, host=self._host, port=self._port))
        app.routes.extend(
            create_jsonrpc_routes(
                request_handler=self.request_handler,
                rpc_url="/a2a/jsonrpc",
                context_builder=context_builder,
            )
        )
        app.routes.extend(
            create_rest_routes(
                request_handler=self.request_handler,
                path_prefix="/a2a/rest",
                context_builder=context_builder,
            )
        )
        if self._oauth_provider == "huggingface":
            app.add_middleware(A2ABearerAuthMiddleware, provider=self._oauth_provider)
        return app

    async def run_async(self, *, host: str | None = None, port: int | None = None) -> None:
        server = uvicorn.Server(
            uvicorn.Config(
                self.asgi_app(),
                host=host or self._host,
                port=port or self._port,
                log_level="warning",
            )
        )
        try:
            await server.serve()
        finally:
            await self.executor.shutdown()


def _select_primary_agent(primary_instance: AgentInstance) -> str:
    selected = resolve_default_agent_name(
        primary_instance.agents,
        is_default=lambda _name, agent: agent_is_default(agent),
        is_tool_only=lambda _name, _agent: False,
    )
    if selected is not None:
        return selected
    return next(iter(primary_instance.agents))


def _build_agent_card(
    *,
    primary_instance: AgentInstance,
    server_name: str,
    server_description: str | None,
    host: str,
    port: int,
    auth_enabled: bool = False,
    push_notifications: bool = False,
) -> AgentCard:
    base_url = _base_url(host=host, port=port)
    security_requirements = _security_requirements() if auth_enabled else []
    skills = [
        _agent_skill_from_fast_agent(
            agent_name,
            agent,
            security_requirements=security_requirements,
        )
        for agent_name, agent in primary_instance.agents.items()
    ]
    return AgentCard(
        name=server_name,
        description=server_description or "A fast-agent A2A server.",
        provider=AgentProvider(organization="fast-agent", url="https://fast-agent.ai"),
        version=_fast_agent_version(),
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=push_notifications,
        ),
        default_input_modes=A2A_INPUT_MODES,
        default_output_modes=A2A_OUTPUT_MODES,
        skills=skills,
        security_schemes=_security_schemes() if auth_enabled else {},
        security_requirements=security_requirements,
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url=f"{base_url}/a2a/jsonrpc",
            ),
            AgentInterface(
                protocol_binding="HTTP+JSON",
                protocol_version="1.0",
                url=f"{base_url}/a2a/rest",
            ),
        ],
    )


def _agent_card_routes(agent_card: AgentCard, *, host: str, port: int) -> list[Any]:
    if not _is_wildcard_host(host):
        return create_agent_card_routes(agent_card=agent_card)

    from starlette.routing import Route

    async def _get_agent_card(request: "Request") -> JSONResponse:
        base_url = (
            os.environ.get("FAST_AGENT_PUBLIC_URL")
            or os.environ.get("FAST_AGENT_OAUTH_RESOURCE_URL")
            or str(request.base_url)
        ).rstrip("/")
        return JSONResponse(agent_card_to_dict(_agent_card_with_base_url(agent_card, base_url)))

    return [
        Route("/.well-known/agent-card.json", endpoint=_get_agent_card, methods=["GET"]),
    ]


def _agent_card_with_base_url(agent_card: AgentCard, base_url: str) -> AgentCard:
    card = copy.deepcopy(agent_card)
    for interface in card.supported_interfaces:
        if interface.protocol_binding == "JSONRPC":
            interface.url = f"{base_url}/a2a/jsonrpc"
        if interface.protocol_binding == "HTTP+JSON":
            interface.url = f"{base_url}/a2a/rest"
    return card


def _base_url(*, host: str, port: int) -> str:
    return f"http://{_url_host(host)}:{port}"


def _url_host(bind_host: str) -> str:
    if _is_wildcard_host(bind_host):
        return "localhost"
    if ":" in bind_host and not bind_host.startswith("["):
        return f"[{bind_host}]"
    return bind_host


def _is_wildcard_host(bind_host: str) -> bool:
    return bind_host in {"0.0.0.0", "::", ""}


def _security_schemes() -> dict[str, SecurityScheme]:
    return {
        A2A_HF_BEARER_SCHEME: SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(
                scheme="bearer",
                bearer_format="HF_TOKEN",
                description="Hugging Face bearer token",
            )
        )
    }


def _security_requirements() -> list[SecurityRequirement]:
    return [SecurityRequirement(schemes={A2A_HF_BEARER_SCHEME: StringList(list=[])})]


def _agent_skill_from_fast_agent(
    agent_name: str,
    agent: AgentProtocol,
    *,
    security_requirements: list[SecurityRequirement] | None = None,
) -> AgentSkill:
    agent_type = str(agent.agent_type) if agent.agent_type else "agent"
    description = (
        agent.config.description or f"Send a message to the {agent_name} fast-agent agent."
    )
    return AgentSkill(
        id=agent_name,
        name=agent_name,
        description=description,
        tags=["fast-agent", agent_type],
        examples=["Hello"],
        input_modes=A2A_INPUT_MODES,
        output_modes=A2A_OUTPUT_MODES,
        security_requirements=security_requirements or [],
    )


def _requested_agent_name(message: Message) -> str | None:
    metadata = MessageToDict(message).get("metadata")
    if not isinstance(metadata, dict):
        return None
    requested = metadata.get("agent") or metadata.get("fast_agent_agent")
    return requested if isinstance(requested, str) and requested else None


def _prompt_from_a2a_message(message: Message) -> PromptMessageExtended:
    content: list[Any] = []
    for part in message.parts:
        content.extend(_content_from_part(part))
    if not content:
        content.append(TextContent(type="text", text=""))
    return PromptMessageExtended(role="user", content=content)


def _content_from_part(part: Part) -> list[Any]:
    if part.HasField("text"):
        return [TextContent(type="text", text=part.text)]
    if part.HasField("url"):
        label = part.filename or part.url
        try:
            return [
                ResourceLink(
                    type="resource_link",
                    name=label,
                    uri=AnyUrl(part.url),
                    mimeType=part.media_type or None,
                )
            ]
        except ValueError:
            return [TextContent(type="text", text=f"[{label}]({part.url})")]
    if part.HasField("raw"):
        data = base64.b64encode(part.raw).decode("ascii")
        if part.media_type.startswith("image/"):
            return [ImageContent(type="image", data=data, mimeType=part.media_type)]
        label = part.filename or "attachment"
        return [
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri=AnyUrl(f"attachment:///{quote(label)}"),
                    mimeType=part.media_type or "application/octet-stream",
                    blob=data,
                ),
            )
        ]
    if part.HasField("data"):
        data = MessageToDict(part).get("data", {})
        return [TextContent(type="text", text=json.dumps(data, indent=2, sort_keys=True))]
    return []


def _parts_from_prompt_message(message: PromptMessageExtended) -> list[Part]:
    parts = [
        part for content in message.content if (part := part_from_content(content)) is not None
    ]
    if not parts:
        parts.append(Part(text=message.all_text()))
    return parts
