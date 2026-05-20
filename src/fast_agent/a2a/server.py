"""Expose fast-agent agents through A2A HTTP transports."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from importlib.metadata import version as get_version
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import uvicorn
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes, create_rest_routes
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Message,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from fastapi import FastAPI
from google.protobuf.json_format import MessageToDict
from mcp.types import ImageContent, ResourceLink, TextContent
from pydantic import AnyUrl

from fast_agent.core.default_agent import agent_is_default, resolve_default_agent_name
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.types import LlmStopReason, PromptMessageExtended, RequestParams

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from a2a.server.agent_execution.context import RequestContext
    from a2a.server.events.event_queue import EventQueue

    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.llm.stream_types import StreamChunk


@runtime_checkable
class _StreamListenerCapable(Protocol):
    def add_stream_listener(self, listener: Any) -> Any:
        """Register a text stream listener."""


logger = get_logger(__name__)


def _fast_agent_version() -> str:
    for package_name in ("fast-agent-mcp", "fast-agent"):
        with contextlib.suppress(Exception):
            return get_version(package_name)
    return "unknown"


class FastAgentA2AExecutor(AgentExecutor):
    """A2A executor that routes tasks into fast-agent agents."""

    def __init__(
        self,
        primary_instance: AgentInstance,
        create_instance: Callable[[], Awaitable[AgentInstance]],
        dispose_instance: Callable[[AgentInstance], Awaitable[None]],
        *,
        primary_agent_name: str,
    ) -> None:
        self._primary_instance = primary_instance
        self._create_instance = create_instance
        self._dispose_instance = dispose_instance
        self._primary_agent_name = primary_agent_name
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

        await event_queue.enqueue_event(
            Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message],
            )
        )

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.start_work(
            message=updater.new_agent_message(parts=[Part(text="fast-agent is working")])
        )

        lock = await self._context_lock(context.context_id)
        async with lock:
            instance = await self._instance_for_context(context.context_id)
            agent = self._select_agent(instance, context.message)
            stream_context = self._prepare_streaming_context(
                agent=agent,
                updater=updater,
            )
            try:
                response = await agent.generate(
                    _prompt_from_a2a_message(context.message),
                    request_params=RequestParams(use_history=True),
                )
            except ProviderKeyError as exc:
                await updater.requires_auth(
                    message=updater.new_agent_message(parts=[Part(text=exc.message)])
                )
                return
            except asyncio.CancelledError:
                await updater.cancel()
                raise
            except Exception as exc:
                await updater.failed(
                    message=updater.new_agent_message(parts=[Part(text=str(exc))])
                )
                return
            finally:
                await self._cleanup_streaming_context(stream_context)

        streamed_text = stream_context.streamed_text()
        response_text = response.all_text()
        if response.stop_reason == LlmStopReason.PAUSE:
            await updater.requires_input(
                message=updater.new_agent_message(parts=_parts_from_prompt_message(response))
            )
            return

        if streamed_text:
            if response_text and response_text != streamed_text:
                await updater.add_artifact(
                    parts=_parts_from_prompt_message(response),
                    artifact_id=stream_context.artifact_id,
                    name="response",
                    append=False,
                    last_chunk=True,
                )
        else:
            await updater.add_artifact(
                parts=_parts_from_prompt_message(response),
                name="response",
                append=False,
                last_chunk=True,
            )
        await updater.complete()

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
        if not isinstance(agent, _StreamListenerCapable):
            return stream_context
        stream_context.start()

        def on_stream_chunk(chunk: StreamChunk) -> None:
            if not chunk.text or chunk.is_reasoning:
                return
            stream_context.record_chunk(chunk.text)

        stream_context.remove_listener = agent.add_stream_listener(on_stream_chunk)
        return stream_context

    async def _cleanup_streaming_context(self, stream_context: "_A2AStreamingContext") -> None:
        if stream_context.remove_listener is not None:
            stream_context.remove_listener()
        await stream_context.drain()
        if stream_context.tasks:
            await asyncio.gather(*stream_context.tasks, return_exceptions=True)

    async def _context_lock(self, context_id: str) -> asyncio.Lock:
        async with self._lock:
            lock = self._context_locks.get(context_id)
            if lock is None:
                lock = asyncio.Lock()
                self._context_locks[context_id] = lock
            return lock

    async def _instance_for_context(self, context_id: str) -> AgentInstance:
        instance = self._context_instances.get(context_id)
        if instance is not None:
            return instance
        instance = await self._create_instance()
        self._context_instances[context_id] = instance
        return instance

    def _select_agent(self, instance: AgentInstance, message: Message) -> AgentProtocol:
        agent_name = _requested_agent_name(message)
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
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self._host = host
        self._port = port
        self._primary_agent_name = _select_primary_agent(primary_instance)
        self.agent_card = _build_agent_card(
            primary_instance=primary_instance,
            server_name=server_name,
            server_description=server_description,
            host=host,
            port=port,
        )
        self.executor = FastAgentA2AExecutor(
            primary_instance=primary_instance,
            create_instance=create_instance,
            dispose_instance=dispose_instance,
            primary_agent_name=self._primary_agent_name,
        )
        self.request_handler = DefaultRequestHandler(
            agent_executor=self.executor,
            task_store=InMemoryTaskStore(),
            agent_card=self.agent_card,
        )

    def asgi_app(self) -> FastAPI:
        app = FastAPI(title=self.agent_card.name)
        app.routes.extend(create_agent_card_routes(agent_card=self.agent_card))
        app.routes.extend(
            create_jsonrpc_routes(request_handler=self.request_handler, rpc_url="/a2a/jsonrpc")
        )
        app.routes.extend(
            create_rest_routes(request_handler=self.request_handler, path_prefix="/a2a/rest")
        )
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
) -> AgentCard:
    base_url = f"http://{host}:{port}"
    skills = [
        AgentSkill(
            id=agent_name,
            name=agent_name,
            description=f"Send a message to the {agent_name} fast-agent agent.",
            tags=["fast-agent"],
            examples=["Hello"],
            input_modes=["text", "file", "image"],
            output_modes=["text", "file", "image", "task-status"],
        )
        for agent_name in primary_instance.agents
    ]
    return AgentCard(
        name=server_name,
        description=server_description or "A fast-agent A2A server.",
        provider=AgentProvider(organization="fast-agent", url="https://fast-agent.ai"),
        version=_fast_agent_version(),
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text", "file", "image"],
        default_output_modes=["text", "file", "image", "task-status"],
        skills=skills,
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
        return [TextContent(type="text", text=f"[{label}: {len(part.raw)} bytes]")]
    if part.HasField("data"):
        data = MessageToDict(part).get("data", {})
        return [TextContent(type="text", text=json.dumps(data, indent=2, sort_keys=True))]
    return []


def _parts_from_prompt_message(message: PromptMessageExtended) -> list[Part]:
    parts: list[Part] = []
    for content in message.content:
        if isinstance(content, TextContent):
            parts.append(Part(text=content.text))
            continue
        if isinstance(content, ImageContent):
            parts.append(
                Part(raw=base64.b64decode(content.data), media_type=content.mimeType)
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
    if not parts:
        parts.append(Part(text=message.all_text()))
    return parts
