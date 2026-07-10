from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from a2a.server.agent_execution import AgentExecutor
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
    HTTPAuthSecurityScheme,
    Message,
    Part,
    Role,
    SecurityRequirement,
    SecurityScheme,
    StringList,
    Task,
    TaskState,
    TaskStatus,
)
from fastapi import FastAPI

from fast_agent import AgentHarness, AgentRequest, FastAgent, RequestParams
from fast_agent.a2a.server import (
    A2A_HF_BEARER_SCHEME,
    A2ABearerAuthMiddleware,
    A2AServerCallContextBuilder,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from a2a.server.agent_execution import RequestContext
    from a2a.server.events import EventQueue
    from mcp.types import ContentBlock

    from fast_agent.tools.execution_environment import ShellEnvironment
    from fast_agent.types import AgentResponse

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8002"))

QUICK_REFINER_AGENT = "research_refiner"
RESEARCH_WORKER_AGENT = "research_worker"
FAST_AGENT_HOME = Path(__file__).with_name(".fast-agent")
RESEARCH_WORKSPACE = "/workspace"
RESEARCH_PROGRESS_HEARTBEAT_SECONDS = 2.0

fast = FastAgent(
    "fast-agent research A2A server",
    home=FAST_AGENT_HOME,
    parse_cli_args=False,
    quiet=True,
)


class ResearchDecisionKind(StrEnum):
    NEEDS_REFINEMENT = "needs_refinement"
    BEGIN_RESEARCH = "begin_research"


@dataclass(frozen=True, slots=True)
class ResearchDecision:
    kind: ResearchDecisionKind
    message: str
    goal: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                **asdict(self),
                "kind": self.kind.value,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, value: str) -> "ResearchDecision":
        data = json.loads(value)
        return cls(
            kind=ResearchDecisionKind(data["kind"]),
            message=str(data["message"]),
            goal=data.get("goal") if isinstance(data.get("goal"), str) else None,
        )


@dataclass(frozen=True, slots=True)
class ResearchRuntimeInfo:
    """Execution environment attached to one accepted research task."""

    environment_label: str
    bucket: str | None = None
    bucket_path: str | None = None

    def status_text(self) -> str:
        if self.bucket is None:
            return f"Research execution environment ready: {self.environment_label}."
        return (
            "Research execution environment ready: "
            f"{self.environment_label}; bucket {self.bucket}/{self.bucket_path or ''}."
        )


class ResearchRuntime:
    """Runtime used to invoke the research worker."""

    def __init__(self, harness: AgentHarness, info: ResearchRuntimeInfo) -> None:
        self._harness = harness
        self.info = info

    async def research(
        self,
        context: RequestContext,
        decision: ResearchDecision,
        *,
        progress_handler: "A2ATaskProgressHandler | None" = None,
    ) -> AgentResponse:
        session_id = _session_id(context)
        request_params = _research_request_params(progress_handler)
        return await self._harness.invoke(
            AgentRequest.text(
                decision.goal or context.get_user_input().strip(),
                agent=RESEARCH_WORKER_AGENT,
                session_id=session_id,
                params=request_params,
                metadata={
                    "transport": "a2a",
                    "phase": "research_task",
                    "a2a_task_id": context.task_id or "",
                    "research_environment": self.info.environment_label,
                    "research_bucket": self.info.bucket or "",
                    "research_bucket_path": self.info.bucket_path or "",
                },
            )
        )


class A2ATaskProgressHandler:
    """Forward fast-agent loop/tool progress to A2A task status updates."""

    def __init__(self, updater: TaskUpdater) -> None:
        self._updater = updater
        self._tool_labels: dict[str, str] = {}
        self._counter = 0

    async def on_tool_start(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict | None,
        tool_use_id: str | None = None,
    ) -> str:
        del arguments
        self._counter += 1
        tool_call_id = tool_use_id or f"a2a-progress-{self._counter}"
        self._tool_labels[tool_call_id] = _progress_label(tool_name, server_name)
        await self.report(f"{self._tool_labels[tool_call_id]} started")
        return tool_call_id

    async def on_tool_progress(
        self,
        tool_call_id: str,
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        label = self._tool_labels.get(tool_call_id, "Research step")
        suffix = message or _progress_counter_text(progress, total)
        await self.report(f"{label}: {suffix}" if suffix else label)

    async def on_tool_complete(
        self,
        tool_call_id: str,
        success: bool,
        content: "list[ContentBlock] | None",
        error: str | None,
    ) -> None:
        del content
        label = self._tool_labels.pop(tool_call_id, "Research step")
        if success:
            await self.report(f"{label} completed")
            return
        detail = f": {error}" if error else ""
        await self.report(f"{label} failed{detail}")

    async def on_tool_permission_denied(
        self,
        tool_name: str,
        server_name: str,
        tool_use_id: str | None,
        error: str | None = None,
    ) -> None:
        del tool_use_id
        label = _progress_label(tool_name, server_name)
        detail = f": {error}" if error else ""
        await self.report(f"{label} permission denied{detail}")

    async def get_tool_call_id_for_tool_use(self, tool_use_id: str) -> str | None:
        if tool_use_id in self._tool_labels:
            return tool_use_id
        return None

    async def ensure_tool_call_exists(
        self,
        tool_call_id: str,
        tool_name: str,
        server_name: str,
        arguments: dict | None = None,
    ) -> str:
        del arguments
        if tool_call_id not in self._tool_labels:
            self._tool_labels[tool_call_id] = _progress_label(tool_name, server_name)
        return tool_call_id

    async def report(self, message: str) -> None:
        await self._updater.update_status(
            TaskState.TASK_STATE_WORKING,
            message=self._updater.new_agent_message(parts=[Part(text=message)]),
        )


@dataclass(frozen=True, slots=True)
class HuggingFaceResearchEnvironmentConfig:
    """Configuration for per-task Hugging Face Sandbox research execution."""

    bucket: str
    image: str = "python:3.12"
    flavor: str = "cpu-basic"
    token: str | None = None
    namespace: str | None = None
    forward_hf_token: bool = False
    create_bucket: bool = True
    private_bucket: bool | None = None


class ResearchRuntimeFactory:
    """Create the execution runtime for an accepted research task."""

    def __init__(
        self,
        shared_harness: AgentHarness,
        *,
        hf_config: HuggingFaceResearchEnvironmentConfig | None = None,
        fast_agent_factory: Callable[[], FastAgent] | None = None,
        environment_factory: Callable[
            [HuggingFaceResearchEnvironmentConfig, str, str], ShellEnvironment
        ]
        | None = None,
    ) -> None:
        self._shared_harness = shared_harness
        self._hf_config = hf_config
        self._fast_agent_factory = fast_agent_factory or _new_research_fast_agent
        self._environment_factory = environment_factory or _huggingface_environment_for_task

    @asynccontextmanager
    async def open(
        self,
        context: RequestContext,
    ) -> AsyncIterator[ResearchRuntime]:
        context_id = context.context_id or str(uuid.uuid4())
        task_id = context.task_id or str(uuid.uuid4())
        if self._hf_config is None:
            yield ResearchRuntime(
                self._shared_harness,
                ResearchRuntimeInfo(environment_label="shared harness environment"),
            )
            return

        await _ensure_huggingface_bucket(self._hf_config)
        bucket_path = _research_bucket_path(context_id=context_id, task_id=task_id)
        environment = self._environment_factory(self._hf_config, task_id, bucket_path)
        research_fast = self._fast_agent_factory()
        async with research_fast.harness(environment=environment) as research_harness:
            yield ResearchRuntime(
                research_harness,
                ResearchRuntimeInfo(
                    environment_label="Hugging Face Sandbox",
                    bucket=self._hf_config.bucket,
                    bucket_path=bucket_path,
                ),
            )


class ResearchA2AHarnessAdapter:
    """A2A adapter that routes protocol requests through the fast-agent harness."""

    def __init__(
        self,
        harness: AgentHarness,
        *,
        runtime_factory: ResearchRuntimeFactory | None = None,
    ) -> None:
        self._harness = harness
        self._runtime_factory = runtime_factory or ResearchRuntimeFactory(
            harness,
            hf_config=_huggingface_research_environment_config_from_env(),
        )

    async def refine(self, context: RequestContext) -> ResearchDecision:
        prompt = context.get_user_input().strip()
        session_id = _session_id(context)
        response = await self._harness.invoke(
            AgentRequest.text(
                prompt,
                agent=QUICK_REFINER_AGENT,
                session_id=session_id,
                metadata={
                    "transport": "a2a",
                    "phase": "research_refinement",
                    "a2a_task_id": context.task_id or "",
                },
            )
        )
        return _coerce_research_decision(response.text_content(), prompt)

    def research_runtime(self, context: RequestContext) -> AsyncIterator[ResearchRuntime]:
        return self._runtime_factory.open(context)


class ResearchA2AExecutor(AgentExecutor):
    """A2A executor for a research intake flow with Message-or-Task exits."""

    def __init__(self, adapter: ResearchA2AHarnessAdapter) -> None:
        self._adapter = adapter
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.message or not context.context_id:
            return

        decision = await self._adapter.refine(context)
        if decision.kind == ResearchDecisionKind.NEEDS_REFINEMENT:
            await request_further_refinement(event_queue, context, decision)
            return

        if not context.task_id:
            return

        task = asyncio.current_task()
        if task is not None:
            self._running_tasks[context.task_id] = task
        try:
            await begin_research(event_queue, context, self._adapter, decision)
        finally:
            self._running_tasks.pop(context.task_id, None)

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


async def request_further_refinement(
    event_queue: EventQueue,
    context: RequestContext,
    decision: ResearchDecision,
) -> None:
    """Exit path that returns a standalone A2A Message without starting a task."""
    await event_queue.enqueue_event(
        Message(
            role=Role.ROLE_AGENT,
            context_id=context.context_id,
            message_id=str(uuid.uuid4()),
            parts=[Part(text=decision.message)],
        )
    )


async def begin_research(
    event_queue: EventQueue,
    context: RequestContext,
    adapter: ResearchA2AHarnessAdapter,
    decision: ResearchDecision,
) -> None:
    """Exit path that starts an A2A task and streams progress artifacts."""
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
        message=updater.new_agent_message(parts=[Part(text=decision.message)])
    )
    await updater.add_artifact(
        parts=[Part(text="Research goal accepted.\n")],
        name="progress",
        artifact_id=f"{context.task_id}:progress",
        last_chunk=False,
    )
    await updater.add_artifact(
        parts=[Part(text="Planning source search and synthesis steps.\n")],
        name="progress",
        artifact_id=f"{context.task_id}:progress",
        append=True,
        last_chunk=False,
    )
    progress_handler = A2ATaskProgressHandler(updater)
    async with adapter.research_runtime(context) as runtime:
        await updater.add_artifact(
            parts=[Part(text=f"{runtime.info.status_text()}\n")],
            name="progress",
            artifact_id=f"{context.task_id}:progress",
            append=True,
            last_chunk=False,
        )
        response = await _run_research_with_progress(
            runtime,
            context,
            decision,
            progress_handler=progress_handler,
        )
    await updater.add_artifact(
        parts=[Part(text=response.text_content())],
        name="response",
        append=False,
        last_chunk=True,
    )
    await updater.complete()


async def main() -> None:
    card = agent_card(host=HOST, port=PORT)
    async with fast.harness() as harness:
        request_handler = DefaultRequestHandler(
            agent_executor=ResearchA2AExecutor(ResearchA2AHarnessAdapter(harness)),
            task_store=InMemoryTaskStore(),
            agent_card=card,
        )
        context_builder = A2AServerCallContextBuilder()
        app = FastAPI(title=card.name)
        app.routes.extend(create_agent_card_routes(agent_card=card))
        app.routes.extend(
            create_jsonrpc_routes(
                request_handler=request_handler,
                rpc_url="/a2a/jsonrpc",
                context_builder=context_builder,
            )
        )
        app.routes.extend(
            create_rest_routes(
                request_handler=request_handler,
                path_prefix="/a2a/rest",
                context_builder=context_builder,
            )
        )
        if _serve_hf_oauth_enabled():
            app.add_middleware(A2ABearerAuthMiddleware, provider="huggingface")

        server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning"))
        await server.serve()


def agent_card(*, host: str, port: int) -> AgentCard:
    base_url = (
        os.environ.get("FAST_AGENT_PUBLIC_URL")
        or os.environ.get("FAST_AGENT_OAUTH_RESOURCE_URL")
        or f"http://{_url_host(host)}:{port}"
    )
    security_requirements = _security_requirements() if _serve_hf_oauth_enabled() else []
    return AgentCard(
        name="research-a2a",
        description="Research intake agent that refines requests before starting tasks.",
        provider=AgentProvider(organization="fast-agent", url="https://fast-agent.ai"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        security_schemes=_security_schemes() if _serve_hf_oauth_enabled() else {},
        security_requirements=security_requirements,
        skills=[
            AgentSkill(
                id="research",
                name="research",
                description="Refine a research request, then run it as an A2A task.",
                tags=["fast-agent", "research", "a2a"],
                examples=[
                    "Compare A2A task lifecycle patterns for agent developers as a short markdown report."
                ],
                input_modes=["text/plain"],
                output_modes=["text/plain"],
                security_requirements=security_requirements,
            )
        ],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url=f"{base_url.rstrip('/')}/a2a/jsonrpc",
            ),
            AgentInterface(
                protocol_binding="HTTP+JSON",
                protocol_version="1.0",
                url=f"{base_url.rstrip('/')}/a2a/rest",
            ),
        ],
    )


def _decision_for_prompt(prompt: str) -> ResearchDecision:
    missing = _missing_research_fields(prompt)
    if missing:
        subject = _subject_preview(prompt)
        return ResearchDecision(
            kind=ResearchDecisionKind.NEEDS_REFINEMENT,
            message=(
                f"I can turn {subject} into a research task, but I need: "
                f"{', '.join(missing)}. Please resend as plain text with those details."
            ),
        )
    return ResearchDecision(
        kind=ResearchDecisionKind.BEGIN_RESEARCH,
        message="Research task accepted",
        goal=prompt,
    )


def _session_id(context: RequestContext) -> str | None:
    return context.context_id


def _coerce_research_decision(text: str, prompt: str) -> ResearchDecision:
    try:
        return ResearchDecision.from_json(_json_payload(text))
    except (json.JSONDecodeError, KeyError, ValueError):
        fallback = _decision_for_prompt(prompt)
        if fallback.kind == ResearchDecisionKind.NEEDS_REFINEMENT and text.strip():
            return ResearchDecision(
                kind=ResearchDecisionKind.NEEDS_REFINEMENT,
                message=text.strip(),
            )
        return fallback


def _json_payload(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped.removeprefix("```json").removesuffix("```").strip()
    elif stripped.startswith("```"):
        stripped = stripped.removeprefix("```").removesuffix("```").strip()
    return stripped


def _missing_research_fields(prompt: str) -> list[str]:
    text = " ".join(prompt.lower().split())
    missing: list[str] = []
    if not _has_research_goal(text):
        missing.append("a concrete research goal")
    if not _has_audience(text):
        missing.append("the intended audience")
    if not _has_output_format(text):
        missing.append("the desired output format")
    return missing


def _has_research_goal(text: str) -> bool:
    research_terms = (
        "analyze",
        "compare",
        "evaluate",
        "explain",
        "find",
        "investigate",
        "research",
        "review",
        "survey",
        "summarize",
    )
    topic_words = [
        word.strip(".,:;!?()[]{}")
        for word in text.split()
        if word.strip(".,:;!?()[]{}") not in {"goal", "audience", "output", "format"}
    ]
    return any(term in text for term in research_terms) and len(topic_words) >= 4


def _has_audience(text: str) -> bool:
    if "audience" in text:
        return True
    audience_terms = (
        "developers",
        "engineers",
        "executives",
        "clinicians",
        "policymakers",
        "practitioners",
        "researchers",
        "scientists",
        "students",
    )
    return any(f"for {term}" in text or f"to {term}" in text for term in audience_terms)


def _has_output_format(text: str) -> bool:
    output_terms = (
        "brief",
        "html",
        "markdown",
        "memo",
        "pdf",
        "report",
        "slides",
        "summary",
        "table",
        "web page",
    )
    return any(term in text for term in output_terms)


def _subject_preview(prompt: str) -> str:
    stripped = " ".join(prompt.split())
    if not stripped:
        return "that request"
    if len(stripped) > 80:
        stripped = f"{stripped[:77]}..."
    return f"'{stripped}'"


def _serve_hf_oauth_enabled() -> bool:
    return os.environ.get("FAST_AGENT_SERVE_OAUTH", "").strip().lower() in {
        "hf",
        "huggingface",
    }


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


def _url_host(bind_host: str) -> str:
    if bind_host in {"0.0.0.0", "::", ""}:
        return "localhost"
    if ":" in bind_host and not bind_host.startswith("["):
        return f"[{bind_host}]"
    return bind_host


async def _run_research_with_progress(
    runtime: ResearchRuntime,
    context: RequestContext,
    decision: ResearchDecision,
    *,
    progress_handler: A2ATaskProgressHandler,
    heartbeat_seconds: float = RESEARCH_PROGRESS_HEARTBEAT_SECONDS,
) -> AgentResponse:
    research_task = asyncio.create_task(
        runtime.research(context, decision, progress_handler=progress_handler)
    )
    heartbeat_count = 0
    while not research_task.done():
        try:
            return await asyncio.wait_for(asyncio.shield(research_task), timeout=heartbeat_seconds)
        except TimeoutError:
            heartbeat_count += 1
            await progress_handler.report(
                f"Research still running ({heartbeat_count * int(heartbeat_seconds)}s)"
            )
    return await research_task


def _research_request_params(
    progress_handler: A2ATaskProgressHandler | None,
) -> RequestParams | None:
    if progress_handler is None:
        return None
    return RequestParams(
        emit_loop_progress=True,
        tool_execution_handler=progress_handler,
    )


def _progress_label(tool_name: str, server_name: str) -> str:
    if tool_name == "agent_loop":
        return f"{server_name} agent loop"
    return f"{server_name}/{tool_name}"


def _progress_counter_text(progress: float, total: float | None) -> str:
    if total is None:
        return f"step {progress:.0f}"
    return f"step {progress:.0f}/{total:.0f}"


def _new_research_fast_agent() -> FastAgent:
    return FastAgent(
        "fast-agent research worker",
        home=FAST_AGENT_HOME,
        parse_cli_args=False,
        quiet=True,
    )


def _huggingface_research_environment_config_from_env() -> (
    HuggingFaceResearchEnvironmentConfig | None
):
    bucket = os.environ.get("FAST_AGENT_RESEARCH_HF_BUCKET", "").strip()
    if not bucket:
        return None
    return HuggingFaceResearchEnvironmentConfig(
        bucket=bucket,
        image=os.environ.get("FAST_AGENT_RESEARCH_HF_IMAGE", "python:3.12").strip()
        or "python:3.12",
        flavor=os.environ.get("FAST_AGENT_RESEARCH_HF_FLAVOR", "cpu-basic").strip()
        or "cpu-basic",
        token=os.environ.get("FAST_AGENT_RESEARCH_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or None,
        namespace=os.environ.get("FAST_AGENT_RESEARCH_HF_NAMESPACE") or None,
        forward_hf_token=_truthy_env("FAST_AGENT_RESEARCH_HF_FORWARD_TOKEN"),
        create_bucket=not _falsey_env("FAST_AGENT_RESEARCH_HF_CREATE_BUCKET"),
        private_bucket=_optional_bool_env("FAST_AGENT_RESEARCH_HF_BUCKET_PRIVATE"),
    )


def _huggingface_environment_for_task(
    config: HuggingFaceResearchEnvironmentConfig,
    task_id: str,
    bucket_path: str,
) -> ShellEnvironment:
    del task_id
    from fast_agent.tools.huggingface_sandbox_environment import (
        HuggingFaceBucketMount,
        HuggingFaceSandboxEnvironment,
    )

    return HuggingFaceSandboxEnvironment(
        image=config.image,
        flavor=config.flavor,
        cwd=RESEARCH_WORKSPACE,
        bucket_mounts=(
            HuggingFaceBucketMount(
                source=config.bucket,
                mount_path=RESEARCH_WORKSPACE,
                read_only=False,
                path=bucket_path,
            ),
        ),
        token=config.token,
        namespace=config.namespace,
        forward_hf_token=config.forward_hf_token,
    )


async def _ensure_huggingface_bucket(config: HuggingFaceResearchEnvironmentConfig) -> None:
    if not config.create_bucket:
        return
    await asyncio.to_thread(_create_huggingface_bucket, config)


def _create_huggingface_bucket(config: HuggingFaceResearchEnvironmentConfig) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=config.token)
    api.create_bucket(
        config.bucket,
        private=config.private_bucket,
        exist_ok=True,
        token=config.token,
    )


def _research_bucket_path(*, context_id: str, task_id: str) -> str:
    return f"a2a-research/{_slug_id(context_id)}/{_slug_id(task_id)}"


def _slug_id(value: str) -> str:
    slug = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    return slug.strip("-") or "session"


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _falsey_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"0", "false", "no", "off"}


def _optional_bool_env(name: str) -> bool | None:
    value = os.environ.get(name, "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


if __name__ == "__main__":
    asyncio.run(main())
