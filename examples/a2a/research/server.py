from __future__ import annotations

import asyncio
import json
import os
import uuid
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

from fast_agent import AgentHarness, AgentRequest, FastAgent
from fast_agent.a2a.server import (
    A2A_HF_BEARER_SCHEME,
    A2ABearerAuthMiddleware,
    A2AServerCallContextBuilder,
)

if TYPE_CHECKING:
    from a2a.server.agent_execution import RequestContext
    from a2a.server.events import EventQueue

    from fast_agent.types import AgentResponse

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8002"))

QUICK_REFINER_AGENT = "research_refiner"
RESEARCH_WORKER_AGENT = "research_worker"
ENVIRONMENT_DIR = Path(__file__).with_name(".fast-agent")

fast = FastAgent(
    "fast-agent research A2A server",
    environment_dir=ENVIRONMENT_DIR,
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


class ResearchA2AHarnessAdapter:
    """A2A adapter that routes protocol requests through the fast-agent harness."""

    def __init__(self, harness: AgentHarness) -> None:
        self._harness = harness

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

    async def research(self, context: RequestContext, decision: ResearchDecision) -> AgentResponse:
        session_id = _session_id(context)
        return await self._harness.invoke(
            AgentRequest.text(
                decision.goal or context.get_user_input().strip(),
                agent=RESEARCH_WORKER_AGENT,
                session_id=session_id,
                metadata={
                    "transport": "a2a",
                    "phase": "research_task",
                    "a2a_task_id": context.task_id or "",
                },
            )
        )


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
    response = await adapter.research(context, decision)
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


if __name__ == "__main__":
    asyncio.run(main())
