from __future__ import annotations

import shlex
import sys
from types import ModuleType
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent import AgentRequest, AgentResponse, AppOpenRequest, DefaultHarnessApp
from fast_agent.core.harness_app import HarnessAppContext, load_harness_app
from fast_agent.skills import SkillManifest
from fast_agent.tools.execution_environment import ShellExecutionResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from fast_agent.core.harness import AgentHarness, HarnessSession


class RecordingHarnessSession:
    def __init__(self, agent_app: object | None = None) -> None:
        self.agent_app = agent_app or object()
        self.session_manager = object()
        self.requests: list[AgentRequest] = []
        self.shell_calls: list[
            tuple[str, Path | str | None, Mapping[str, str] | None, float | None]
        ] = []

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse.text(f"agent={request.agent}")

    async def shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        self.shell_calls.append((command, cwd, env, timeout))
        return ShellExecutionResult(stdout="ok", stderr="", exit_code=0)


class RecordingHarness:
    def __init__(self, agent_app: object | None = None) -> None:
        self.session_instance = RecordingHarnessSession(agent_app=agent_app)
        self.opened: list[tuple[str | None, str | None]] = []

    async def session(
        self,
        session_id: str | None = None,
        *,
        agent_name: str | None = None,
    ) -> HarnessSession:
        self.opened.append((session_id, agent_name))
        return cast("HarnessSession", self.session_instance)


@pytest.mark.asyncio
async def test_default_harness_app_opens_session_and_exposes_agent_app() -> None:
    harness = RecordingHarness()
    app = DefaultHarnessApp(cast("AgentHarness", harness))

    async with app.open(AppOpenRequest(session_id="demo", agent="support")) as session:
        response = await session.invoke(AgentRequest.text("hello", agent="support"))

    assert harness.opened == [("demo", "support")]
    assert session.agent_app is harness.session_instance.agent_app
    assert session.env.session_manager is harness.session_instance.session_manager
    assert response.text_content() == "agent=support"


@pytest.mark.asyncio
async def test_runtime_agent_invoke_uses_scoped_agent_when_request_omits_agent() -> None:
    harness = RecordingHarness()
    app = DefaultHarnessApp(cast("AgentHarness", harness))

    async with app.open(AppOpenRequest(session_id="demo")) as session:
        response = await session.env.agent("support").invoke(AgentRequest.text("hello"))

    assert response.text_content() == "agent=support"
    assert harness.session_instance.requests[0].agent == "support"


@pytest.mark.asyncio
async def test_runtime_tools_execute_quotes_arguments_and_uses_session_shell() -> None:
    harness = RecordingHarness()
    app = DefaultHarnessApp(cast("AgentHarness", harness))

    async with app.open(AppOpenRequest(session_id="demo")) as session:
        result = await session.env.tools.execute(
            "python",
            args=["-c", "print('hello world')"],
            timeout=5,
        )

    assert result.stdout == "ok"
    expected_command = shlex.join(["python", "-c", "print('hello world')"])
    assert harness.session_instance.shell_calls == [(expected_command, None, None, 5)]


class SkillAgent:
    def __init__(self) -> None:
        self.skill_manifests: list[SkillManifest] = []

    def set_skill_manifests(self, manifests: "Sequence[SkillManifest]") -> None:
        self.skill_manifests = list(manifests)


class SkillAgentApp:
    def __init__(self, agent: SkillAgent) -> None:
        self.agent = agent

    def resolve_agent(self, name: str | None = None) -> SkillAgent:
        assert name in {None, "dev"}
        return self.agent


@pytest.mark.asyncio
async def test_runtime_skills_adds_manifests_to_session_agent(tmp_path: Path) -> None:
    first = SkillManifest(
        name="review",
        description="Review repositories",
        body="Use review procedures.",
        path=tmp_path / "review" / "SKILL.md",
    )
    duplicate = SkillManifest(
        name="Review",
        description="Duplicate review skill",
        body="Duplicate body.",
        path=tmp_path / "duplicate" / "SKILL.md",
    )
    second = SkillManifest(
        name="release",
        description="Release repositories",
        body="Use release procedures.",
        path=tmp_path / "release" / "SKILL.md",
    )
    agent = SkillAgent()
    harness = RecordingHarness(agent_app=SkillAgentApp(agent))
    app = DefaultHarnessApp(cast("AgentHarness", harness))

    async with app.open(AppOpenRequest(session_id="demo", agent="dev")) as session:
        applied = session.env.skills.add([first, duplicate, second], agent="dev")

    assert applied == [first, second]
    assert agent.skill_manifests == [first, second]


def test_load_harness_app_uses_entrypoint_factory() -> None:
    harness = RecordingHarness()
    module = ModuleType("test_harness_app_entrypoint")
    contexts: list[HarnessAppContext] = []

    def create_app(context: HarnessAppContext) -> object:
        contexts.append(context)
        return context.default_app

    setattr(module, "create_app", create_app)
    sys.modules[module.__name__] = module
    try:
        app = load_harness_app(
            session_provider=cast("AgentHarness", harness),
            entrypoint=f"{module.__name__}:create_app",
        )
    finally:
        sys.modules.pop(module.__name__, None)

    assert isinstance(app, DefaultHarnessApp)
    assert len(contexts) == 1
    assert isinstance(contexts[0].default_app, DefaultHarnessApp)


def test_load_harness_app_rejects_invalid_entrypoint_format() -> None:
    harness = RecordingHarness()

    with pytest.raises(ValueError, match="module:function"):
        load_harness_app(
            session_provider=cast("AgentHarness", harness),
            entrypoint="missing_function",
        )
