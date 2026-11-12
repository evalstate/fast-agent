"""Tests for instruction utility helpers."""

from __future__ import annotations

import asyncio
import os
import platform
from typing import TYPE_CHECKING

import pytest

from fast_agent import FastAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.workflow.iterative_planner import IterativePlanner
from fast_agent.agents.workflow.router_agent import RouterAgent
from fast_agent.constants import DEFAULT_AGENT_INSTRUCTION
from fast_agent.context import Context
from fast_agent.core.instruction_refresh import build_instruction
from fast_agent.core.instruction_utils import (
    apply_instruction_context,
    build_agent_instruction_context,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt_templates import (
    _format_client_info,
    _format_execution_environment,
    enrich_with_environment_context,
    load_skills_for_context,
    refresh_execution_environment_context,
)
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.tools.execution_environment import ShellRuntimeInfo
from fast_agent.tools.local_shell_executor import LocalEnvironment

if TYPE_CHECKING:
    from pathlib import Path


class StubAgent:
    """Minimal non-MCP agent used for instruction context tests."""

    def __init__(
        self,
        *,
        name: str,
        instruction_template: str,
        source_path: Path | None = None,
        agent_type: AgentType = AgentType.BASIC,
    ) -> None:
        self.name = name
        self._instruction = instruction_template
        self.instruction_template = instruction_template
        self.agent_type = agent_type
        self.config = AgentConfig(
            name=name,
            instruction=instruction_template,
            source_path=source_path,
            agent_type=agent_type,
        )

    @property
    def instruction(self) -> str:
        return self._instruction

    def set_instruction(self, instruction: str) -> None:
        self._instruction = instruction


class StubMcpAggregator:
    async def get_server_instructions(self) -> dict[str, tuple[str | None, list[str]]]:
        return {}


class StubMcpAgent(StubAgent):
    """MCP-capable test double for context propagation checks."""

    def __init__(
        self,
        *,
        name: str,
        instruction_template: str,
        source_path: Path | None = None,
        agent_type: AgentType = AgentType.BASIC,
    ) -> None:
        super().__init__(
            name=name,
            instruction_template=instruction_template,
            source_path=source_path,
            agent_type=agent_type,
        )
        self._instruction_context: dict[str, str] = {}
        self._skill_registry = None
        self._skill_manifests: list[object] = []
        self._aggregator = StubMcpAggregator()

    @property
    def instruction_context(self) -> dict[str, str]:
        return self._instruction_context

    @property
    def aggregator(self) -> StubMcpAggregator:
        return self._aggregator

    @property
    def skill_manifests(self) -> list[object]:
        return self._skill_manifests

    @property
    def skill_registry(self):
        return self._skill_registry

    @skill_registry.setter
    def skill_registry(self, value) -> None:
        self._skill_registry = value

    def set_skill_manifests(self, manifests) -> None:
        self._skill_manifests = list(manifests)

    def set_instruction_context(self, context: dict[str, str]) -> None:
        self._instruction_context.update(context)

    @property
    def skill_read_tool_name(self) -> str:
        return "read_skill"


class _PromptEnvironment:
    def __init__(self, kind: str = "remote") -> None:
        self._kind = kind

    @property
    def cwd(self) -> str:
        return "/workspace"

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(
            name="sh",
            kind=self._kind,
            provider="huggingface",
            environment_name="hf-gpu",
        )

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def execute(self, request, *, callbacks=None):
        del request, callbacks
        raise NotImplementedError


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_type", [RouterAgent, IterativePlanner])
@pytest.mark.parametrize("with_context", [False, True])
async def test_workflow_instructions_survive_context_refresh(
    workflow_type: type[RouterAgent] | type[IterativePlanner], with_context: bool
) -> None:
    worker = LlmAgent(AgentConfig("research_worker", description="Finds primary research sources"))
    context_template = "Workspace: {{workspaceRoot}}\n" if with_context else ""
    if workflow_type is RouterAgent:
        # Router routing_instruction uses str.format before context rendering.
        workflow = RouterAgent(
            AgentConfig("router"),
            [worker],
            routing_instruction=(
                "Workspace: {{{{workspaceRoot}}}}\n{context}" if with_context else None
            ),
        )
    else:
        workflow = IterativePlanner(
            AgentConfig("planner", instruction=context_template + "Workers:\n{{agents}}"),
            [worker],
        )
    original_config_instruction = workflow.config.instruction

    try:
        await workflow.initialize()
        initialized_instruction = workflow.instruction
        await apply_instruction_context([workflow], {})
        assert workflow.instruction == initialized_instruction

        llm = await workflow.attach_llm(ModelFactory.create_factory("passthrough"))
        for workspace in ("/first", "/second", "/second"):
            await apply_instruction_context([workflow], {"workspaceRoot": workspace})
            assert workflow.instruction.count("research_worker") == 1
            assert "Finds primary research sources" in workflow.instruction
            assert "{{agents}}" not in workflow.instruction
            assert llm.default_request_params.system_prompt == workflow.instruction
            assert workflow.config.instruction == original_config_instruction
            if with_context:
                assert llm.instruction == workflow.instruction
                assert f"Workspace: {workspace}" in workflow.instruction
                assert "{{workspaceRoot}}" not in workflow.instruction
    finally:
        await workflow.shutdown()


def test_format_client_info_ignores_blank_title_and_trims_version() -> None:
    assert (
        _format_client_info({"title": "   ", "name": " zed ", "version": " 1.2.3 "}) == "zed 1.2.3"
    )


def test_format_client_info_omits_blank_and_unknown_versions() -> None:
    assert _format_client_info({"name": "editor", "version": "   "}) == "editor"
    assert _format_client_info({"name": "editor", "version": "unknown"}) == "editor"


def test_format_client_info_normalizes_via_client() -> None:
    assert (
        _format_client_info(
            {
                "name": "fast-agent",
                "version": "1.0",
                "viaTitle": "   ",
                "viaName": " terminal ",
                "viaVersion": " 2.0 ",
            }
        )
        == "fast-agent 1.0 via terminal 2.0"
    )


def test_format_execution_environment_deduplicates_labels() -> None:
    assert (
        _format_execution_environment(
            name="local",
            kind="local",
            provider=None,
            shell_name="bash",
            cwd="/app",
        )
        == "local (shell: bash, cwd: /app)"
    )
    assert (
        _format_execution_environment(
            name="sandbox",
            kind="remote",
            provider="sandbox",
            shell_name="sh",
            cwd="/workspace",
        )
        == "sandbox remote (shell: sh, cwd: /workspace)"
    )


def test_enrich_environment_context_describes_active_execution_environment() -> None:
    context: dict[str, str] = {}

    enrich_with_environment_context(
        context,
        "/host/project",
        {"name": "fast-agent"},
        shell_environment=_PromptEnvironment(),
    )

    assert context["workspaceRoot"] == "/workspace"
    assert context["hostWorkspaceRoot"] == "/host/project"
    assert context["executionEnvironmentName"] == "hf-gpu"
    assert context["executionEnvironmentKind"] == "remote"
    assert context["executionEnvironmentProvider"] == "huggingface"
    assert context["executionEnvironmentShell"] == "sh"
    assert context["executionEnvironmentCwd"] == "/workspace"
    assert "Workspace root: /workspace" in context["env"]
    assert "/host/project" not in context["env"]
    assert (
        "Execution environment: hf-gpu remote huggingface (shell: sh, cwd: /workspace)"
        in context["env"]
    )
    assert (
        f"Fast-agent runtime: {platform.python_implementation()} {platform.python_version()} "
        "(host process)" in context["env"]
    )
    assert "Client host platform:" not in context["env"]
    assert "Host platform:" not in context["env"]


def test_refresh_execution_environment_context_preserves_client_summary() -> None:
    context: dict[str, str] = {}
    enrich_with_environment_context(context, "/host/project", {"name": "fast-agent"})

    refresh_execution_environment_context(context, _PromptEnvironment())

    assert "Client: fast-agent" in context["env"]
    assert "Execution environment: hf-gpu remote huggingface" in context["env"]
    assert "Workspace root: /workspace" in context["env"]
    assert "/host/project" not in context["env"]


def test_build_agent_instruction_context_includes_agent_metadata(tmp_path: Path) -> None:
    card_path = tmp_path / "cards" / "agent.md"
    agent = StubAgent(
        name="helper",
        instruction_template="Hello",
        source_path=card_path,
        agent_type=AgentType.BASIC,
    )

    context = build_agent_instruction_context(agent, {"workspaceRoot": "/workspace"})

    assert context["workspaceRoot"] == "/workspace"
    assert context["agentName"] == "helper"
    assert context["agentType"] == AgentType.BASIC.value
    assert context["agentCardPath"] == str(card_path.resolve())
    assert context["agentCardDir"] == str(card_path.parent.resolve())


def test_build_agent_instruction_context_uses_internal_when_no_card_path() -> None:
    agent = StubAgent(
        name="agent",
        instruction_template="Hello",
        source_path=None,
        agent_type=AgentType.BASIC,
    )

    context = build_agent_instruction_context(agent, {"workspaceRoot": "/workspace"})

    assert context["workspaceRoot"] == "/workspace"
    assert context["agentName"] == "agent"
    assert context["agentType"] == AgentType.BASIC.value
    assert context["agentCardPath"] == "(internal)"
    assert context["agentCardDir"] == "(internal)"


def test_apply_instruction_context_resolves_agent_metadata_placeholders(tmp_path: Path) -> None:
    card_path = tmp_path / "agents" / "reviewer.md"
    agent = StubAgent(
        name="reviewer",
        instruction_template=(
            "Name={{agentName}} Type={{agentType}} "
            "Card={{agentCardPath}} Dir={{agentCardDir}} Root={{workspaceRoot}}"
        ),
        source_path=card_path,
        agent_type=AgentType.BASIC,
    )

    asyncio.run(apply_instruction_context([agent], {"workspaceRoot": "/workspace"}))

    assert "{{agentName}}" not in agent.instruction
    assert "{{agentCardPath}}" not in agent.instruction
    assert "Name=reviewer" in agent.instruction
    assert f"Card={card_path.resolve()}" in agent.instruction
    assert f"Dir={card_path.parent.resolve()}" in agent.instruction
    assert "Root=/workspace" in agent.instruction


def test_apply_instruction_context_sets_agent_metadata_on_mcp_agent(tmp_path: Path) -> None:
    card_path = tmp_path / "cards" / "agent.md"
    agent = StubMcpAgent(
        name="planner",
        instruction_template="Agent {{agentName}} @ {{workspaceRoot}}",
        source_path=card_path,
        agent_type=AgentType.BASIC,
    )

    asyncio.run(apply_instruction_context([agent], {"workspaceRoot": "/workspace"}))

    assert agent.instruction_context["workspaceRoot"] == "/workspace"
    assert agent.instruction_context["agentName"] == "planner"
    assert agent.instruction_context["agentType"] == AgentType.BASIC.value
    assert agent.instruction_context["agentCardPath"] == str(card_path.resolve())
    assert agent.instruction_context["agentCardDir"] == str(card_path.parent.resolve())
    assert agent.instruction == "Agent planner @ /workspace"


def test_apply_instruction_context_resolves_agent_skills_for_non_mcp_agent(
    tmp_path: Path,
) -> None:
    skills_dir = tmp_path / ".fast-agent" / "skills" / "test-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        """---
name: test-skill
description: A test skill for unit testing
---
""",
        encoding="utf-8",
    )

    context: dict[str, str] = {}
    original_home = os.environ.pop("FAST_AGENT_HOME", None)
    import fast_agent.config as config_module

    original_settings = getattr(config_module, "_settings", None)
    config_module._settings = None
    try:
        enrich_with_environment_context(
            context,
            str(tmp_path),
            {"name": "test-client"},
        )
    finally:
        config_module._settings = original_settings
        if original_home is not None:
            os.environ["FAST_AGENT_HOME"] = original_home

    agent = StubAgent(
        name="workflow",
        instruction_template="Workflow skills:\n{{agentSkills}}",
        source_path=tmp_path / "cards" / "workflow.md",
        agent_type=AgentType.CHAIN,
    )

    asyncio.run(apply_instruction_context([agent], context))

    assert "{{agentSkills}}" not in agent.instruction
    assert "test-skill" in agent.instruction
    assert "A test skill for unit testing" in agent.instruction


def test_apply_instruction_context_uses_filtered_agent_skills_per_mcp_agent(
    tmp_path: Path,
) -> None:
    for skill_name, description in (
        ("skill-a", "Skill A description"),
        ("skill-b", "Skill B description"),
    ):
        skill_dir = tmp_path / ".fast-agent" / "skills" / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"""---
name: {skill_name}
description: {description}
---
""",
            encoding="utf-8",
        )

    context: dict[str, str] = {}
    original_home = os.environ.pop("FAST_AGENT_HOME", None)
    import fast_agent.config as config_module

    original_settings = getattr(config_module, "_settings", None)
    config_module._settings = None
    try:
        enrich_with_environment_context(
            context,
            str(tmp_path),
            {"name": "test-client"},
        )
        all_manifests = load_skills_for_context(str(tmp_path))
    finally:
        config_module._settings = original_settings
        if original_home is not None:
            os.environ["FAST_AGENT_HOME"] = original_home

    manifests_by_name = {manifest.name: manifest for manifest in all_manifests}

    agent_a = StubMcpAgent(
        name="agent-a",
        instruction_template="Skills:\n{{agentSkills}}",
        source_path=tmp_path / "cards" / "agent-a.md",
        agent_type=AgentType.BASIC,
    )
    agent_a.set_skill_manifests([manifests_by_name["skill-a"]])

    agent_b = StubMcpAgent(
        name="agent-b",
        instruction_template="Skills:\n{{agentSkills}}",
        source_path=tmp_path / "cards" / "agent-b.md",
        agent_type=AgentType.BASIC,
    )
    agent_b.set_skill_manifests([manifests_by_name["skill-b"]])

    asyncio.run(apply_instruction_context([agent_a, agent_b], context))

    assert "skill-a" in agent_a.instruction
    assert "skill-b" not in agent_a.instruction
    assert "skill-b" in agent_b.instruction
    assert "skill-a" not in agent_b.instruction


def test_apply_instruction_context_preserves_shared_skills_for_mcp_agent_without_override(
    tmp_path: Path,
) -> None:
    skills_dir = tmp_path / ".fast-agent" / "skills" / "test-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        """---
name: test-skill
description: A test skill for unit testing
---
""",
        encoding="utf-8",
    )

    context: dict[str, str] = {}
    original_home = os.environ.pop("FAST_AGENT_HOME", None)
    import fast_agent.config as config_module

    original_settings = getattr(config_module, "_settings", None)
    config_module._settings = None
    try:
        enrich_with_environment_context(
            context,
            str(tmp_path),
            {"name": "test-client"},
        )
    finally:
        config_module._settings = original_settings
        if original_home is not None:
            os.environ["FAST_AGENT_HOME"] = original_home

    agent = StubMcpAgent(
        name="planner",
        instruction_template="Skills:\n{{agentSkills}}",
        source_path=tmp_path / "cards" / "planner.md",
        agent_type=AgentType.BASIC,
    )

    asyncio.run(apply_instruction_context([agent], context))

    assert "{{agentSkills}}" not in agent.instruction
    assert "test-skill" in agent.instruction
    assert "A test skill for unit testing" in agent.instruction


def test_apply_instruction_context_blanks_shared_skills_for_explicit_mcp_disable(
    tmp_path: Path,
) -> None:
    skills_dir = tmp_path / ".fast-agent" / "skills" / "test-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        """---
name: test-skill
description: A test skill for unit testing
---
""",
        encoding="utf-8",
    )

    context: dict[str, str] = {}
    original_home = os.environ.pop("FAST_AGENT_HOME", None)
    import fast_agent.config as config_module

    original_settings = getattr(config_module, "_settings", None)
    config_module._settings = None
    try:
        enrich_with_environment_context(
            context,
            str(tmp_path),
            {"name": "test-client"},
        )
    finally:
        config_module._settings = original_settings
        if original_home is not None:
            os.environ["FAST_AGENT_HOME"] = original_home

    agent = StubMcpAgent(
        name="planner",
        instruction_template="Skills:\n{{agentSkills}}",
        source_path=tmp_path / "cards" / "planner.md",
        agent_type=AgentType.BASIC,
    )
    agent.config.skills = []

    asyncio.run(apply_instruction_context([agent], context))

    assert "{{agentSkills}}" not in agent.instruction
    assert "test-skill" not in agent.instruction
    assert "A test skill for unit testing" not in agent.instruction


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_id", "socket_path", "kind", "shell_enabled", "expected"),
    [
        ("t7", "/run/hc.sock", "local", True, True),
        ("", "/run/hc.sock", "local", True, False),
        ("t7", "", "local", True, False),
        ("   ", "/run/hc.sock", "local", True, False),
        ("t7", "   ", "local", True, False),
        ("t7", "/run/hc.sock", "remote", True, False),
        ("t7", "/run/hc.sock", "docker", True, False),
        ("t7", "/run/hc.sock", "custom", True, False),
        ("t7", "/run/hc.sock", None, True, False),
        ("t7", "/run/hc.sock", "local", False, False),
    ],
)
async def test_herdr_instructions_follow_execution_capability(
    monkeypatch: pytest.MonkeyPatch,
    terminal_id: str,
    socket_path: str,
    kind: str | None,
    shell_enabled: bool,
    expected: bool,
) -> None:
    monkeypatch.setenv("HC_TERMINAL_ID", terminal_id)
    monkeypatch.setenv("HC_TERMINAL_SOCKET", socket_path)
    # TERM_PROGRAM is not a reliable indicator of hc ownership.
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    context: dict[str, str] = {}
    environment = (
        LocalEnvironment(logger=get_logger(__name__))
        if kind == "local"
        else _PromptEnvironment(kind)
        if kind is not None
        else None
    )
    refresh_execution_environment_context(context, environment)
    agent = McpAgent(
        AgentConfig(name="worker", instruction="Before\n{{herdrInstructions}}\nAfter"),
        context=Context(shell_runtime=shell_enabled),
    )
    assert not agent.config.shell
    assert agent.shell_runtime_enabled is shell_enabled

    await apply_instruction_context([agent], context)

    assert "{{herdrInstructions}}" not in agent.instruction
    assert ("hc --skill" in agent.instruction) is expected
    assert ("HC_TERMINAL_ID" in agent.instruction) is expected
    if not expected:
        assert agent.instruction == "Before\n\nAfter"


@pytest.mark.asyncio
async def test_herdr_guidance_refreshes_and_custom_prompts_can_omit_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HC_TERMINAL_ID", "t7")
    monkeypatch.setenv("HC_TERMINAL_SOCKET", "/run/hc.sock")
    context: dict[str, str] = {}
    local = LocalEnvironment(logger=get_logger(__name__))
    default_agent = McpAgent(
        AgentConfig(name="default", instruction=DEFAULT_AGENT_INSTRUCTION),
        context=Context(shell_runtime=True),
    )
    custom_agent = McpAgent(
        AgentConfig(name="custom", instruction="Only my instructions."),
        context=Context(shell_runtime=True),
    )

    for environment, expected in [
        (local, True),
        (_PromptEnvironment(), False),
        (local, True),
        (None, False),
    ]:
        refresh_execution_environment_context(context, environment)
        await apply_instruction_context([default_agent, custom_agent], context)
        assert ("hc --skill" in default_agent.instruction) is expected
        assert "{{herdrInstructions}}" not in default_agent.instruction
        assert custom_agent.instruction == "Only my instructions."

    monkeypatch.delenv("HC_TERMINAL_ID")
    monkeypatch.delenv("HC_TERMINAL_SOCKET")
    refresh_execution_environment_context(context, local)
    await apply_instruction_context([default_agent], context)
    assert "hc --skill" not in default_agent.instruction


@pytest.mark.asyncio
async def test_herdr_template_without_harness_context_is_empty() -> None:
    assert await build_instruction("Before{{herdrInstructions}}After") == "BeforeAfter"


@pytest.mark.asyncio
@pytest.mark.parametrize("local", [True, False])
async def test_herdr_guidance_reaches_harness_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, local: bool
) -> None:
    monkeypatch.setenv("HC_TERMINAL_ID", "t7")
    monkeypatch.setenv("HC_TERMINAL_SOCKET", "/run/hc.sock")
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text("default_model: passthrough\nptc:\n  enabled: false\n", encoding="utf-8")
    fast = FastAgent(
        "herdr-test",
        config_path=str(config_path),
        parse_cli_args=False,
        isolated=True,
        no_home=True,
        quiet=True,
    )

    @fast.agent(name="worker", model="passthrough", shell=True)
    async def worker() -> None:
        pass

    environment = None if local else _PromptEnvironment()
    async with fast.run(environment=environment) as app:
        agent = app["worker"]
        assert ("hc --skill" in agent.instruction) is local
        assert "{{herdrInstructions}}" not in agent.instruction


@pytest.mark.asyncio
async def test_herdr_guidance_honors_runtime_no_shell_override() -> None:
    agent = McpAgent(
        AgentConfig(name="worker", instruction="{{herdrInstructions}}", shell=True),
        context=Context(no_shell=True),
    )
    assert agent.config.shell
    assert not agent.shell_runtime_enabled

    await apply_instruction_context([agent], {"herdrInstructions": "hc --skill"})

    assert "hc --skill" not in agent.instruction
