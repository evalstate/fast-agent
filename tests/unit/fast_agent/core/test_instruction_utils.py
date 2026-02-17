"""Tests for instruction utility helpers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.core.instruction_utils import (
    apply_instruction_context,
    build_agent_instruction_context,
)

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
        agent_type: AgentType = AgentType.SMART,
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
    def has_filesystem_runtime(self) -> bool:
        return False


def test_build_agent_instruction_context_includes_agent_metadata(tmp_path: Path) -> None:
    card_path = tmp_path / "cards" / "smart.md"
    agent = StubAgent(
        name="smarty",
        instruction_template="Hello",
        source_path=card_path,
        agent_type=AgentType.SMART,
    )

    context = build_agent_instruction_context(agent, {"workspaceRoot": "/workspace"})

    assert context["workspaceRoot"] == "/workspace"
    assert context["agentName"] == "smarty"
    assert context["agentType"] == AgentType.SMART.value
    assert context["agentCardPath"] == str(card_path.resolve())
    assert context["agentCardDir"] == str(card_path.parent.resolve())


def test_build_agent_instruction_context_uses_internal_when_no_card_path() -> None:
    agent = StubAgent(
        name="agent",
        instruction_template="Hello",
        source_path=None,
        agent_type=AgentType.SMART,
    )

    context = build_agent_instruction_context(agent, {"workspaceRoot": "/workspace"})

    assert context["workspaceRoot"] == "/workspace"
    assert context["agentName"] == "agent"
    assert context["agentType"] == AgentType.SMART.value
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
        agent_type=AgentType.SMART,
    )

    asyncio.run(apply_instruction_context([agent], {"workspaceRoot": "/workspace"}))

    assert "{{agentName}}" not in agent.instruction
    assert "{{agentCardPath}}" not in agent.instruction
    assert "Name=reviewer" in agent.instruction
    assert f"Card={card_path.resolve()}" in agent.instruction
    assert f"Dir={card_path.parent.resolve()}" in agent.instruction
    assert "Root=/workspace" in agent.instruction


def test_apply_instruction_context_sets_agent_metadata_on_mcp_agent(tmp_path: Path) -> None:
    card_path = tmp_path / "cards" / "smart.md"
    agent = StubMcpAgent(
        name="planner",
        instruction_template="Agent {{agentName}} @ {{workspaceRoot}}",
        source_path=card_path,
        agent_type=AgentType.SMART,
    )

    asyncio.run(apply_instruction_context([agent], {"workspaceRoot": "/workspace"}))

    assert agent.instruction_context["workspaceRoot"] == "/workspace"
    assert agent.instruction_context["agentName"] == "planner"
    assert agent.instruction_context["agentType"] == AgentType.SMART.value
    assert agent.instruction_context["agentCardPath"] == str(card_path.resolve())
    assert agent.instruction_context["agentCardDir"] == str(card_path.parent.resolve())
    assert agent.instruction == "Agent planner @ /workspace"
