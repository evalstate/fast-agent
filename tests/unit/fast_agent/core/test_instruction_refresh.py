"""Tests for instruction building and refresh utilities."""

import asyncio

from fast_agent.core.instruction_refresh import (
    McpInstructionCapable,
    build_instruction,
    format_server_instructions,
    rebuild_agent_instruction,
)


class StubAggregator:
    """Stub aggregator that returns predefined server instructions."""

    def __init__(self, instructions: dict[str, tuple[str | None, list[str]]] | None = None):
        self._instructions = instructions or {}

    async def get_server_instructions(self) -> dict[str, tuple[str | None, list[str]]]:
        return self._instructions


class StubAgent:
    """Stub that implements McpInstructionCapable for testing."""

    def __init__(
        self,
        template: str = "Test instruction",
        aggregator: StubAggregator | None = None,
    ) -> None:
        self._instruction = template
        self._instruction_template = template
        self._instruction_context: dict[str, str] = {}
        self._skill_manifests: list = []
        self._skill_registry = None
        self._aggregator = aggregator or StubAggregator()
        self._has_filesystem_runtime = False

    @property
    def instruction(self) -> str:
        return self._instruction

    def set_instruction(self, instruction: str) -> None:
        self._instruction = instruction

    @property
    def instruction_template(self) -> str:
        return self._instruction_template

    @property
    def instruction_context(self) -> dict[str, str]:
        return self._instruction_context

    @property
    def aggregator(self):
        return self._aggregator

    @property
    def skill_manifests(self) -> list:
        return self._skill_manifests

    @property
    def skill_registry(self):
        return self._skill_registry

    @skill_registry.setter
    def skill_registry(self, value):
        self._skill_registry = value

    def set_skill_manifests(self, manifests) -> None:
        self._skill_manifests = list(manifests)

    def set_instruction_context(self, context: dict[str, str]) -> None:
        self._instruction_context.update(context)

    @property
    def has_filesystem_runtime(self) -> bool:
        return self._has_filesystem_runtime


# Ensure StubAgent is recognized as McpInstructionCapable
assert isinstance(StubAgent(), McpInstructionCapable)


# ─────────────────────────────────────────────────────────────────────────────
# Test format_server_instructions
# ─────────────────────────────────────────────────────────────────────────────


def test_format_server_instructions_empty() -> None:
    result = format_server_instructions({})
    assert result == ""


def test_format_server_instructions_with_data() -> None:
    data = {
        "test-server": ("Do helpful things", ["tool1", "tool2"]),
    }
    result = format_server_instructions(data)
    assert "test-server" in result
    assert "Do helpful things" in result
    assert "test-server__tool1" in result
    assert "test-server__tool2" in result


def test_format_server_instructions_skips_none() -> None:
    data = {
        "server1": ("Instructions", ["tool1"]),
        "server2": (None, ["tool2"]),  # Should be skipped
    }
    result = format_server_instructions(data)
    assert "server1" in result
    assert "server2" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Test build_instruction
# ─────────────────────────────────────────────────────────────────────────────


def test_build_instruction_resolves_builtins() -> None:
    template = "Today is {{currentDate}}. Platform: {{hostPlatform}}"
    result = asyncio.run(build_instruction(template))
    # Should not contain the placeholders anymore
    assert "{{currentDate}}" not in result
    assert "{{hostPlatform}}" not in result


def test_build_instruction_with_context() -> None:
    template = "Root: {{workspaceRoot}}"
    result = asyncio.run(build_instruction(template, context={"workspaceRoot": "/test/path"}))
    assert result == "Root: /test/path"


def test_build_instruction_with_aggregator() -> None:
    template = "{{serverInstructions}}"
    aggregator = StubAggregator({"my-server": ("Be helpful", ["do_thing"])})
    result = asyncio.run(build_instruction(template, aggregator=aggregator))
    assert "my-server" in result
    assert "Be helpful" in result


# ─────────────────────────────────────────────────────────────────────────────
# Test rebuild_agent_instruction
# ─────────────────────────────────────────────────────────────────────────────


def test_rebuild_agent_instruction_updates_fields() -> None:
    agent = StubAgent(template="Hello {{workspaceRoot}}")
    result = asyncio.run(
        rebuild_agent_instruction(
            agent,
            context={"workspaceRoot": "/my/path"},
        )
    )
    assert agent.instruction == "Hello /my/path"
    assert agent.instruction_context == {"workspaceRoot": "/my/path"}
    assert result.updated_context is True
    assert result.rebuilt_instruction is True


def test_rebuild_agent_instruction_updates_skill_manifests() -> None:
    agent = StubAgent()
    result = asyncio.run(
        rebuild_agent_instruction(
            agent,
            skill_manifests=["manifest1", "manifest2"],
        )
    )
    assert agent.skill_manifests == ["manifest1", "manifest2"]
    assert result.updated_skill_manifests is True


def test_rebuild_agent_instruction_updates_skill_registry() -> None:
    agent = StubAgent()
    result = asyncio.run(
        rebuild_agent_instruction(
            agent,
            skill_registry="my-registry",
        )
    )
    assert agent.skill_registry == "my-registry"
    assert result.updated_skill_registry is True


def test_rebuild_agent_instruction_handles_non_mcp_agent() -> None:
    class MinimalAgent:
        pass

    agent = MinimalAgent()
    result = asyncio.run(rebuild_agent_instruction(agent))
    assert result.updated_skill_manifests is False
    assert result.updated_context is False
    assert result.updated_skill_registry is False
    assert result.rebuilt_instruction is False


def test_rebuild_agent_instruction_handles_empty_template() -> None:
    agent = StubAgent(template="")
    result = asyncio.run(rebuild_agent_instruction(agent))
    assert result.rebuilt_instruction is False
