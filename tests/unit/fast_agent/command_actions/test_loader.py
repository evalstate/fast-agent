from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.command_actions import (
    PluginCommandActionContext,
    PluginCommandActionRegistry,
    PluginCommandActionResult,
    normalize_plugin_command_action_result,
)
from fast_agent.command_actions.loader import load_plugin_command_action_function
from fast_agent.command_actions.models import PluginCommandActionSpec
from fast_agent.core.exceptions import AgentConfigError

if TYPE_CHECKING:
    from pathlib import Path


def test_load_plugin_command_action_function_accepts_async_handler(tmp_path: Path) -> None:
    module_path = tmp_path / "commands.py"
    module_path.write_text(
        "async def run(ctx):\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    handler = load_plugin_command_action_function("commands.py:run", base_path=tmp_path)

    assert handler.__name__ == "run"


def test_load_plugin_command_action_function_rejects_sync_handler(tmp_path: Path) -> None:
    module_path = tmp_path / "commands.py"
    module_path.write_text(
        "def run(ctx):\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentConfigError, match="must be async"):
        load_plugin_command_action_function("commands.py:run", base_path=tmp_path)


@pytest.mark.asyncio
async def test_registry_executes_and_normalizes_string_result(tmp_path: Path) -> None:
    module_path = tmp_path / "commands.py"
    module_path.write_text(
        "async def greet(ctx):\n"
        "    return f'hello {ctx.arguments}'\n",
        encoding="utf-8",
    )
    registry = PluginCommandActionRegistry.from_specs(
        {
            "greet": PluginCommandActionSpec(
                name="greet",
                description="Greet",
                handler="commands.py:greet",
            )
        },
        base_path=tmp_path,
    )

    result = await registry.execute(
        "greet",
        PluginCommandActionContext(
            command_name="greet",
            arguments="world",
            agent=_CommandAgent(),
        ),
    )

    assert result == PluginCommandActionResult(message="hello world")


def test_normalize_plugin_command_action_result_handles_none_and_strings() -> None:
    assert normalize_plugin_command_action_result(None) == PluginCommandActionResult()
    assert normalize_plugin_command_action_result("ok") == PluginCommandActionResult(message="ok")


class _CommandAgent:
    name = "agent"
    context = None
    config = None
    agent_registry = None
    message_history = []

    def load_message_history(self, messages):
        self.message_history = messages or []

    def get_agent(self, name: str):
        return None

    async def send(self, message: str) -> str:
        return message
