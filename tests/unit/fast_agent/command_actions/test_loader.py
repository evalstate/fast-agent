from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.command_actions import (
    PluginCommandActionContext,
    PluginCommandActionRegistry,
    PluginCommandActionResult,
    normalize_plugin_command_action_result,
    parse_plugin_command_action_specs,
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


def test_load_plugin_command_action_function_rejects_missing_handler(tmp_path: Path) -> None:
    module_path = tmp_path / "commands.py"
    module_path.write_text("async def other(ctx):\n    return 'ok'\n", encoding="utf-8")

    with pytest.raises(AgentConfigError, match="Command action function 'run' not found"):
        load_plugin_command_action_function("commands.py:run", base_path=tmp_path)


@pytest.mark.parametrize("spec", ["commands.py:", ":run"])
def test_load_plugin_command_action_function_rejects_incomplete_handler_specs(
    tmp_path: Path,
    spec: str,
) -> None:
    with pytest.raises(AgentConfigError, match="Expected format: 'module.py:function_name'"):
        load_plugin_command_action_function(spec, base_path=tmp_path)


def test_load_plugin_command_action_function_rejects_non_callable_target(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "commands.py"
    module_path.write_text("run = None\n", encoding="utf-8")

    with pytest.raises(AgentConfigError, match="Command action target 'run' is not callable"):
        load_plugin_command_action_function("commands.py:run", base_path=tmp_path)


@pytest.mark.asyncio
async def test_load_plugin_command_action_function_registers_module_during_import(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "commands.py"
    module_path.write_text(
        "import sys\n"
        "MODULE_REGISTERED = sys.modules[__name__] is sys.modules.get(__name__)\n"
        "async def run(ctx):\n"
        "    return MODULE_REGISTERED\n",
        encoding="utf-8",
    )

    handler = load_plugin_command_action_function("commands.py:run", base_path=tmp_path)

    assert await handler(cast("PluginCommandActionContext", None)) is True


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
        "GREET",
        PluginCommandActionContext(
            command_name="GREET",
            arguments="world",
            agent=_CommandAgent(),
        ),
    )

    assert result == PluginCommandActionResult(message="hello world")


def test_normalize_plugin_command_action_result_handles_none_and_strings() -> None:
    assert normalize_plugin_command_action_result(None) == PluginCommandActionResult()
    assert normalize_plugin_command_action_result("ok") == PluginCommandActionResult(message="ok")


def test_parse_plugin_command_action_specs_normalizes_string_fields() -> None:
    specs = parse_plugin_command_action_specs(
        {
            "/Review": {
                "description": "  Review code  ",
                "handler": "  commands.py:review  ",
                "input_hint": "  optional notes  ",
                "key": "  ctrl-r  ",
            },
            "empty_optional": {
                "description": "Empty optional values",
                "handler": "commands.py:empty",
                "input_hint": "   ",
                "key": "",
            },
        },
        source="plugin.yaml",
    )

    assert specs is not None
    assert specs["review"] == PluginCommandActionSpec(
        name="review",
        description="Review code",
        handler="commands.py:review",
        input_hint="optional notes",
        key="ctrl-r",
    )
    assert specs["empty_optional"].input_hint is None
    assert specs["empty_optional"].key is None


def test_parse_plugin_command_action_specs_rejects_duplicate_normalized_names() -> None:
    with pytest.raises(
        AgentConfigError,
        match="Duplicate command action 'review' after normalization in plugin.yaml",
    ):
        parse_plugin_command_action_specs(
            {
                "/Review": {
                    "description": "Review code",
                    "handler": "commands.py:review",
                },
                "review": {
                    "description": "Review code again",
                    "handler": "commands.py:review_again",
                },
            },
            source="plugin.yaml",
        )


def test_parse_plugin_command_action_specs_casefolds_normalized_names() -> None:
    with pytest.raises(
        AgentConfigError,
        match="Duplicate command action 'strasse' after normalization in plugin.yaml",
    ):
        parse_plugin_command_action_specs(
            {
                "/Straße": {
                    "description": "Review code",
                    "handler": "commands.py:review",
                },
                "strasse": {
                    "description": "Review code again",
                    "handler": "commands.py:review_again",
                },
            },
            source="plugin.yaml",
        )


def test_parse_plugin_command_action_specs_rejects_non_string_names() -> None:
    with pytest.raises(
        AgentConfigError,
        match="Command action names must be strings in plugin.yaml",
    ):
        parse_plugin_command_action_specs(
            {
                123: {
                    "description": "Review code",
                    "handler": "commands.py:review",
                },
            },
            source="plugin.yaml",
        )


def test_parse_plugin_command_action_specs_rejects_invalid_optional_strings() -> None:
    with pytest.raises(
        AgentConfigError,
        match="Command action 'review' field 'input_hint' must be a string in plugin.yaml",
    ):
        parse_plugin_command_action_specs(
            {
                "review": {
                    "description": "Review code",
                    "handler": "commands.py:review",
                    "input_hint": ["not", "a", "string"],
                }
            },
            source="plugin.yaml",
        )


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
