from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any, cast

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.command_actions import PluginCommandActionSpec
from fast_agent.ui.prompt.completer import AgentCompleter
from fast_agent.ui.prompt.keybindings import create_keybindings
from fast_agent.utils.async_utils import run_in_thread

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp


def test_plugin_command_keybinding_is_registered() -> None:
    kb = create_keybindings(
        agent_provider=cast(
            "AgentApp",
            _Provider(
                AgentConfig(
                    name="dev",
                    commands={
                        "draft-next": PluginCommandActionSpec(
                            name="draft-next",
                            description="Draft next",
                            handler="commands.py:draft_next",
                            key="c-x d",
                        )
                    },
                )
            ),
        ),
        agent_name="dev",
    )

    assert (Keys.ControlX, "d") in {binding.keys for binding in kb.bindings}


def test_plugin_commands_are_added_to_tui_completion() -> None:
    completer = AgentCompleter(
        agents=["dev"],
        current_agent="dev",
        agent_provider=cast(
            "AgentApp",
            _Provider(
                AgentConfig(
                    name="dev",
                    commands={
                        "review-last": PluginCommandActionSpec(
                            name="review-last",
                            description="Review the last assistant response",
                            input_hint="[agent]",
                            handler="commands.py:review_last",
                            key="c-x r",
                        )
                    },
                )
            ),
        ),
    )

    assert (
        completer.commands["review-last"]
        == "Review the last assistant response [agent] (key: c-x r)"
    )


def test_agent_plugin_command_overrides_global_completion() -> None:
    completer = AgentCompleter(
        agents=["dev"],
        current_agent="dev",
        agent_provider=cast(
            "AgentApp",
            _Provider(
                AgentConfig(
                    name="dev",
                    commands={
                        "draft": PluginCommandActionSpec(
                            name="draft",
                            description="Agent draft",
                            handler="agent.py:draft",
                        )
                    },
                ),
                plugin_commands={
                    "draft": PluginCommandActionSpec(
                        name="draft",
                        description="Global draft",
                        handler="global.py:draft",
                    )
                },
            ),
        ),
    )

    assert completer.commands["draft"] == "Agent draft"


def test_plugin_argument_completion_resolves_agent_relative_completer(tmp_path) -> None:
    command_file = tmp_path / "commands.py"
    command_file.write_text(
        "async def complete(ctx):\n"
        "    return ['alpha']\n",
        encoding="utf-8",
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    completer = AgentCompleter(
        agents=["dev"],
        current_agent="dev",
        cwd=cwd,
        agent_provider=cast(
            "AgentApp",
            _Provider(
                AgentConfig(
                    name="dev",
                    source_path=tmp_path / "agent.md",
                    commands={
                        "review": PluginCommandActionSpec(
                            name="review",
                            description="Review",
                            handler="./commands.py:review",
                            completer="./commands.py:complete",
                        )
                    },
                )
            ),
        ),
    )

    completions = list(completer.get_completions(Document("/review ", 8), None))

    assert [completion.text for completion in completions] == ["alpha"]


@pytest.mark.asyncio
async def test_plugin_argument_completion_runs_with_active_loop(tmp_path) -> None:
    command_file = tmp_path / "commands.py"
    command_file.write_text(
        "import asyncio\n"
        "async def complete(ctx):\n"
        "    await asyncio.sleep(0)\n"
        "    return ['alpha']\n",
        encoding="utf-8",
    )
    completer = AgentCompleter(
        agents=["dev"],
        current_agent="dev",
        agent_provider=cast(
            "AgentApp",
            _Provider(
                AgentConfig(name="dev"),
                plugin_commands={
                    "review": PluginCommandActionSpec(
                        name="review",
                        description="Review",
                        handler="./commands.py:review",
                        completer="./commands.py:complete",
                    )
                },
                plugin_command_base_path=tmp_path,
            ),
        ),
    )

    await asyncio.sleep(0)
    completions = list(completer.get_completions(Document("/review ", 8), None))

    assert [completion.text for completion in completions] == ["alpha"]


@pytest.mark.asyncio
async def test_plugin_argument_completion_timeout_does_not_wait_for_worker() -> None:
    release_worker = threading.Event()
    completer = AgentCompleter(agents=["dev"])
    completer._completion_wait_timeout_seconds = 0.01

    async def slow_completion():
        await run_in_thread(release_worker.wait)
        return ["late"]

    started_at = time.perf_counter()
    try:
        result = completer._run_plugin_command_completion(slow_completion)
        elapsed = time.perf_counter() - started_at

        assert result is None
        assert elapsed < 0.25
    finally:
        release_worker.set()
        await asyncio.sleep(0.05)


class _Agent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config


class _Provider:
    def __init__(
        self,
        config: AgentConfig,
        plugin_commands: dict[str, PluginCommandActionSpec] | None = None,
        plugin_command_base_path=None,
    ) -> None:
        self.agent = _Agent(config)
        self.plugin_commands = plugin_commands
        self.plugin_command_base_path = plugin_command_base_path

    def get_agent(self, name: str) -> Any:
        return self.agent
