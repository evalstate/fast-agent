from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from prompt_toolkit.keys import Keys

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.command_actions import PluginCommandActionSpec
from fast_agent.ui.prompt.completer import AgentCompleter
from fast_agent.ui.prompt.keybindings import create_keybindings

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


class _Agent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config


class _Provider:
    plugin_command_base_path = None

    def __init__(
        self,
        config: AgentConfig,
        plugin_commands: dict[str, PluginCommandActionSpec] | None = None,
    ) -> None:
        self.agent = _Agent(config)
        self.plugin_commands = plugin_commands

    def get_agent(self, name: str) -> Any:
        return self.agent
