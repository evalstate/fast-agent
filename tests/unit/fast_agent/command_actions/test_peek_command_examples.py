from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from mcp.types import TextContent

from fast_agent.command_actions import PluginCommandActionContext
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

if TYPE_CHECKING:
    from types import ModuleType

    from fast_agent.command_actions.models import PluginCommandAgentProtocol


EXAMPLE_PATH = Path(__file__).parents[4] / "examples" / "plugin-commands" / "peek_commands.py"


@pytest.mark.asyncio
async def test_peek_restores_history_after_send() -> None:
    module = _load_example_module()
    first = PromptMessageExtended(role="user", content=[TextContent(type="text", text="hi")])
    agent = _CommandAgent([first], response="peek result")

    result = await module.peek(
        PluginCommandActionContext(
            command_name="peek",
            arguments="try this",
            agent=cast("PluginCommandAgentProtocol", agent),
        )
    )

    assert result.markdown == "peek result"
    assert agent.sent == ["try this"]
    assert agent.message_history == [first]


@pytest.mark.asyncio
async def test_html_summary_uses_helper_and_restores_helper_history(tmp_path: Path) -> None:
    module = _load_example_module()
    current = PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text="summarise me")],
    )
    helper_message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="helper history")],
    )
    helper = _CommandAgent(
        [helper_message],
        response="```html\n<!doctype html><html><body>Summary</body></html>\n```",
    )
    agent = _CommandAgent([current], response="", agents={"frontend": helper})

    result = await module.html_summary(
        PluginCommandActionContext(
            command_name="html-summary",
            arguments="frontend summary.html",
            agent=cast("PluginCommandAgentProtocol", agent),
            session_cwd=tmp_path,
        )
    )

    target = tmp_path / "summary.html"
    written = target.read_text(encoding="utf-8")
    assert written.startswith("<!doctype html>")
    assert "<style>" in written
    assert '<div class="wrap">' in written
    assert "Summary" in written
    assert result.markdown is not None
    assert result.markdown.startswith(
        "HTML summary written: [open summary.html](http://127.0.0.1:"
    )
    assert result.markdown.endswith("/summary.html)")
    assert helper.message_history == [helper_message]
    assert "summarise me" in helper.sent[0]


def _load_example_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("peek_commands_example", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CommandAgent:
    name = "agent"
    context = None
    config = None

    def __init__(
        self,
        history: list[PromptMessageExtended],
        *,
        response: str,
        agents: dict[str, _CommandAgent] | None = None,
    ) -> None:
        self.message_history = history
        self.agent_registry = agents
        self._response = response
        self.sent: list[str] = []

    def load_message_history(self, messages: list[PromptMessageExtended] | None) -> None:
        self.message_history = messages or []

    def get_agent(self, name: str) -> _CommandAgent | None:
        if self.agent_registry is None:
            return None
        return self.agent_registry.get(name)

    async def send(self, message: str) -> str:
        self.sent.append(message)
        self.message_history.append(
            PromptMessageExtended(role="user", content=[TextContent(type="text", text=message)])
        )
        self.message_history.append(
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text=self._response)],
            )
        )
        return self._response
