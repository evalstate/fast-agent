from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp_types import Tool

from fast_agent.agents.agent_types import AgentType
from fast_agent.core.agent_app import AgentRefreshResult
from fast_agent.ptc.mentions import TOOL_DEFINITIONS_META
from fast_agent.tools.tool_surface import (
    ProgrammaticSurface,
    SurfaceTool,
    ToolExposure,
    ToolSurface,
)
from fast_agent.types import PromptMessageExtended
from fast_agent.ui import interactive_prompt
from fast_agent.ui.interactive_prompt import (
    InteractivePrompt,
    PendingCommandExecution,
    PromptLoopRuntimeState,
)

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.ui.console_display import ConsoleDisplay


class _Agent:
    def __init__(self, *, programmatic: bool) -> None:
        self.programmatic = programmatic
        self.message_history: list[PromptMessageExtended] = []

    async def tool_surface(self) -> ToolSurface:
        if not self.programmatic:
            return ToolSurface(tools=())
        return ToolSurface(
            programmatic=ProgrammaticSurface(
                tool_name="exec_sandbox",
                vocabulary=1,
                tool_format="json",
                instructions="",
                type_check=False,
                max_duration_seconds=30,
                max_memory_mb=64,
            ),
            tools=(
                SurfaceTool(
                    tool=Tool(name="jobs__list", input_schema={}),
                    exposure=ToolExposure.PROGRAMMATIC,
                    context_tokens=0,
                    python_name="jobs_list",
                    stub="def jobs_list(): ...",
                ),
            ),
        )


class _App:
    """Minimal prompt provider: ``main`` has no PTC, ``worker`` does."""

    no_home_mode = False
    missing_shell_cwd_policy_override = None

    def __init__(self) -> None:
        self.agents = {"main": _Agent(programmatic=False), "worker": _Agent(programmatic=True)}

    async def refresh_if_needed(self) -> bool:
        return False

    def latest_refresh_result(self) -> AgentRefreshResult:
        return AgentRefreshResult(changed=False)

    def _agent(self, name: str) -> _Agent:
        return self.agents[name]

    def visible_agent_names(self, *, force_include: str | None = None) -> list[str]:
        del force_include
        return list(self.agents)

    def visible_agent_types(self, *, force_include: str | None = None) -> dict[str, AgentType]:
        del force_include
        return dict.fromkeys(self.agents, AgentType.BASIC)

    def registered_agent_names(self) -> list[str]:
        return list(self.agents)

    def registered_agents(self) -> dict[str, _Agent]:
        return self.agents

    def resolve_target_agent_name(self, agent_name: str | None = None) -> str | None:
        return agent_name or "main"


@pytest.mark.asyncio
async def test_failed_mention_keeps_the_draft_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = "Use ^tool:typo to list jobs"
    prefills: list[str] = []
    inputs = iter([draft, "STOP"])

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        prefills.append(kwargs.get("pre_populate_buffer", ""))
        return next(inputs)

    printed: list[object] = []
    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)
    monkeypatch.setattr(interactive_prompt, "rich_print", printed.append)
    sent: list[object] = []

    async def send(payload: object, _agent_name: str) -> str:
        sent.append(payload)
        return "ok"

    await InteractivePrompt().prompt_loop(
        send_func=send,
        default_agent="worker",
        available_agents=["worker"],
        prompt_provider=cast("Any", _App()),
    )

    assert sent == []
    assert prefills == ["", draft]
    assert any("^tool:typo" in str(getattr(item, "plain", item)) for item in printed)


async def _hash_send(message: str) -> tuple[list[tuple[object, str]], str | None]:
    sent: list[tuple[object, str]] = []

    async def send(payload: object, agent_name: str) -> str:
        sent.append((payload, agent_name))
        return "done"

    result = await InteractivePrompt()._handle_pending_execution(
        pending=PendingCommandExecution(hash_send_target="worker", hash_send_message=message),
        send_func=send,
        quiet_send_func=None,
        prompt_provider=cast("AgentApp", _App()),
        agent_name="main",
        display=cast("ConsoleDisplay", object()),
        current_result="",
        runtime_state=PromptLoopRuntimeState(),
    )
    assert result.handled
    return sent, result.buffer_prefill


@pytest.mark.asyncio
async def test_hash_send_resolves_mentions_against_the_destination_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(interactive_prompt, "rich_print", lambda *_args: None)

    sent, _ = await _hash_send("Use ^tool:jobs_list now")

    [(payload, agent_name)] = sent
    assert agent_name == "worker"
    assert isinstance(payload, PromptMessageExtended)
    assert payload.first_text() == "Use `jobs_list` now"
    assert (payload.content[-1].meta or {}).get(TOOL_DEFINITIONS_META) == ["jobs_list"]


@pytest.mark.asyncio
async def test_failed_hash_send_mention_restores_the_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(interactive_prompt, "rich_print", lambda *_args: None)

    sent, buffer_prefill = await _hash_send("Use ^tool:typo now")

    assert sent == []
    assert buffer_prefill == "#worker Use ^tool:typo now"
