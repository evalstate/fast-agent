from __future__ import annotations

import pytest

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import sessions as session_handlers


class _StubIO:
    async def emit(self, message):
        return None

    async def prompt_text(self, prompt: str, *, default=None, allow_empty=True):
        return default

    async def prompt_selection(
        self, prompt: str, *, options, allow_cancel=False, default=None
    ):
        return default

    async def prompt_model_selection(
        self,
        *,
        initial_provider=None,
        default_model=None,
    ):
        del initial_provider, default_model
        return None

    async def prompt_argument(self, arg_name: str, *, description=None, required=True):
        return None

    async def display_history_turn(self, agent_name: str, turn, *, turn_index=None, total_turns=None):
        return None

    async def display_history_overview(self, agent_name: str, history, usage=None):
        return None

    async def display_usage_report(self, agents):
        return None

    async def display_system_prompt(self, agent_name: str, system_prompt: str, *, server_count=0):
        return None


class _StubAgentProvider:
    def _agent(self, name: str):  # noqa: ARG002
        return object()

    def resolve_target_agent_name(self, agent_name: str | None = None):
        return agent_name or "agent"

    def visible_agent_names(self, *, force_include: str | None = None):
        del force_include
        return ["agent"]

    def registered_agent_names(self):
        return ["agent"]

    def registered_agents(self):
        return {"agent": object()}

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):  # noqa: ARG002
        return {}


def _build_noenv_context() -> CommandContext:
    return CommandContext(
        agent_provider=_StubAgentProvider(),
        current_agent_name="agent",
        io=_StubIO(),
        noenv=True,
    )


@pytest.mark.asyncio
async def test_noenv_list_sessions_returns_disabled_message() -> None:
    outcome = await session_handlers.handle_list_sessions(
        _build_noenv_context(),
        show_help=True,
    )

    assert outcome.messages
    assert str(outcome.messages[0].text) == session_handlers.NOENV_SESSION_MESSAGE
    assert outcome.messages[0].channel == "warning"


@pytest.mark.asyncio
async def test_noenv_resume_session_returns_disabled_message() -> None:
    outcome = await session_handlers.handle_resume_session(
        _build_noenv_context(),
        agent_name="agent",
        session_id="latest",
    )

    assert outcome.messages
    assert str(outcome.messages[0].text) == session_handlers.NOENV_SESSION_MESSAGE
    assert outcome.messages[0].channel == "warning"
