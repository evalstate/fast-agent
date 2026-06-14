from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from fast_agent.config import CompactionSettings, Settings
from fast_agent.context import Context
from fast_agent.core.direct_factory import _auto_compaction_after_turn_hook
from fast_agent.hooks.compaction import auto_compact_history
from fast_agent.hooks.hook_context import HookContext
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.usage_tracking import UsageAccumulator
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


class _Agent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.config = SimpleNamespace(tool_only=False, use_history=True)
        self.context = Context(config=Settings(compaction=CompactionSettings(threshold=0.5)))
        self.message_history: list[PromptMessageExtended] = []
        self.usage_accumulator = UsageAccumulator()
        self.usage_accumulator.set_context_window_size(100)
        self.usage_accumulator.set_context_estimate(90)
        self.agent_registry = None

    def load_message_history(self, messages: list[PromptMessageExtended] | None) -> None:
        self.message_history = list(messages or [])

    def get_agent(self, name: str) -> None:
        del name
        return None


class _Runner:
    def __init__(self, agent: _Agent, request_params: RequestParams | None = None) -> None:
        self.agent = agent
        self.iteration = 0
        self.request_params = request_params


def _complete_message() -> PromptMessageExtended:
    return PromptMessageExtended(role="assistant", content=[], stop_reason=LlmStopReason.END_TURN)


@pytest.mark.asyncio
async def test_auto_compaction_skips_no_history_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _Agent("source")
    runner = _Runner(agent, RequestParams(use_history=False))

    async def fail_compact(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no-history turns must not auto-compact")

    monkeypatch.setattr("fast_agent.hooks.compaction.compact_conversation", fail_compact)

    await auto_compact_history(
        HookContext(
            runner=runner,
            agent=agent,
            message=_complete_message(),
            hook_type="after_turn_complete",
        )
    )


@pytest.mark.asyncio
async def test_copied_auto_compaction_hook_uses_runner_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _Agent("source")
    clone = _Agent("clone")
    compacted_agents: list[object] = []

    async def fake_compact(agent: object, **_kwargs: object) -> object:
        compacted_agents.append(agent)
        return SimpleNamespace(
            agent_name="clone",
            messages_before=4,
            messages_after=2,
            tokens_before=90,
            tokens_after_estimate=10,
            context_window=100,
            archive_file=None,
        )

    monkeypatch.setattr("fast_agent.hooks.compaction.compact_conversation", fake_compact)

    hook = _auto_compaction_after_turn_hook(cast("Any", source), None)
    await hook(_Runner(clone), _complete_message())

    assert compacted_agents == [clone]
