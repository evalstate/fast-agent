from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from mcp_types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.config import CompactionSettings, Settings
from fast_agent.context import Context
from fast_agent.core.direct_factory import _auto_compaction_after_turn_hook
from fast_agent.history.compaction import MID_TURN_RECENT_TOOL_EXCHANGES, CompactionResult
from fast_agent.hooks.compaction import auto_compact_history, auto_compact_history_mid_turn
from fast_agent.hooks.hook_context import HookContext
from fast_agent.llm.compaction_support import CompactAt, CompactionSupport
from fast_agent.llm.provider.copilot.messages import CopilotMessagesLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.usage_tracking import UsageAccumulator
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


class _Agent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.config = AgentConfig(name=name, tool_only=False, use_history=True)
        self.context = Context(config=Settings(compaction=CompactionSettings(threshold=0.5)))
        self.message_history: list[PromptMessageExtended] = []
        self.usage_accumulator = UsageAccumulator()
        self.usage_accumulator.set_context_window_size(100)
        self.usage_accumulator.set_context_estimate(90)
        self.agent_registry = None
        self.llm: Any = None

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
        self.delta_messages: list[PromptMessageExtended] = []


def _result(agent_name: str, *, tokens_before: int = 90) -> CompactionResult:
    return CompactionResult(
        agent_name=agent_name,
        summary_text="checkpoint",
        messages_before=4,
        messages_after=2,
        tokens_before=tokens_before,
        tokens_after_estimate=10,
        context_window=100,
        archive_file=None,
    )


def _complete_message() -> PromptMessageExtended:
    return PromptMessageExtended(role="assistant", content=[], stop_reason=LlmStopReason.END_TURN)


def _tool_use_message() -> PromptMessageExtended:
    return PromptMessageExtended(role="assistant", content=[], stop_reason=LlmStopReason.TOOL_USE)


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
        return _result("clone")

    monkeypatch.setattr("fast_agent.hooks.compaction.compact_conversation", fake_compact)

    hook = _auto_compaction_after_turn_hook(cast("Any", source), None)
    await hook(_Runner(clone), _complete_message())

    assert compacted_agents == [clone]


@pytest.mark.asyncio
async def test_mid_turn_auto_compaction_preserves_active_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _Agent("source")
    kwargs_seen: list[dict[str, Any]] = []

    async def fake_compact(_agent: object, **kwargs: Any) -> object:
        kwargs_seen.append(kwargs)
        return _result("source")

    monkeypatch.setattr("fast_agent.hooks.compaction.compact_conversation", fake_compact)

    await auto_compact_history_mid_turn(
        HookContext(
            runner=_Runner(agent),
            agent=agent,
            message=_tool_use_message(),
            hook_type="after_tool_loop_iteration",
        )
    )

    assert kwargs_seen
    assert kwargs_seen[0]["recent_tool_exchanges"] == MID_TURN_RECENT_TOOL_EXCHANGES
    assert kwargs_seen[0]["pending_context_tokens"] == 0


@pytest.mark.asyncio
async def test_mid_turn_auto_compaction_accounts_for_pending_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _Agent("source")
    agent.usage_accumulator.set_context_estimate(40)
    runner = _Runner(agent)
    runner.delta_messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="x" * 80)],
        )
    ]
    compacted = False
    pending_context_tokens: int | None = None

    async def fake_compact(_agent: object, **kwargs: Any) -> object:
        nonlocal compacted, pending_context_tokens
        compacted = True
        pending_context_tokens = kwargs["pending_context_tokens"]
        return _result("source", tokens_before=40)

    monkeypatch.setattr("fast_agent.hooks.compaction.compact_conversation", fake_compact)

    await auto_compact_history_mid_turn(
        HookContext(
            runner=runner,
            agent=agent,
            message=_tool_use_message(),
            hook_type="before_followup_llm_call",
        )
    )

    assert compacted
    assert pending_context_tokens == 20


@pytest.mark.asyncio
@pytest.mark.parametrize("mid_turn", [False, True])
async def test_in_request_provider_compaction_suppresses_auto_compaction(
    monkeypatch: pytest.MonkeyPatch,
    mid_turn: bool,
) -> None:
    agent = _Agent("source")
    agent.llm = SimpleNamespace(
        compaction_support=CompactionSupport.IN_REQUEST,
        usage_accumulator=agent.usage_accumulator,
        compact_at_override=None,
        default_compact_at=None,
    )

    async def fail_compact(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("provider in-request compaction must replace harness auto-compaction")

    monkeypatch.setattr("fast_agent.hooks.compaction.compact_conversation", fail_compact)

    hook = auto_compact_history_mid_turn if mid_turn else auto_compact_history
    await hook(
        HookContext(
            runner=_Runner(agent),
            agent=agent,
            message=_tool_use_message() if mid_turn else _complete_message(),
            hook_type="before_followup_llm_call" if mid_turn else "after_turn_complete",
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", ["client", "custom_prompt", "unknown_window"])
async def test_inactive_provider_compaction_keeps_harness_auto_compaction(
    monkeypatch: pytest.MonkeyPatch,
    fallback: str,
) -> None:
    agent = _Agent("source")
    settings = CompactionSettings(threshold=0.5)
    if fallback == "client":
        settings.strategy = "client"
    elif fallback == "custom_prompt":
        settings.prompt = "Preserve all file paths."
    agent.context = Context(config=Settings(compaction=settings))
    agent.llm = CopilotMessagesLLM(context=agent.context, model="claude-sonnet-5.5")
    if fallback == "unknown_window":
        # The provider cannot send a trigger; harness-observed usage is still eligible.
        agent.llm.usage_accumulator.model = None
        agent.llm.usage_accumulator.set_context_window_size(None)
    assert agent.llm._in_request_compaction_edit() is None
    compacted: list[object] = []

    async def fake_compact(compacted_agent: object, **_kwargs: object) -> object:
        compacted.append(compacted_agent)
        return _result("source")

    monkeypatch.setattr("fast_agent.hooks.compaction.compact_conversation", fake_compact)

    await auto_compact_history(
        HookContext(
            runner=_Runner(agent),
            agent=agent,
            message=_complete_message(),
            hook_type="after_turn_complete",
        )
    )

    assert compacted == [agent]


class _SummaryLLM:
    verb: str | None = None
    compaction_support = CompactionSupport.CLIENT
    compact_at_override: CompactAt | None = None
    default_compact_at: CompactAt | None = None
    usage_accumulator = None

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages, request_params=None, tools=None):
        self.calls += 1
        return PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="checkpoint")],
            stop_reason=LlmStopReason.END_TURN,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [4, 200_000])
@pytest.mark.parametrize("current, compacted", [(2000, False), (4000, True)])
async def test_pending_image_compaction_uses_fixed_estimate(
    size: int,
    current: int,
    compacted: bool,
) -> None:
    from mcp_types import CallToolRequest, CallToolRequestParams, CallToolResult, ImageContent

    from fast_agent.history.compaction import estimate_tokens, is_compaction_message

    class SummaryAgent(_Agent):
        def __init__(self) -> None:
            super().__init__("images")
            self.llm = _SummaryLLM()
            self.context = Context(
                config=Settings(
                    session_history=False,
                    compaction=CompactionSettings(threshold=0.5),
                )
            )

    agent = SummaryAgent()
    agent.usage_accumulator.set_context_window_size(10_000)
    agent.usage_accumulator.set_context_estimate(current)
    history = [
        PromptMessageExtended(role=role, content=[TextContent(type="text", text="old turn")])
        for _ in range(6)
        for role in ("user", "assistant")
    ]
    pending_call = _tool_use_message()
    pending_call.tool_calls = {
        "image": CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name="image", arguments={}),
        )
    }
    history.append(pending_call)
    agent.load_message_history(history)
    runner = _Runner(agent)
    runner.delta_messages = [
        PromptMessageExtended(
            role="user",
            content=[],
            tool_results={
                "image": CallToolResult(
                    content=[ImageContent(type="image", data="AAAA" * size, mime_type="image/png")]
                )
            },
        )
    ]

    await auto_compact_history_mid_turn(
        HookContext(
            runner=runner,
            agent=agent,
            message=_tool_use_message(),
            hook_type="before_followup_llm_call",
        )
    )

    assert agent.llm.calls == int(compacted)
    assert any(is_compaction_message(msg) for msg in agent.message_history) == compacted
    if compacted:
        assert agent.usage_accumulator.current_context_tokens == (
            estimate_tokens(agent.message_history) + estimate_tokens(runner.delta_messages)
        )
    else:
        assert agent.message_history == history
        assert agent.usage_accumulator.current_context_tokens == current


@pytest.mark.asyncio
@pytest.mark.parametrize("proceeds", [False, True])
async def test_compaction_is_announced_only_once_it_starts(
    monkeypatch: pytest.MonkeyPatch, proceeds: bool
) -> None:
    from fast_agent.history.compaction import CompactionSkipped

    agent = _Agent("source")
    shown: list[str] = []

    async def fake_compact(_agent: object, *, on_start: Any, **_kwargs: object) -> object:
        if not proceeds:
            raise CompactionSkipped("Not enough conversation history to compact.")
        on_start()
        return _result("source")

    def record(_ctx: object, message: object, **_kwargs: object) -> None:
        shown.append(str(message))

    monkeypatch.setattr("fast_agent.hooks.compaction.compact_conversation", fake_compact)
    monkeypatch.setattr("fast_agent.hooks.compaction.show_hook_message", record)

    await auto_compact_history_mid_turn(
        HookContext(
            runner=_Runner(agent),
            agent=agent,
            message=_tool_use_message(),
            hook_type="before_followup_llm_call",
        )
    )

    announcements = [line for line in shown if "compacting history" in line]
    assert announcements == (["context at 90% — compacting history mid-turn"] if proceeds else [])
