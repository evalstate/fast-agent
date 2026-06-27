"""Unit tests for conversation history compaction."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ImageContent,
    TextContent,
)

from fast_agent.config import CompactionSettings, Settings, get_settings
from fast_agent.context import Context
from fast_agent.history.compaction import (
    DEFAULT_COMPACTION_PROMPT,
    FAST_AGENT_COMPACTION_CHANNEL,
    CompactionSkipped,
    _plan_compaction_with_budget,
    build_summary_message,
    compact_conversation,
    estimate_tokens,
    is_compaction_message,
    persist_compacted_session,
    plan_compaction,
    resolve_compaction_prompt,
    should_auto_compact,
)
from fast_agent.llm.usage_tracking import UsageAccumulator
from fast_agent.session import SessionManager, reset_session_manager
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.event_progress import ProgressAction


def _user(text: str, *, template: bool = False) -> PromptMessageExtended:
    msg = PromptMessageExtended(role="user", content=[TextContent(type="text", text=text)])
    msg.is_template = template
    return msg


def _tool_result(text: str) -> PromptMessageExtended:
    msg = PromptMessageExtended(role="user", content=[])
    msg.tool_results = {
        "call_1": CallToolResult(content=[TextContent(type="text", text=text)], isError=False)
    }
    return msg


def _assistant(text: str, *, tool_call: bool = False) -> PromptMessageExtended:
    msg = PromptMessageExtended(role="assistant", content=[TextContent(type="text", text=text)])
    if tool_call:
        msg.stop_reason = LlmStopReason.TOOL_USE
        msg.tool_calls = {
            "call_1": CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="search", arguments={"q": text}),
            )
        }
    else:
        msg.stop_reason = LlmStopReason.END_TURN
    return msg


def _turn(user_text: str, assistant_text: str) -> list[PromptMessageExtended]:
    return [_user(user_text), _assistant(assistant_text)]


def _tool_turn(user_text: str) -> list[PromptMessageExtended]:
    return [
        _user(user_text),
        _assistant("calling tool", tool_call=True),
        _tool_result("tool output"),
        _assistant("done"),
    ]


@pytest.mark.unit
class TestPlanCompaction:
    def test_keeps_templates_and_tail_turns(self):
        history = (
            [_user("system-ish template", template=True)]
            + _turn("one", "1")
            + _turn("two", "2")
            + _turn("three", "3")
        )

        plan = plan_compaction(history, keep_turns=2)

        assert len(plan.templates) == 1
        assert plan.templates[0].is_template
        # First turn compacted, last two kept verbatim
        assert [m.first_text() for m in plan.compact_region] == ["one", "1"]
        assert plan.retained_tail[0].first_text() == "two"
        assert len(plan.retained_tail) == 4

    def test_tail_starts_at_turn_boundary_keeping_tool_pairs(self):
        history = _turn("one", "1") + _tool_turn("two")

        plan = plan_compaction(history, keep_turns=1)

        # The tool turn stays intact: user, tool_call, tool_result, final
        assert len(plan.retained_tail) == 4
        assert plan.retained_tail[1].tool_calls
        assert plan.retained_tail[2].tool_results

    def test_always_compacts_at_least_first_turn(self):
        history = _turn("one", "1") + _turn("two", "2")

        plan = plan_compaction(history, keep_turns=5)

        assert [m.first_text() for m in plan.compact_region] == ["one", "1"]
        assert len(plan.retained_tail) == 2

    def test_keep_zero_turns_compacts_everything(self):
        history = _turn("one", "1") + _turn("two", "2")

        plan = plan_compaction(history, keep_turns=0)

        assert len(plan.compact_region) == 4
        assert plan.retained_tail == []

    def test_budget_planner_can_preserve_active_turn_tail(self):
        history = _turn("one", "1") + [_user("two"), _assistant("calling", tool_call=True)]

        plan = _plan_compaction_with_budget(
            history,
            keep_turns=0,
            max_tokens_after=None,
            min_keep_turns=1,
        )

        assert [m.first_text() for m in plan.compact_region] == ["one", "1"]
        assert [m.first_text() for m in plan.retained_tail] == ["two", "calling"]

    def test_budget_planner_skips_when_active_turn_is_only_turn(self):
        history = [_user("one"), _assistant("calling", tool_call=True)]

        with pytest.raises(CompactionSkipped):
            _plan_compaction_with_budget(
                history,
                keep_turns=0,
                max_tokens_after=None,
                min_keep_turns=1,
            )

    def test_skips_tiny_history(self):
        with pytest.raises(CompactionSkipped):
            plan_compaction([_user("hello")], keep_turns=2)

    def test_single_turn_history_compacts_fully(self):
        # A single (possibly huge) turn must remain compactable even when
        # keep_turns would otherwise retain it.
        plan = plan_compaction(_turn("one", "1"), keep_turns=2)
        assert len(plan.compact_region) == 2
        assert plan.retained_tail == []

    def test_prior_summary_is_not_counted_as_a_turn(self):
        # Regression: a leading compaction summary used to be counted as a user
        # turn, so keep_turns protected it plus the recent real turns, leaving
        # nothing to compact even at high context usage.
        summary = build_summary_message(
            "prior summary",
            prompt_text="p",
            instructions=None,
            messages_compacted=20,
            tokens_before=5000,
            context_window=100_000,
            model="m",
        )
        history = [summary] + _turn("one", "1") + _turn("two", "2")

        plan = plan_compaction(history, keep_turns=2)

        # The old summary folds into the compact region (single rolling summary),
        # the first real turn is compacted, the last real turn is kept.
        assert is_compaction_message(plan.compact_region[0])
        assert [m.first_text() for m in plan.compact_region[1:]] == ["one", "1"]
        assert [m.first_text() for m in plan.retained_tail] == ["two", "2"]


@pytest.mark.unit
class TestSummaryMessage:
    def test_round_trip_metadata(self):
        message = build_summary_message(
            "the summary",
            prompt_text="the prompt",
            instructions="focus",
            messages_compacted=10,
            tokens_before=1000,
            context_window=200_000,
            model="gpt-5.4-mini",
        )

        assert message.role == "user"
        assert is_compaction_message(message)
        assert "the summary" in message.first_text()

        assert message.channels is not None
        blocks = message.channels[FAST_AGENT_COMPACTION_CHANNEL]
        assert isinstance(blocks[0], TextContent)
        metadata = json.loads(blocks[0].text)
        assert metadata["messages_compacted"] == 10
        assert metadata["prompt"] == "the prompt"
        assert metadata["instructions"] == "focus"
        assert metadata["tokens_before"] == 1000

    def test_regular_messages_are_not_compaction(self):
        assert not is_compaction_message(_user("hello"))

    def test_survives_serialization_round_trip(self):
        message = build_summary_message(
            "the summary",
            prompt_text="p",
            instructions=None,
            messages_compacted=1,
            tokens_before=None,
            context_window=None,
            model=None,
        )
        data = message.model_dump(mode="json")
        restored = PromptMessageExtended.model_validate(data)
        assert is_compaction_message(restored)


@pytest.mark.unit
class TestShouldAutoCompact:
    def _usage(self, current: int, window: int | None) -> UsageAccumulator:
        usage = UsageAccumulator()
        usage.set_context_window_size(window)
        usage.set_context_estimate(current)
        return usage

    def test_triggers_at_threshold(self):
        settings = CompactionSettings(auto=True, threshold=0.85)
        assert should_auto_compact(self._usage(85_000, 100_000), settings)
        assert not should_auto_compact(self._usage(84_000, 100_000), settings)

    def test_disabled(self):
        settings = CompactionSettings(auto=False, threshold=0.85)
        assert not should_auto_compact(self._usage(99_000, 100_000), settings)

    def test_unknown_window_never_triggers(self):
        settings = CompactionSettings(auto=True, threshold=0.85)
        assert not should_auto_compact(self._usage(99_000, None), settings)
        assert not should_auto_compact(None, settings)


@pytest.mark.unit
class TestResolvePrompt:
    def test_default(self):
        assert resolve_compaction_prompt(None) == DEFAULT_COMPACTION_PROMPT
        assert resolve_compaction_prompt(CompactionSettings()) == DEFAULT_COMPACTION_PROMPT

    def test_inline_override(self):
        settings = CompactionSettings(prompt="Summarize tersely.")
        assert resolve_compaction_prompt(settings) == "Summarize tersely."

    def test_file_override(self, tmp_path):
        prompt_file = tmp_path / "compact.md"
        prompt_file.write_text("From file.", encoding="utf-8")
        settings = CompactionSettings(prompt=str(prompt_file))
        assert resolve_compaction_prompt(settings) == "From file."

    def test_relative_file_override_resolves_from_config_file(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config-home"
        config_dir.mkdir()
        prompt_file = config_dir / "compact.md"
        prompt_file.write_text("From config dir.", encoding="utf-8")
        settings = CompactionSettings(prompt="compact.md")
        settings._config_file = str(config_dir / "fastagent.config.yaml")

        other_cwd = tmp_path / "workspace"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)

        assert resolve_compaction_prompt(settings) == "From config dir."

    def test_relative_file_override_resolves_from_loaded_home_config(self, tmp_path, monkeypatch):
        home = tmp_path / "fast-agent-home"
        home.mkdir()
        (home / "compact.md").write_text("From FAST_AGENT_HOME config.", encoding="utf-8")
        (home / "fast-agent.yaml").write_text(
            "compaction:\n  prompt: compact.md\n",
            encoding="utf-8",
        )
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        settings = get_settings(env_dir=home)

        assert resolve_compaction_prompt(settings.compaction) == "From FAST_AGENT_HOME config."


@pytest.mark.unit
class TestEstimateTokens:
    def test_counts_tool_traffic(self):
        plain = estimate_tokens(_turn("hello there", "hi"))
        with_tools = estimate_tokens(_tool_turn("hello there"))
        assert with_tools > plain > 0

    def test_counts_non_text_content_and_channels(self):
        plain = estimate_tokens([_user("hello")])
        msg = _user("hello")
        msg.content.append(ImageContent(type="image", data="x" * 20_000, mimeType="image/png"))
        msg.channels = {"diagnostics": [TextContent(type="text", text="y" * 20_000)]}

        assert estimate_tokens([msg]) > plain + 5_000


class _FakeLLM:
    def __init__(self, summary: str = "SUMMARY OF WORK") -> None:
        self.summary = summary
        self.verb: str | ProgressAction | None = None
        self.requests: list[list[PromptMessageExtended]] = []

    async def generate(self, messages, request_params=None, tools=None):
        assert tools is None
        self.requests.append(list(messages))
        return PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text=self.summary)],
            stop_reason=LlmStopReason.END_TURN,
        )


class _FakeAgent:
    def __init__(self, history: list[PromptMessageExtended], summary: str = "SUMMARY") -> None:
        self.name = "test-agent"
        self._history = list(history)
        self.llm: Any = _FakeLLM(summary)
        self.usage_accumulator = UsageAccumulator()
        self.usage_accumulator.set_context_window_size(100_000)
        self.usage_accumulator.set_context_estimate(90_000)
        self.context: Context | None = None

    @property
    def message_history(self) -> list[PromptMessageExtended]:
        return self._history

    def load_message_history(self, messages) -> None:
        self._history = list(messages or [])


@pytest.mark.unit
@pytest.mark.anyio
class TestCompactConversation:
    async def test_compacts_history_and_sets_estimate(self):
        history = (
            [_user("template", template=True)]
            + _turn("one", "1")
            + _turn("two", "2")
            + _turn("three", "3")
        )
        agent = _FakeAgent(history, summary="checkpoint summary")
        settings = CompactionSettings(keep_turns=1)

        result = await compact_conversation(agent, settings=settings)

        # Templates + summary + last turn
        assert agent.message_history[0].is_template
        assert is_compaction_message(agent.message_history[1])
        assert "checkpoint summary" in agent.message_history[1].first_text()
        assert [m.first_text() for m in agent.message_history[2:]] == ["three", "3"]

        assert result.messages_before == 7
        assert result.messages_after == 4
        assert result.tokens_before == 90_000
        assert result.context_window == 100_000
        # Estimate override replaces the stale server-observed number
        assert agent.usage_accumulator.current_context_tokens == result.tokens_after_estimate
        assert result.tokens_after_estimate < 90_000

    async def test_summarizer_sees_compact_region_plus_prompt(self):
        history = _turn("one", "1") + _turn("two", "2") + _turn("three", "3")
        agent = _FakeAgent(history)
        settings = CompactionSettings(keep_turns=1)

        await compact_conversation(agent, settings=settings, instructions="focus on X")

        request = agent.llm.requests[0]
        assert [message.first_text() for message in request[:-1]] == ["one", "1", "two", "2"]
        final = request[-1].first_text()
        assert DEFAULT_COMPACTION_PROMPT in final
        assert "focus on X" in final

    async def test_reduces_retained_tail_when_tail_exceeds_budget(self):
        huge_tail = _user("latest huge artifact")
        huge_tail.content.append(
            ImageContent(type="image", data="x" * 300_000, mimeType="image/png")
        )
        history = _turn("one", "1") + _turn("two", "2") + [huge_tail, _assistant("after huge")]
        agent = _FakeAgent(history, summary="summary including huge artifact")
        agent.usage_accumulator.set_context_window_size(20_000)
        settings = CompactionSettings(keep_turns=2, threshold=0.85)

        await compact_conversation(agent, settings=settings)

        assert len(agent.message_history) == 1
        assert is_compaction_message(agent.message_history[0])
        request = agent.llm.requests[0]
        assert any(message.first_text() == "latest huge artifact" for message in request)

    async def test_empty_summary_leaves_history_unchanged(self):
        from fast_agent.history.compaction import CompactionError

        history = _turn("one", "1") + _turn("two", "2")
        agent = _FakeAgent(history, summary="   ")
        settings = CompactionSettings(keep_turns=1)

        with pytest.raises(CompactionError):
            await compact_conversation(agent, settings=settings)

        assert [m.first_text() for m in agent.message_history] == ["one", "1", "two", "2"]

    async def test_recompaction_of_compacted_history(self):
        history = _turn("one", "1") + _turn("two", "2") + _turn("three", "3")
        agent = _FakeAgent(history, summary="first summary")
        settings = CompactionSettings(keep_turns=1)

        await compact_conversation(agent, settings=settings)
        first_pass = list(agent.message_history)
        assert is_compaction_message(first_pass[0])

        # Add more turns and compact again
        agent._history = first_pass + _turn("four", "4") + _turn("five", "5")
        agent.llm = _FakeLLM("second summary")
        await compact_conversation(agent, settings=settings)

        summaries = [m for m in agent.message_history if is_compaction_message(m)]
        assert len(summaries) == 1
        assert "second summary" in summaries[0].first_text()
        assert agent.message_history[-2].first_text() == "five"

    async def test_archives_to_resolved_session_when_none_is_current(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
        monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)
        reset_session_manager()
        try:
            history = _turn("one", "1") + _turn("two", "2")
            agent = _FakeAgent(history, summary="checkpoint summary")
            manager = SessionManager(
                cwd=workspace,
                environment_override=workspace / ".fast-agent",
                respect_env_override=False,
            )
            agent.context = Context(config=Settings(session_history=True), session_manager=manager)
            settings = CompactionSettings(keep_turns=1)

            result = await compact_conversation(agent, settings=settings)

            assert result.archive_file is not None
            archive_path = Path(result.archive_file)
            assert archive_path.is_file()
            assert archive_path.parent.parent == workspace / ".fast-agent" / "sessions"
            assert manager.current_session is not None
            assert archive_path.name not in manager.current_session.info.history_files
        finally:
            reset_session_manager()

    async def test_noenv_skips_archive_session_writes(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
        monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)
        reset_session_manager()
        try:
            history = _turn("one", "1") + _turn("two", "2")
            agent = _FakeAgent(history, summary="checkpoint summary")
            settings_obj = Settings(session_history=True)
            settings_obj._fast_agent_noenv = True
            agent.context = Context(config=settings_obj)
            settings = CompactionSettings(keep_turns=1)

            result = await compact_conversation(agent, settings=settings)

            assert result.archive_file is None
            assert not (workspace / ".fast-agent").exists()
        finally:
            reset_session_manager()

    async def test_persist_compacted_session_explicit_noenv_short_circuits(self, monkeypatch):
        agent = _FakeAgent(_turn("one", "1") + _turn("two", "2"))

        def fail_session_persistence_enabled(_agent: object) -> bool:
            raise AssertionError("session persistence should not be checked in explicit noenv mode")

        monkeypatch.setattr(
            "fast_agent.history.compaction._session_persistence_enabled",
            fail_session_persistence_enabled,
        )

        await persist_compacted_session(agent, noenv=True)


@pytest.mark.unit
class TestUsageEstimateOverride:
    def test_add_turn_clears_estimate(self):
        from fast_agent.llm.usage_tracking import FastAgentUsage, TurnUsage

        usage = UsageAccumulator()
        usage.set_context_estimate(1234)
        assert usage.current_context_tokens == 1234

        usage.add_turn(
            TurnUsage.from_fast_agent(
                FastAgentUsage(
                    input_chars=400,
                    output_chars=40,
                    model_type="fake",
                    tool_calls=0,
                    delay_seconds=0.0,
                ),
                model="fake",
            )
        )
        assert usage.current_context_tokens != 1234

    def test_context_estimate_does_not_change_cumulative_billing_totals(self):
        from fast_agent.llm.usage_tracking import FastAgentUsage, TurnUsage

        usage = UsageAccumulator()
        usage.add_turn(
            TurnUsage.from_fast_agent(
                FastAgentUsage(
                    input_chars=400,
                    output_chars=40,
                    model_type="fake",
                    tool_calls=0,
                    delay_seconds=0.0,
                ),
                model="fake",
            )
        )

        assert usage.cumulative_input_tokens == 400
        assert usage.cumulative_output_tokens == 40
        assert usage.cumulative_billing_tokens == 440

        usage.set_context_estimate(123)

        assert usage.current_context_tokens == 123
        assert usage.cumulative_input_tokens == 400
        assert usage.cumulative_output_tokens == 40
        assert usage.cumulative_billing_tokens == 440
        assert usage.get_summary()["current_context_tokens"] == 123
        assert usage.get_summary()["cumulative_billing_tokens"] == 440
