"""OpenAI provider-native compaction on the codexresponses route (no network).

Stream replays use sanitized live captures from
``scripts/openai_native_compaction_smoke.py --fixture-out``: SSE leaves
``response.completed.output`` empty, WebSocket includes it and Codex-only events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import CallToolRequest, CallToolRequestParams, TextContent

from fast_agent.config import (
    CodexResponsesSettings,
    CompactionSettings,
    CopilotSettings,
    OpenAISettings,
    Settings,
)
from fast_agent.constants import OPENAI_COMPACTION_ITEMS
from fast_agent.context import Context
from fast_agent.core.exceptions import ProviderCheckpointError
from fast_agent.core.logging.listeners import convert_log_event
from fast_agent.core.logging.logger import Logger
from fast_agent.event_progress import ProgressAction
from fast_agent.history.compaction import (
    CompactionError,
    build_summary_message,
    compact_conversation,
)
from fast_agent.llm.compaction_support import CompactionStrategy, CompactionSupport
from fast_agent.llm.provider.copilot.responses import CopilotResponsesLLM
from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.provider.openai.compaction_replay import capture_compaction_replay
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.responses_cache import prepare_cached_input
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason
from tests.support.llm_trace_replay import ResponsesReplayStream

if TYPE_CHECKING:
    from fast_agent.core.logging.events import Event

FIXTURES = (
    Path(__file__).resolve().parents[5]
    / "tests/fixtures/llm_traces/sanitized/codexresponses/gpt-6-astra"
)
TRANSPORTS = ("sse", "websocket")


def _llm(*, native: bool = True) -> CodexResponsesLLM:
    settings = Settings(codexresponses=CodexResponsesSettings(native_compaction=native))
    return CodexResponsesLLM(context=Context(config=settings), model="gpt-6-astra")


def _replay_stream(transport: str) -> ResponsesReplayStream:
    path = FIXTURES / f"compaction_trigger_{transport}" / "stream.jsonl"
    return ResponsesReplayStream([json.loads(line) for line in path.read_text().splitlines()])


async def _replayed_response(llm: ResponsesLLM, transport: str) -> Any:
    # Deliberately not normalized to an SDK ``Response``: like the live WebSocket
    # transport, the replay yields attribute views over raw JSON, and capture must
    # handle exactly what ``_process_stream`` returns.
    response, _ = await llm._process_stream(
        _replay_stream(transport), "gpt-6-astra", capture_filename=None
    )
    return response


def _checkpoint(items: list[Any]) -> PromptMessageExtended:
    return build_summary_message(
        "OpenAI compaction checkpoint",
        strategy=CompactionStrategy.OPENAI_COMPACTION,
        provider_channels={OPENAI_COMPACTION_ITEMS: items},
        prompt_text=None,
        instructions=None,
        messages_compacted=3,
        tokens_before=4_000,
        context_window=272_000,
        model="gpt-6-astra",
    )


def _user(text: str) -> PromptMessageExtended:
    return PromptMessageExtended(role="user", content=[TextContent(type="text", text=text)])


def _route_llm(route: str, *, native: bool) -> ResponsesLLM:
    settings = Settings(
        codexresponses=CodexResponsesSettings(native_compaction=native),
        responses=OpenAISettings(api_key="test", native_compaction=native),
        copilot=CopilotSettings(native_compaction=native),
    )
    context = Context(config=settings)
    match route:
        case "codexresponses":
            return CodexResponsesLLM(context=context, model="gpt-6-sol")
        case "responses":
            return ResponsesLLM(context=context, model="gpt-6-sol")
        case _:
            return CopilotResponsesLLM(context=context, model="gpt-6-sol")


@pytest.mark.parametrize("route", ["codexresponses", "responses", "copilot"])
@pytest.mark.parametrize(("native", "expected"), [(False, "client"), (True, "on_demand")])
def test_route_setting_controls_native_compaction(route: str, native: bool, expected: str) -> None:
    assert _route_llm(route, native=native).compaction_support == CompactionSupport(expected)


@pytest.mark.parametrize(
    "llm_class", [CodexResponsesLLM, ResponsesLLM, CopilotResponsesLLM], ids=lambda c: c.__name__
)
def test_native_compaction_defaults_on_without_a_route_section(llm_class: type) -> None:
    # Copilot credentials come from its broker; only the API-key route takes one.
    credentials = {"api_key": "test"} if llm_class is ResponsesLLM else {}
    llm = llm_class(context=Context(config=Settings()), model="gpt-6-sol", **credentials)

    assert llm.compaction_support == CompactionSupport.ON_DEMAND


@pytest.mark.parametrize("route", ["codexresponses", "responses", "copilot"])
@pytest.mark.parametrize(
    ("setting", "mode", "expected"),
    [
        # The model string wins over the route setting, in both directions.
        (True, "client", "client"),
        (False, "native", "on_demand"),
        (False, None, "client"),
    ],
)
def test_model_string_compaction_overrides_the_route_setting(
    route: str, setting: bool, mode: str | None, expected: str
) -> None:
    settings = Settings(
        codexresponses=CodexResponsesSettings(native_compaction=setting),
        responses=OpenAISettings(api_key="test", native_compaction=setting),
        copilot=CopilotSettings(native_compaction=setting),
    )
    llm_class = {
        "codexresponses": CodexResponsesLLM,
        "responses": ResponsesLLM,
        "copilot": CopilotResponsesLLM,
    }[route]
    llm = llm_class(context=Context(config=settings), model="gpt-6-sol", compaction=mode)

    assert llm.compaction_support == CompactionSupport(expected)


def test_other_responses_family_providers_never_compact_natively() -> None:
    # MetaAI shares the Responses implementation but is not verified for compaction.
    settings = Settings(responses=OpenAISettings(api_key="test", native_compaction=True))
    llm = ResponsesLLM(
        context=Context(config=settings), model="gpt-6-sol", provider=Provider.META_AI
    )

    assert llm.compaction_support == CompactionSupport.CLIENT


class _RecordingLogger(Logger):
    def __init__(self) -> None:
        super().__init__("fast_agent.llm.provider.openai.responses")
        self.events: list[Event] = []

    def _emit_event(self, event: Event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize(
    ("verb", "expected"),
    [
        (None, ProgressAction.COMPACTING_IN_REQUEST),
        (ProgressAction.COMPACTING_CHECKPOINT, ProgressAction.COMPACTING_CHECKPOINT),
    ],
)
async def test_compaction_stream_replays_into_a_replayable_checkpoint(
    transport: str, verb: ProgressAction | None, expected: ProgressAction
) -> None:
    llm = _llm()
    llm.verb = verb
    logger = _RecordingLogger()
    llm.logger = logger

    response = await _replayed_response(llm, transport)
    blocks = llm._capture_compaction_blocks(response, origin="codexresponses")
    replayed = llm._convert_message_to_items(_checkpoint(blocks))

    progress = [convert_log_event(event) for event in logger.events]
    assert expected in [event.action for event in progress if event is not None]
    # Only the input fields the API accepts back; nothing else is replayed.
    assert [sorted(item) for item in replayed] == [["encrypted_content", "id", "type"]]
    assert replayed[0]["type"] == "compaction"
    assert replayed[0]["id"].startswith("cmp_")


def test_compaction_only_reply_is_not_empty() -> None:
    reply = PromptMessageExtended(
        role="assistant",
        content=[],
        stop_reason=LlmStopReason.END_TURN,
        channels=_checkpoint_with_item().channels,
    )

    assert not CodexResponsesLLM._is_empty_responses_completion(reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_native_compaction_request_ends_with_trigger_and_returns_checkpoint(
    transport: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _llm()
    blocks = llm._capture_compaction_blocks(
        await _replayed_response(llm, transport), origin="codexresponses"
    )
    sent: list[tuple[list[dict[str, Any]], Any]] = []

    async def fake_completion(input_items: list[dict[str, Any]], params: Any, tools: Any = None):
        sent.append((input_items, tools))
        return PromptMessageExtended(
            role="assistant",
            content=[],
            stop_reason=LlmStopReason.END_TURN,
            channels={OPENAI_COMPACTION_ITEMS: blocks},
        )

    monkeypatch.setattr(llm, "_responses_completion", fake_completion)

    native = await llm.native_compaction([_user("hello")])

    [(input_items, tools)] = sent
    assert input_items[-1] == {"type": "compaction_trigger"}
    assert tools is None
    assert native.strategy == "openai_compaction"
    assert native.checkpoint_channels == {OPENAI_COMPACTION_ITEMS: blocks}
    # Sized from the (redacted) encrypted payload, not the envelope JSON.
    assert native.opaque_bytes == len("REDACTED-ENCRYPTED-COMPACTION-CHECKPOINT")


@pytest.mark.asyncio
async def test_native_compaction_without_items_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _llm()

    async def fake_completion(*_args: Any, **_kwargs: Any) -> PromptMessageExtended:
        return PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="no checkpoint")],
            stop_reason=LlmStopReason.END_TURN,
        )

    monkeypatch.setattr(llm, "_responses_completion", fake_completion)

    with pytest.raises(RuntimeError, match="no compaction item"):
        await llm.native_compaction([_user("hello")])


def _effort_updates(items: list[dict[str, Any]]) -> list[tuple[int, str]]:
    return [
        (index, item["reasoning"]["effort"])
        for index, item in enumerate(items)
        if item.get("type") == "configuration_update"
    ]


@pytest.mark.parametrize("next_turn", ["user", "tool_continuation"])
def test_astra_effort_is_resent_after_a_checkpoint(next_turn: str) -> None:
    """OpenAI: after compaction, send a fresh update before the next user message."""
    llm = _llm()
    checkpoint = _checkpoint_with_item()
    following = (
        [_user("next question")]
        if next_turn == "user"
        else [PromptMessageExtended(role="user", content=[], tool_results={})]
    )

    items, state, _ = prepare_cached_input(
        [checkpoint, *following],
        model="gpt-6-astra",
        key="key",
        effort="medium",
        convert=llm._convert_message_to_items,
    )

    types = [item.get("type") or item.get("role") for item in items]
    if next_turn == "user":
        # Same effort as the request baseline, yet re-sent: the checkpoint resets it.
        assert types == ["compaction", "configuration_update", "message"]
        assert _effort_updates(items) == [(1, "medium")]
    else:
        assert _effort_updates(items) == []
    assert state.effort == "medium"


def _checkpoint_with_item(origin: str | None = "codexresponses") -> PromptMessageExtended:
    envelope: dict[str, object] | None = (
        dict(
            capture_compaction_replay(
                {"type": "compaction", "id": "cmp_1", "encrypted_content": "x"}, origin=origin
            )
            or {}
        )
        if origin is not None
        else {
            # Envelopes written before the producing route was recorded.
            "schema": "fast-agent.openai-responses.compaction-replay",
            "version": 1,
            "item": {"type": "compaction", "id": "cmp_1", "encrypted_content": "x"},
        }
    )
    assert envelope
    return _checkpoint([TextContent(type="text", text=json.dumps(envelope))])


def test_unannotated_retained_turn_after_checkpoint_gets_the_update() -> None:
    """Imported/unannotated retained turns still receive the post-compaction update."""
    llm = _llm()
    retained = [
        _user("retained question"),
        PromptMessageExtended(role="assistant", content=[TextContent(type="text", text="a")]),
    ]

    items, state, _ = prepare_cached_input(
        [_checkpoint_with_item(), *retained, _user("new question")],
        model="gpt-6-astra",
        key="key",
        effort="medium",
        convert=llm._convert_message_to_items,
    )

    types = [item.get("type") or item.get("role") for item in items]
    # Exactly one update, before the first user message after the checkpoint.
    assert types == ["compaction", "configuration_update", "message", "message", "message"]
    assert state.effort == "medium"


@pytest.mark.parametrize("provider", [Provider.META_AI, Provider.DEEPSEEK])
def test_other_providers_receive_the_checkpoint_text_not_the_encrypted_item(
    provider: Provider,
) -> None:
    settings = Settings(responses=OpenAISettings(api_key="test"))
    llm = ResponsesLLM(context=Context(config=settings), model="gpt-6-sol", provider=provider)

    [item] = llm._convert_message_to_items(_checkpoint_with_item())

    assert item["type"] == "message"
    assert "[COMPACTED HISTORY]" in json.dumps(item)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "metadata", "resent"),
    [
        ("gpt-6-astra", None, "medium"),
        # The request's own effort wins over the configured default.
        ("gpt-6-astra", {"reasoning": {"effort": "high"}}, "high"),
        ("gpt-6-sol", None, None),
    ],
)
async def test_copilot_astra_resends_effort_after_a_checkpoint(
    model: str, metadata: dict[str, Any] | None, resent: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = CopilotResponsesLLM(context=Context(config=Settings()), model=model)
    sent: list[list[dict[str, Any]]] = []

    async def fake_completion(input_items: list[dict[str, Any]], *_args: Any) -> Any:
        sent.append(input_items)
        return PromptMessageExtended(
            role="assistant", content=[TextContent(type="text", text="ok")]
        )

    monkeypatch.setattr(llm, "_responses_completion", fake_completion)

    await llm._apply_prompt_provider_specific(
        [_checkpoint_with_item(), _user("question")], RequestParams(metadata=metadata)
    )

    [input_items] = sent
    types = [item.get("type") or item.get("role") for item in input_items]
    assert types == (
        ["compaction", "configuration_update", "message"] if resent else ["compaction", "message"]
    )
    assert _effort_updates(input_items) == ([(1, resent)] if resent else [])


@pytest.mark.parametrize("payload", ["", "{", "[]", "{}", '{"version": 99}'])
def test_unreadable_checkpoint_replay_requires_archive_recovery(payload: str) -> None:
    llm = _llm()
    # One valid item must not hide a corrupt item later in the same channel.
    blocks = list((_checkpoint_with_item().channels or {})[OPENAI_COMPACTION_ITEMS])
    blocks.append(TextContent(type="text", text=payload))
    with pytest.raises(ProviderCheckpointError, match=r"unreadable.*?/history load <archive>"):
        llm._convert_message_to_items(_checkpoint(blocks))


@pytest.mark.parametrize("missing_channel", [False, True])
def test_checkpoint_without_items_is_not_silently_erased(missing_channel: bool) -> None:
    checkpoint = _checkpoint([])
    assert checkpoint.channels is not None
    if missing_channel:
        checkpoint.channels = {
            name: blocks
            for name, blocks in checkpoint.channels.items()
            if name != OPENAI_COMPACTION_ITEMS
        }
    with pytest.raises(ProviderCheckpointError, match="unreadable"):
        _llm()._convert_message_to_items(checkpoint)


def test_ordinary_response_replays_compaction_text_and_tool_call() -> None:
    llm = _llm()
    checkpoint = _checkpoint_with_item()
    reply = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="Checking the workspace.")],
        tool_calls={
            "call_1": CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="read_file", arguments={"path": "notes.md"}),
            )
        },
        channels={OPENAI_COMPACTION_ITEMS: (checkpoint.channels or {})[OPENAI_COMPACTION_ITEMS]},
        stop_reason=LlmStopReason.TOOL_USE,
    )

    items = llm._convert_message_to_items(reply)

    assert [item["type"] for item in items] == ["compaction", "message", "function_call"]
    assert "Checking the workspace." in json.dumps(items[1])
    assert items[2]["name"] == "read_file"
    assert json.loads(items[2]["arguments"]) == {"path": "notes.md"}
    assert items[2]["call_id"] == "call_1"


class _CompactionAgent:
    """Minimal harness surface backed by the real provider, not a fake native LLM."""

    name = "codex-compaction"

    def __init__(self, llm: CodexResponsesLLM) -> None:
        self.llm = llm
        self.context = llm.context
        assert self.context.config is not None
        self.context.config.session_history = False
        self.usage_accumulator = llm.usage_accumulator
        self.message_history = [
            _user("template instructions"),
            _user("old question one"),
            PromptMessageExtended(
                role="assistant", content=[TextContent(type="text", text="old answer one")]
            ),
            _user("old question two"),
            PromptMessageExtended(
                role="assistant", content=[TextContent(type="text", text="old answer two")]
            ),
            _user("retained question"),
            PromptMessageExtended(
                role="assistant", content=[TextContent(type="text", text="retained answer")]
            ),
        ]
        self.message_history[0].is_template = True

    def load_message_history(self, messages: list[PromptMessageExtended] | None) -> None:
        self.message_history = list(messages or [])


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, "", "{", "[]", "{}"])
async def test_invalid_native_capture_leaves_harness_history_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    payload: str | None,
) -> None:
    llm = _llm()
    agent = _CompactionAgent(llm)
    history = list(agent.message_history)
    blocks = list((_checkpoint_with_item().channels or {})[OPENAI_COMPACTION_ITEMS])
    if payload is None:
        blocks = []
    else:
        blocks.append(TextContent(type="text", text=payload))

    async def fake_completion(*_args: Any, **_kwargs: Any) -> PromptMessageExtended:
        return PromptMessageExtended(
            role="assistant",
            content=[],
            stop_reason=LlmStopReason.END_TURN,
            channels={OPENAI_COMPACTION_ITEMS: blocks},
        )

    monkeypatch.setattr(llm, "_responses_completion", fake_completion)

    with pytest.raises(CompactionError, match="history unchanged"):
        await compact_conversation(agent, settings=CompactionSettings(keep_turns=1))
    assert agent.message_history == history


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_harness_native_checkpoint_round_trip(
    transport: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _llm()
    llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="medium"))
    agent = _CompactionAgent(llm)
    sent: list[list[dict[str, Any]]] = []
    planner = llm._new_ws_request_planner()
    planned_arguments: list[dict[str, Any]] = []

    async def fake_transport(
        *,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: Any,
        model_name: str,
    ) -> tuple[Any, list[str], list[dict[str, Any]]]:
        sent.append(input_items)
        arguments = llm._build_response_args(input_items, request_params, tools)
        planned = planner.plan(arguments)
        planned_arguments.append(planned.arguments)
        # Drive real streaming normalization, output capture, and finalization.
        response = await _replayed_response(llm, transport)
        planner.commit(arguments, planned, response)
        if len(sent) == 1:
            # Prove the planner has a usable response-id chain before compacting.
            continuation = planner.plan(
                {
                    **arguments,
                    "input": [*input_items, *llm._convert_message_to_items(_user("followup"))],
                }
            )
            assert continuation.arguments["previous_response_id"] == response.id
        return response, [], input_items

    # Below completion: keep the real request preparation and output-capture boundaries.
    monkeypatch.setattr(llm, "_responses_completion_sse", fake_transport)
    monkeypatch.setattr(llm, "_responses_completion_ws", fake_transport)

    # Establish a real pre-compaction response-id chain.
    await llm.generate(agent.message_history[:-1], RequestParams())
    result = await compact_conversation(agent, settings=CompactionSettings(keep_turns=1))
    await llm.generate([*agent.message_history, _user("new question")], RequestParams())

    assert result.is_native
    assert len(sent) == 3
    assert sent[1][-1] == {"type": "compaction_trigger"}
    next_items = sent[2]
    assert [item["type"] for item in next_items] == [
        "message",
        "compaction",
        "configuration_update",
        "message",
        "message",
        "message",
    ]
    assert next_items[2]["reasoning"]["effort"] == "medium"
    assert [
        part["text"] for item in next_items if item["type"] == "message" for part in item["content"]
    ] == ["template instructions", "retained question", "retained answer", "new question"]
    assert next_items[1]["id"].startswith("cmp_")
    assert "previous_response_id" not in planned_arguments[2]
    assert planned_arguments[2]["input"] == next_items


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_reply_with_compaction_item_is_marked_as_a_provider_checkpoint(
    transport: str,
) -> None:
    from fast_agent.constants import FAST_AGENT_PROVIDER_CHECKPOINT

    llm = _llm()
    response = await _replayed_response(llm, transport)

    channels = llm._responses_reasoning_channels(response, []) or {}

    # History rewrites (process-poll folding) must keep checkpoint-bearing replies.
    assert FAST_AGENT_PROVIDER_CHECKPOINT in channels
    assert OPENAI_COMPACTION_ITEMS in channels


def _destination_llm(route: str) -> ResponsesLLM:
    settings = Settings(responses=OpenAISettings(api_key="test"))
    context = Context(config=settings)
    match route:
        case "codexresponses":
            return CodexResponsesLLM(context=context, model="gpt-6-sol")
        case "responses":
            return ResponsesLLM(context=context, model="gpt-6-sol")
        case _:
            return CopilotResponsesLLM(context=context, model="gpt-6-sol")


# Verified live: Codex and API-key checkpoints work on all three routes; Copilot's use
# Copilot-bound ids and encryption that OpenAI's routes reject (HTTP 400).
@pytest.mark.parametrize(
    ("origin", "destination", "readable"),
    [
        ("codexresponses", "responses", True),
        ("codexresponses", "copilot", True),
        ("responses", "codexresponses", True),
        ("responses", "copilot", True),
        ("copilot", "copilot", True),
        ("copilot", "codexresponses", False),
        ("copilot", "responses", False),
        # Earlier envelopes without an origin keep replaying.
        (None, "codexresponses", True),
    ],
)
def test_checkpoint_replays_only_where_its_origin_is_readable(
    origin: str | None, destination: str, readable: bool
) -> None:
    llm = _destination_llm(destination)
    checkpoint = _checkpoint_with_item(origin)

    if readable:
        [item] = llm._convert_message_to_items(checkpoint)
        assert item["type"] == "compaction"
    else:
        with pytest.raises(ProviderCheckpointError, match=r"compacted on 'copilot'"):
            llm._convert_message_to_items(checkpoint)


@pytest.mark.asyncio
async def test_captured_checkpoint_records_the_producing_route() -> None:
    llm = _destination_llm("copilot")
    response = await _replayed_response(llm, "websocket")

    channels = llm._responses_reasoning_channels(response, []) or {}

    [block] = channels[OPENAI_COMPACTION_ITEMS]
    assert isinstance(block, TextContent)
    assert json.loads(block.text)["origin"] == "copilot"
