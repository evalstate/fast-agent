from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.provider.openai.openresponses_streaming import OpenResponsesStreamingMixin
from fast_agent.llm.provider.openai.responses_streaming import ResponsesStreamingMixin

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "tests" / "support").is_dir()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_replay = importlib.import_module("tests.support.llm_trace_replay")


class _BaseStreamingHarness:
    def __init__(self) -> None:
        self.logger = get_logger("test.responses.replay")
        self.name = "test"
        self.stream_events: list[dict[str, Any]] = []
        self.tool_events: list[dict[str, Any]] = []

    def _notify_tool_stream_listeners(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.tool_events.append(
            {
                "event_type": event_type,
                "payload": payload or {},
            }
        )

    def _notify_stream_listeners(self, chunk: Any) -> None:
        self.stream_events.append(
            {
                "text": getattr(chunk, "text", ""),
                "is_reasoning": bool(getattr(chunk, "is_reasoning", False)),
            }
        )

    def _update_streaming_progress(
        self,
        content: str,
        model: str,
        estimated_tokens: int,
    ) -> int:
        del content, model
        return estimated_tokens

    def chat_turn(self) -> int:
        return 1

    async def _emit_streaming_progress(
        self,
        model: str,
        new_total: int,
        type: Any,
    ) -> None:
        del model, new_total, type

    def _emit_stream_text_delta(
        self,
        *,
        text: str,
        model: str,
        estimated_tokens: int,
    ) -> int:
        self._notify_stream_listeners(SimpleNamespace(text=text, is_reasoning=False))
        self._notify_tool_stream_listeners("text", {"chunk": text})
        del model
        return estimated_tokens


class _ResponsesHarness(ResponsesStreamingMixin, _BaseStreamingHarness):
    def __init__(self) -> None:
        _BaseStreamingHarness.__init__(self)


class _OpenResponsesHarness(OpenResponsesStreamingMixin, _BaseStreamingHarness):
    def __init__(self) -> None:
        _BaseStreamingHarness.__init__(self)


class _FakeResponsesStream:
    def __init__(self, events: list[Any], final_response: Any) -> None:
        self._events = events
        self._final_response = final_response

    def __aiter__(self):
        self._iterator = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def get_final_response(self) -> Any:
        return self._final_response


def _fixture_params() -> list[Any]:
    params: list[Any] = []
    for case, fixture in _replay.load_replay_fixtures("responses", "openresponses"):
        if case.family == "responses":
            harness_factory = _ResponsesHarness
        elif case.family == "openresponses":
            harness_factory = _OpenResponsesHarness
        else:  # pragma: no cover - family filter guards this
            continue
        params.append(pytest.param(case, fixture, harness_factory, id=case.id))
    return params


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "fixture", "harness_factory"),
    _fixture_params(),
)
async def test_responses_family_stream_replay(
    case: Any,
    fixture: Any,
    harness_factory: Any,
) -> None:
    harness: Any = harness_factory()
    final_response, _reasoning_segments = await harness._process_stream(
        fixture.responses_stream(),
        model=fixture.meta()["resolved_model"],
        capture_filename=None,
    )

    summary = _replay.summarize_responses_replay(
        final_response=final_response,
        stream_events=harness.stream_events,
        tool_events=harness.tool_events,
    )
    _replay.assert_replay_case(case, fixture, summary)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openresponses_status_before_added_emits_single_start() -> None:
    harness = _OpenResponsesHarness()
    final_response = SimpleNamespace(output=[], usage=None)
    stream = _FakeResponsesStream(
        events=[
            SimpleNamespace(
                type="response.function_call.started",
                output_index=0,
                item_id="fc_123",
            ),
            SimpleNamespace(
                type="response.output_item.added",
                output_index=0,
                item_id="fc_123",
                item=SimpleNamespace(
                    type="function_call",
                    id="fc_123",
                    call_id="call_123",
                    name="weather",
                ),
            ),
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item_id="fc_123",
                item=SimpleNamespace(
                    type="function_call",
                    id="fc_123",
                    call_id="call_123",
                    name="weather",
                ),
            ),
            SimpleNamespace(type="response.completed", response=final_response),
        ],
        final_response=final_response,
    )

    await harness._process_stream(stream, model="gpt-test", capture_filename=None)

    start_events = [
        event for event in harness.tool_events if event["event_type"] == "start"
    ]
    assert len(start_events) == 1
    assert start_events[0]["payload"]["tool_use_id"] == "call_123"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openresponses_status_after_added_uses_registered_tool_state() -> None:
    harness = _OpenResponsesHarness()
    final_response = SimpleNamespace(output=[], usage=None)
    stream = _FakeResponsesStream(
        events=[
            SimpleNamespace(
                type="response.output_item.added",
                output_index=0,
                item_id="fc_123",
                item=SimpleNamespace(
                    type="function_call",
                    id="fc_123",
                    call_id="call_123",
                    name="weather",
                ),
            ),
            SimpleNamespace(
                type="response.function_call.in_progress",
                output_index=0,
                item_id="fc_123",
            ),
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item_id="fc_123",
                item=SimpleNamespace(
                    type="function_call",
                    id="fc_123",
                    call_id="call_123",
                    name="weather",
                ),
            ),
            SimpleNamespace(type="response.completed", response=final_response),
        ],
        final_response=final_response,
    )

    await harness._process_stream(stream, model="gpt-test", capture_filename=None)

    start_events = [
        event for event in harness.tool_events if event["event_type"] == "start"
    ]
    status_events = [
        event for event in harness.tool_events if event["event_type"] == "status"
    ]
    stop_events = [
        event for event in harness.tool_events if event["event_type"] == "stop"
    ]
    assert len(start_events) == 1
    assert start_events[0]["payload"]["tool_use_id"] == "call_123"
    assert len(status_events) == 1
    assert status_events[0]["payload"]["tool_use_id"] == "call_123"
    assert status_events[0]["payload"]["item_id"] == "fc_123"
    assert len(stop_events) == 1
    assert stop_events[0]["payload"]["tool_use_id"] == "call_123"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openresponses_out_of_order_tool_events_are_ignored() -> None:
    harness = _OpenResponsesHarness()
    final_response = SimpleNamespace(output=[], usage=None)
    stream = _FakeResponsesStream(
        events=[
            SimpleNamespace(
                type="response.function_call.in_progress",
                item_id="fc_123",
            ),
            SimpleNamespace(
                type="response.output_item.done",
                output_index=5,
                item_id="fc_123",
                item=SimpleNamespace(
                    type="function_call",
                    id="fc_123",
                    call_id="call_123",
                    name="weather",
                ),
            ),
            SimpleNamespace(type="response.completed", response=final_response),
        ],
        final_response=final_response,
    )

    await harness._process_stream(stream, model="gpt-test", capture_filename=None)

    stop_events = [
        event for event in harness.tool_events if event["event_type"] == "stop"
    ]
    assert stop_events == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("harness_factory", "response_event_type"),
    [
        pytest.param(_ResponsesHarness, "response.incomplete", id="responses"),
        pytest.param(_OpenResponsesHarness, "response.incomplete", id="openresponses"),
    ],
)
async def test_incomplete_responses_return_final_payload(
    harness_factory: Any,
    response_event_type: str,
) -> None:
    harness = harness_factory()
    final_response = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=[],
        usage=None,
    )
    stream = _FakeResponsesStream(
        events=[
            SimpleNamespace(
                type="response.output_item.added",
                output_index=0,
                item_id="fc_123",
                item=SimpleNamespace(
                    type="function_call",
                    id="fc_123",
                    call_id="call_123",
                    name="weather",
                ),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                output_index=0,
                item_id="fc_123",
                delta="{\"city\":\"Paris\"",
            ),
            SimpleNamespace(type=response_event_type, response=final_response),
        ],
        final_response=final_response,
    )

    returned_response, _reasoning_segments = await harness._process_stream(
        stream,
        model="gpt-test",
        capture_filename=None,
    )

    assert returned_response is final_response
