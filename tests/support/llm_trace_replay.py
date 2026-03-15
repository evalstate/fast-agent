from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from anthropic.lib.streaming._beta_messages import ParsedBetaMessageStreamEvent
from google.genai import types as google_types
from openai.types.responses import Response
from pydantic import TypeAdapter

from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "llm_traces"
SANITIZED_FIXTURE_ROOT = TRACE_FIXTURE_ROOT / "sanitized"
MANIFEST_ROOT = TRACE_FIXTURE_ROOT / "manifests"
REPLAY_CASES_PATH = MANIFEST_ROOT / "replay_cases.json"

VALID_FAMILIES = {
    "anthropic",
    "google",
    "openai-chat",
    "openresponses",
    "responses",
}
VALID_ASSERTION_PROFILES = {
    "plain_text",
    "tool_use",
    "web_search",
}
_ANTHROPIC_EVENT_ADAPTER = TypeAdapter(ParsedBetaMessageStreamEvent)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class AttrObjectView:
    """Expose mapping keys as attributes recursively."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = {key: _to_attr_object(value) for key, value in data.items()}

    def __getattr__(self, key: str) -> Any:
        if key in self._data:
            return self._data[key]
        raise AttributeError(key)

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return _to_plain_data(self._data)

    def __repr__(self) -> str:
        return f"AttrObjectView({self._data!r})"


def _to_attr_object(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrObjectView(value)
    if isinstance(value, list):
        return [_to_attr_object(item) for item in value]
    return value


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, AttrObjectView):
        return {key: _to_plain_data(item) for key, item in value._data.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    return value


def _normalize_stop_reason(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    normalized = str(raw).replace("-", "_").strip().lower()
    if normalized == "tooluse":
        return "tool_use"
    if normalized == "endturn":
        return "end_turn"
    return normalized


def _json_argument_payload(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


@dataclass(frozen=True, slots=True)
class ReplayCase:
    id: str
    family: str
    model_label: str
    scenario: str
    fixture_dir: str
    assertion_profile: str
    notes: str | None = None
    skip_reason: str | None = None

    @property
    def fixture_path(self) -> Path:
        return SANITIZED_FIXTURE_ROOT / self.fixture_dir


@dataclass(frozen=True, slots=True)
class ToolCallSummary:
    name: str
    arguments: Any


@dataclass(slots=True)
class ReplaySummary:
    stop_reason: str | None
    assistant_text: str
    tool_calls: list[ToolCallSummary]
    stream_events: list[dict[str, Any]]
    tool_events: list[dict[str, Any]]


class ResponsesReplayStream:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._events = [_to_attr_object(payload) for payload in payloads]
        self._index = 0
        response_payload = _final_responses_payload(payloads)
        self._final_response = Response.model_validate(response_payload)

    def __aiter__(self) -> ResponsesReplayStream:
        self._index = 0
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event

    async def get_final_response(self) -> Response:
        return self._final_response


class AnthropicReplayStream:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = payloads
        self._events = [_ANTHROPIC_EVENT_ADAPTER.validate_python(payload) for payload in payloads]
        self._index = 0
        self._final_message = _final_anthropic_message(payloads)

    def __aiter__(self) -> AnthropicReplayStream:
        self._index = 0
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event

    async def get_final_message(self) -> Any:
        return self._final_message


class GoogleReplayStream:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._events = [
            google_types.GenerateContentResponse.model_validate(payload) for payload in payloads
        ]
        self._index = 0
        self.closed = False

    def __aiter__(self) -> GoogleReplayStream:
        self._index = 0
        return self

    async def __anext__(self) -> google_types.GenerateContentResponse:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event

    async def aclose(self) -> None:
        self.closed = True


class ReplayFixture:
    def __init__(self, case: ReplayCase) -> None:
        self.case = case
        self.path = case.fixture_path
        self.validate()

    def validate(self) -> None:
        if not self.path.is_dir():
            raise FileNotFoundError(f"Missing replay fixture directory: {self.path}")
        for name in ("meta.json", "result.json", "stream.jsonl"):
            if not (self.path / name).exists():
                raise FileNotFoundError(f"Missing required replay fixture file: {self.path / name}")

        meta = self.meta()
        if meta.get("family") != self.case.family:
            raise ValueError(
                f"Fixture family mismatch for {self.case.id}: "
                f"{meta.get('family')} != {self.case.family}"
            )
        if meta.get("model_label") != self.case.model_label:
            raise ValueError(
                f"Fixture model_label mismatch for {self.case.id}: "
                f"{meta.get('model_label')} != {self.case.model_label}"
            )
        if meta.get("scenario") != self.case.scenario:
            raise ValueError(
                f"Fixture scenario mismatch for {self.case.id}: "
                f"{meta.get('scenario')} != {self.case.scenario}"
            )

    def meta(self) -> dict[str, Any]:
        return _read_json(self.path / "meta.json")

    def result_message(self) -> PromptMessageExtended:
        return PromptMessageExtended.model_validate(_read_json(self.path / "result.json"))

    def stream_payloads(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.path / "stream.jsonl")

    def listener_stream_events(self) -> list[dict[str, Any]]:
        path = self.path / "listener_stream.jsonl"
        if not path.exists():
            return []
        return _read_jsonl(path)

    def listener_tool_events(self) -> list[dict[str, Any]]:
        path = self.path / "listener_tools.jsonl"
        if not path.exists():
            return []
        return _read_jsonl(path)

    def responses_stream(self) -> ResponsesReplayStream:
        return ResponsesReplayStream(self.stream_payloads())

    def anthropic_stream(self) -> AnthropicReplayStream:
        return AnthropicReplayStream(self.stream_payloads())

    def google_stream(self) -> GoogleReplayStream:
        return GoogleReplayStream(self.stream_payloads())


def load_replay_cases(
    *,
    manifest_path: Path = REPLAY_CASES_PATH,
    families: Iterable[str] | None = None,
) -> list[ReplayCase]:
    payload = _read_json(manifest_path)
    version = payload.get("version")
    if version != 1:
        raise ValueError(f"Unsupported replay manifest version: {version!r}")

    family_filter = set(families or [])
    cases: list[ReplayCase] = []
    for item in payload.get("cases", []):
        case = ReplayCase(**item)
        if case.family not in VALID_FAMILIES:
            raise ValueError(f"Unsupported replay family: {case.family}")
        if case.assertion_profile not in VALID_ASSERTION_PROFILES:
            raise ValueError(f"Unsupported assertion profile: {case.assertion_profile}")
        if family_filter and case.family not in family_filter:
            continue
        cases.append(case)
    return cases


def load_replay_fixtures(*families: str) -> list[tuple[ReplayCase, ReplayFixture]]:
    return [(case, ReplayFixture(case)) for case in load_replay_cases(families=families or None)]


def summarize_responses_replay(
    *,
    final_response: Response,
    stream_events: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
) -> ReplaySummary:
    tool_calls: list[ToolCallSummary] = []
    text_chunks: list[str] = []
    saw_function_tool = False

    for item in getattr(final_response, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type in {"function_call", "custom_tool_call"}:
            saw_function_tool = True
            tool_calls.append(
                ToolCallSummary(
                    name=str(getattr(item, "name", None) or "tool"),
                    arguments=_json_argument_payload(getattr(item, "arguments", None) or {}),
                )
            )
            continue
        if item_type != "message":
            continue
        for part in getattr(item, "content", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                text_chunks.append(str(part_text))

    stop_reason = "tool_use" if saw_function_tool else "end_turn"
    return ReplaySummary(
        stop_reason=stop_reason,
        assistant_text="".join(text_chunks),
        tool_calls=tool_calls,
        stream_events=stream_events,
        tool_events=tool_events,
    )


def summarize_anthropic_replay(
    *,
    message: Any,
    stream_events: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
) -> ReplaySummary:
    text_chunks: list[str] = []
    tool_calls: list[ToolCallSummary] = []
    for block in getattr(message, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            block_text = getattr(block, "text", None)
            if block_text:
                text_chunks.append(str(block_text))
            continue
        if block_type != "tool_use":
            continue
        tool_calls.append(
            ToolCallSummary(
                name=str(getattr(block, "name", None) or "tool"),
                arguments=getattr(block, "input", None) or {},
            )
        )

    return ReplaySummary(
        stop_reason=_normalize_stop_reason(getattr(message, "stop_reason", None)),
        assistant_text="".join(text_chunks),
        tool_calls=tool_calls,
        stream_events=stream_events,
        tool_events=tool_events,
    )


def summarize_google_replay(
    *,
    final_response: google_types.GenerateContentResponse | None,
    stream_events: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
) -> ReplaySummary:
    if final_response is None:
        return ReplaySummary(
            stop_reason=None,
            assistant_text="",
            tool_calls=[],
            stream_events=stream_events,
            tool_events=tool_events,
        )

    text_chunks: list[str] = []
    tool_calls: list[ToolCallSummary] = []

    candidates = getattr(final_response, "candidates", None) or []
    if candidates:
        parts = getattr(candidates[0].content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                text_chunks.append(str(text))
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                tool_calls.append(
                    ToolCallSummary(
                        name=str(getattr(function_call, "name", None) or "tool"),
                        arguments=getattr(function_call, "args", None) or {},
                    )
                )

    stop_reason = "tool_use" if tool_calls else "end_turn"
    return ReplaySummary(
        stop_reason=stop_reason,
        assistant_text="".join(text_chunks),
        tool_calls=tool_calls,
        stream_events=stream_events,
        tool_events=tool_events,
    )


def assert_replay_case(case: ReplayCase, fixture: ReplayFixture, summary: ReplaySummary) -> None:
    expected = fixture.result_message()
    expected_text = expected.last_text() or ""
    expected_tool_calls = [
        ToolCallSummary(
            name=str(call.params.name),
            arguments=call.params.arguments,
        )
        for call in (expected.tool_calls or {}).values()
    ]

    assert summary.stream_events == fixture.listener_stream_events()
    assert summary.tool_events == fixture.listener_tool_events()

    expected_stop_reason = _normalize_stop_reason(expected.stop_reason)
    if expected_stop_reason is not None:
        assert summary.stop_reason == expected_stop_reason

    if case.assertion_profile == "plain_text":
        assert summary.assistant_text == expected_text
        assert summary.tool_calls == []
        return

    if case.assertion_profile == "tool_use":
        assert summary.assistant_text == expected_text
        assert summary.tool_calls == expected_tool_calls
        return

    if case.assertion_profile == "web_search":
        assert summary.assistant_text == expected_text
        assert any(
            (event.get("payload") or {}).get("tool_name") == "web_search"
            for event in summary.tool_events
        )
        return

    raise AssertionError(f"Unhandled assertion profile: {case.assertion_profile}")


def _final_responses_payload(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    for payload in reversed(payloads):
        if payload.get("type") in {"response.completed", "response.incomplete", "response.done"}:
            response_payload = payload.get("response")
            if isinstance(response_payload, dict):
                return response_payload
    raise ValueError("Replay stream is missing a final Responses payload")


def _final_anthropic_message(payloads: list[dict[str, Any]]) -> Any:
    for payload in reversed(payloads):
        if payload.get("type") == "message_stop":
            return _ANTHROPIC_EVENT_ADAPTER.validate_python(payload).message

    message_payload: dict[str, Any] | None = None
    content_by_index: dict[int, Any] = {}
    for payload in payloads:
        event_type = payload.get("type")
        if event_type == "message_start":
            raw_message = payload.get("message")
            if isinstance(raw_message, dict):
                message_payload = raw_message
        elif event_type == "content_block_stop":
            index = payload.get("index")
            if isinstance(index, int):
                content_by_index[index] = payload.get("content_block")
        elif event_type == "message_delta" and message_payload is not None:
            delta = payload.get("delta") or {}
            if isinstance(delta, dict):
                if "stop_reason" in delta:
                    message_payload["stop_reason"] = delta.get("stop_reason")
                if "stop_sequence" in delta:
                    message_payload["stop_sequence"] = delta.get("stop_sequence")
            usage = payload.get("usage")
            if usage is not None:
                message_payload["usage"] = usage

    if message_payload is None:
        raise ValueError("Replay stream is missing an Anthropic message_start payload")

    message_payload = dict(message_payload)
    content = [content_by_index[index] for index in sorted(content_by_index)]
    message_payload["content"] = content
    if message_payload.get("stop_reason") is None:
        message_payload["stop_reason"] = (
            "tool_use"
            if any((block or {}).get("type") == "tool_use" for block in content)
            else "end_turn"
        )
    message_stop_payload = {
        "type": "message_stop",
        "message": message_payload,
    }
    return _ANTHROPIC_EVENT_ADAPTER.validate_python(message_stop_payload).message
