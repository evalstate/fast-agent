from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.llm.provider.openai.streaming_utils import (
    fetch_and_finalize_stream_response,
    record_completed_output_item,
)

if TYPE_CHECKING:
    from fast_agent.core.logging.logger import Logger


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def info(self, message: str, data: dict[str, Any] | None = None) -> None:
        del message, data

    def warning(self, message: str, data: dict[str, Any] | None = None, **_kwargs: Any) -> None:
        self.warnings.append((message, data or {}))

    def error(self, message: str, data: dict[str, Any] | None = None, **_kwargs: Any) -> None:
        del message, data


class _FetchOnlyStream:
    """A stream that only surfaces its response via ``get_final_response()``."""

    def __init__(self, response: Any) -> None:
        self._response = response

    async def get_final_response(self) -> Any:
        return self._response


async def _finalize(
    *,
    stream: Any,
    final_response: Any | None,
    completed_output_items: list[tuple[int | None, int, Any]],
    logger: _RecordingLogger,
) -> Any:
    return await fetch_and_finalize_stream_response(
        stream=stream,
        final_response=final_response,
        fetch_failure_message="failed",
        use_exc_info_on_fetch_failure=False,
        incomplete_entries=(),
        model="gpt-test",
        agent_name="agent",
        chat_turn=lambda: 1,
        logger=cast("Logger", logger),
        notified_tool_indices=set(),
        emit_tool_fallback=lambda *_args, **_kwargs: None,
        completed_output_items=completed_output_items,
    )


def _done_event(*, output_index: int | None, sequence_number: int | None, name: str) -> Any:
    return SimpleNamespace(
        type="response.output_item.done",
        output_index=output_index,
        sequence_number=sequence_number,
        item=SimpleNamespace(type="message", name=name),
    )


@pytest.mark.asyncio
async def test_empty_output_is_reconstructed_on_the_fetched_response_path() -> None:
    """Reconstruction must also cover responses that arrive via ``get_final_response()``.

    When no terminal event is seen inline the response is fetched from the SDK, and
    that payload can be empty too.
    """
    logger = _RecordingLogger()
    fetched = SimpleNamespace(status="completed", output=[], usage=None)
    items: list[tuple[int | None, int, Any]] = []
    record_completed_output_item(
        items, _done_event(output_index=0, sequence_number=1, name="first"), fallback_sequence=0
    )

    result = await _finalize(
        stream=_FetchOnlyStream(fetched),
        final_response=None,
        completed_output_items=items,
        logger=logger,
    )

    assert [item.name for item in result.output] == ["first"]
    assert fetched.output == []
    assert [message for message, _data in logger.warnings] == [
        "Reconstructed empty terminal Responses output from completed stream items"
    ]


@pytest.mark.asyncio
async def test_populated_output_is_left_alone() -> None:
    logger = _RecordingLogger()
    existing = SimpleNamespace(type="message", name="provider")
    final_response = SimpleNamespace(status="completed", output=[existing], usage=None)
    items: list[tuple[int | None, int, Any]] = []
    record_completed_output_item(
        items, _done_event(output_index=0, sequence_number=1, name="streamed"), fallback_sequence=0
    )

    result = await _finalize(
        stream=_FetchOnlyStream(final_response),
        final_response=final_response,
        completed_output_items=items,
        logger=logger,
    )

    assert result is final_response
    assert result.output == [existing]
    assert logger.warnings == []


@pytest.mark.asyncio
async def test_reconstruction_orders_items_by_output_index_then_sequence() -> None:
    logger = _RecordingLogger()
    items: list[tuple[int | None, int, Any]] = []
    # Arrives out of order, and the last item has no output_index to sort on.
    record_completed_output_item(
        items, _done_event(output_index=2, sequence_number=9, name="third"), fallback_sequence=0
    )
    record_completed_output_item(
        items, _done_event(output_index=0, sequence_number=3, name="first"), fallback_sequence=1
    )
    record_completed_output_item(
        items, _done_event(output_index=1, sequence_number=5, name="second"), fallback_sequence=2
    )
    record_completed_output_item(
        items,
        _done_event(output_index=None, sequence_number=None, name="unindexed"),
        fallback_sequence=3,
    )

    result = await _finalize(
        stream=_FetchOnlyStream(None),
        final_response=SimpleNamespace(status="completed", output=[], usage=None),
        completed_output_items=items,
        logger=logger,
    )

    assert [item.name for item in result.output] == ["first", "second", "third", "unindexed"]


def test_items_without_a_payload_are_not_recorded() -> None:
    items: list[tuple[int | None, int, Any]] = []
    record_completed_output_item(
        items,
        SimpleNamespace(type="response.output_item.done", output_index=0, sequence_number=1),
        fallback_sequence=0,
    )

    assert items == []
