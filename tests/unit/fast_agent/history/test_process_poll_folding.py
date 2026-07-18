import json

import pytest
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    TextContent,
)

from fast_agent.agents.tool_result_channels import build_tool_result_message
from fast_agent.constants import (
    FAST_AGENT_PROCESS_POLL_FOLD,
    FAST_AGENT_SHELL_PROCESS_METADATA,
    FAST_AGENT_TIMING,
    FAST_AGENT_TOOL_TIMING,
    FAST_AGENT_USAGE,
)
from fast_agent.history.process_poll_folding import (
    fold_managed_process_poll_history as fold_completed_process_poll_history,
)
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.session.trace_export_atif import AtifRunSource, build_atif_trajectory
from fast_agent.types.llm_stop_reason import LlmStopReason


def _poll_request(
    index: int,
    *,
    process_id: str = "process-1",
    narration: str | None = None,
) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        content=(
            [TextContent(type="text", text=narration)]
            if narration is not None
            else []
        ),
        tool_calls={
            f"call-{index}": CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name="poll_process",
                    arguments={
                        "process_id": process_id,
                        "wait_sec": 30,
                        "wake_on_output": False,
                    },
                ),
            )
        },
        stop_reason=LlmStopReason.TOOL_USE,
    )


def _poll_result(
    index: int,
    *,
    status: str = "running",
    process_id: str = "process-1",
    output_line_count: int | None = None,
) -> PromptMessageExtended:
    result = CallToolResult(
        content=[TextContent(type="text", text=f"poll output {index}")],
        isError=status == "failed",
    )
    line_count = index if output_line_count is None else output_line_count
    metadata: dict[str, object] = {
        "process_id": process_id,
        "process_status": status,
        "process_yield_reason": "completion" if status != "running" else "deadline",
        "process_elapsed_seconds": index * 30.0,
        "output_line_count": line_count,
        "total_output_bytes": line_count * 100,
    }
    if status in {"completed", "failed"}:
        metadata["exit_code"] = 0 if status == "completed" else 1
    result.meta = {FAST_AGENT_SHELL_PROCESS_METADATA: metadata}
    return PromptMessageExtended(
        role="user",
        tool_results={f"call-{index}": result},
    )


def _update_result_metadata(message: PromptMessageExtended, **updates: object) -> None:
    results = message.tool_results
    assert results is not None
    result = next(iter(results.values()))
    assert result.meta is not None
    result.meta[FAST_AGENT_SHELL_PROCESS_METADATA].update(updates)


def _history_before_terminal(polls: int) -> list[PromptMessageExtended]:
    history = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="build the project")],
        )
    ]
    for index in range(1, polls):
        history.extend(
            [_poll_request(index), _poll_result(index, output_line_count=0)]
        )
    history.append(_poll_request(polls))
    return history


def _summary_result(folded) -> CallToolResult:
    results = folded.tool_message.tool_results
    assert results is not None
    return next(iter(results.values()))


def _add_usage(
    message: PromptMessageExtended,
    index: int,
    *,
    cost_usd: float | None = None,
) -> None:
    attempt: dict[str, object] = {
        "provider": "responses",
        "usage_schema": "openai-responses",
        "model": "model",
        "prompt": {
            "total": 100 + index,
            "uncached": 20 + index,
            "cache_read": 80,
            "cache_write": 0,
            "tool_use": index,
        },
        "completion": {
            "total": 10 + index,
            "reasoning": index,
        },
        "tool_calls": 1,
    }
    if cost_usd is not None:
        attempt["cost_usd"] = cost_usd
    message.channels = {
        FAST_AGENT_USAGE: [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "provider_attempts": [attempt]
                    }
                ),
            )
        ],
        FAST_AGENT_TIMING: [
            TextContent(
                type="text",
                text=json.dumps({"duration_ms": 1_000 + index}),
            )
        ],
        "fast-agent-provider-diagnostics": [
            TextContent(
                type="text",
                text=json.dumps({"websocket_request_mode": "continuation"}),
            )
        ],
    }


def _add_sparse_usage(message: PromptMessageExtended, index: int) -> None:
    message.channels = {
        FAST_AGENT_USAGE: [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "provider_attempts": [
                            {
                                "provider": "openai",
                                "usage_schema": "openai-chat",
                                "model": "model",
                                "prompt": {
                                    "total": 100 + index,
                                    "cache_read": 80,
                                    "cache_write": 0,
                                },
                                "completion": {"total": 10 + index},
                                "tool_calls": 1,
                            }
                        ]
                    }
                ),
            )
        ],
        FAST_AGENT_TIMING: [
            TextContent(
                type="text",
                text=json.dumps({"duration_ms": 1_000 + index}),
            )
        ],
    }


def _add_tool_timing(message: PromptMessageExtended, index: int) -> None:
    channels = dict(message.channels or {})
    channels[FAST_AGENT_TOOL_TIMING] = [
        TextContent(
            type="text",
            text=json.dumps({f"call-{index}": {"timing_ms": 30_000 + index}}),
        )
    ]
    message.channels = channels


def _move_result_metadata_to_channel(message: PromptMessageExtended) -> None:
    results = message.tool_results
    assert results is not None
    call_id, result = next(iter(results.items()))
    assert result.meta is not None
    metadata = result.meta.pop(FAST_AGENT_SHELL_PROCESS_METADATA)
    result.meta = result.meta or None
    message.channels = {
        FAST_AGENT_SHELL_PROCESS_METADATA: [
            TextContent(type="text", text=json.dumps({call_id: metadata}))
        ]
    }


def test_folds_successful_poll_suffix_and_retains_terminal_pair() -> None:
    history = _history_before_terminal(5)

    folded = fold_completed_process_poll_history(
        history,
        _poll_result(5, status="completed"),
    )

    assert folded is not None
    assert len(folded.history) == 2
    assert folded.history[0].role == "user"
    assert folded.history[1].tool_calls is not None
    audit = folded.metadata["audit"]
    assert isinstance(audit, dict)
    audit_mapping = {str(key): value for key, value in audit.items()}
    removed_exchanges = audit_mapping["removed_exchanges"]
    assert isinstance(removed_exchanges, list)
    removed_call_ids: list[object] = []
    for exchange in removed_exchanges:
        assert isinstance(exchange, dict)
        exchange_mapping = {
            str(key): value for key, value in exchange.items()
        }
        removed_call_ids.append(exchange_mapping["call_id"])
    assert removed_call_ids == [
        "call-1",
        "call-2",
        "call-3",
        "call-4",
    ]
    retained_exchanges = audit_mapping["retained_exchanges"]
    assert isinstance(retained_exchanges, list)
    assert len(retained_exchanges) == 1
    retained_exchange = retained_exchanges[0]
    assert isinstance(retained_exchange, dict)
    retained_exchange_mapping = {
        str(key): value for key, value in retained_exchange.items()
    }
    assert retained_exchange_mapping["call_id"] == "call-5"
    metadata_without_audit = {
        key: value
        for key, value in folded.metadata.items()
        if key not in {"audit", "folded_usage"}
    }
    assert metadata_without_audit == {
        "process_id": "process-1",
        "polls": 5,
        "polls_folded": 4,
        "polls_retained": 1,
        "messages_removed": 8,
        "requested_waits": {30: 5},
        "wake_reasons": {"completion": 1, "deadline": 4},
        "output_lines": 5,
        "terminal_status": "completed",
        "elapsed_seconds": 150.0,
        "output_bytes": 500,
        "exit_code": 0,
    }
    # No usage channels were recorded, so the summary keeps what it does know
    # (poll counts and per-turn wait/yield data) and reports token fields as None.
    folded_usage = folded.metadata["folded_usage"]
    assert isinstance(folded_usage, dict)
    assert folded_usage["llm_calls"] == 4
    assert folded_usage["provider_attempts"] is None
    assert folded_usage["prompt_tokens"] is None
    turns = folded_usage["turns"]
    assert isinstance(turns, list)
    assert len(turns) == 4
    assert all(turn["wait_sec"] == 30 for turn in turns)
    assert all(turn["prompt_tokens"] is None for turn in turns)
    summary_result = _summary_result(folded)
    first = summary_result.content[0]
    assert isinstance(first, TextContent)
    assert "5 polls" in first.text
    assert "30s × 5" in first.text
    assert "poll output 4" not in first.text
    assert isinstance(summary_result.content[1], TextContent)
    assert summary_result.content[1].text == "poll output 5"


def test_folding_uses_effective_wait_when_poll_request_omits_wait() -> None:
    history = _history_before_terminal(3)
    for message in history:
        for request in (message.tool_calls or {}).values():
            assert request.params.arguments is not None
            request.params.arguments.pop("wait_sec")
        if message.tool_results:
            _update_result_metadata(message, poll_wait_sec=45)

    terminal = _poll_result(3, status="completed")
    _update_result_metadata(terminal, poll_wait_sec=45)

    folded = fold_completed_process_poll_history(history, terminal)

    assert folded is not None
    assert folded.metadata["requested_waits"] == {45: 3}
    folded_usage = folded.metadata["folded_usage"]
    assert isinstance(folded_usage, dict)
    folded_usage_mapping = {
        str(key): value for key, value in folded_usage.items()
    }
    turns = folded_usage_mapping["turns"]
    assert isinstance(turns, list)
    for turn in turns:
        assert isinstance(turn, dict)
        turn_mapping = {str(key): value for key, value in turn.items()}
        assert turn_mapping["wait_sec"] == 45


def test_failed_process_retains_last_two_poll_pairs() -> None:
    history = _history_before_terminal(6)

    folded = fold_completed_process_poll_history(
        history,
        _poll_result(6, status="failed"),
    )

    assert folded is not None
    assert len(folded.history) == 4
    assert folded.metadata["polls_folded"] == 4
    assert folded.metadata["polls_retained"] == 2
    retained_result = folded.history[2].tool_results
    assert retained_result is not None
    retained_content = next(iter(retained_result.values())).content[0]
    assert isinstance(retained_content, TextContent)
    assert retained_content.text == "poll output 5"


def test_failed_process_audit_restores_original_poll_order() -> None:
    history = _history_before_terminal(6)
    for index in range(1, 7):
        _add_usage(history[index * 2 - 1], index)
        if index < 6:
            _add_tool_timing(history[index * 2], index)
    failed = _poll_result(6, status="failed")
    _add_tool_timing(failed, 6)
    folded = fold_completed_process_poll_history(history, failed)
    assert folded is not None

    final = PromptMessageExtended(role="assistant")
    _add_usage(final, 7)
    exported_history = [*folded.history, folded.tool_message, final]
    trajectory = build_atif_trajectory(
        AtifRunSource(
            session_id="session",
            agent_name="agent",
            model_name="model",
            provider="provider",
            history=exported_history,
            message_timestamps=(None,) * len(exported_history),
        )
    )

    poll_steps = [
        step
        for step in trajectory.steps
        if step.tool_calls
        and step.tool_calls[0].function_name == "poll_process"
    ]
    assert [
        step.tool_calls[0].tool_call_id
        for step in poll_steps
        if step.tool_calls
    ] == [f"call-{index}" for index in range(1, 7)]
    boundary = trajectory.steps[7]
    assert boundary.extra is not None
    context_management = boundary.extra["context_management"]
    assert isinstance(context_management, dict)
    assert context_management["removed_step_ids"] == [2, 3, 4, 5]
    assert context_management["retained_step_ids"] == [6, 7]


def test_running_process_with_output_is_not_folded() -> None:
    history = _history_before_terminal(5)
    _update_result_metadata(history[2], output_line_count=1)

    assert (
        fold_completed_process_poll_history(
            history,
            _poll_result(5, output_line_count=0),
        )
        is None
    )


def test_terminal_process_with_output_in_earlier_poll_is_not_folded() -> None:
    history = _history_before_terminal(5)
    result = history[4].tool_results
    assert result is not None
    output_result = next(iter(result.values()))
    _update_result_metadata(history[4], output_line_count=3)
    output_result.content = [TextContent(type="text", text="compiler error details")]

    assert (
        fold_completed_process_poll_history(
            history,
            _poll_result(5, status="failed", output_line_count=0),
        )
        is None
    )


def test_quiet_running_process_folds_earlier_polls() -> None:
    history = _history_before_terminal(3)

    folded = fold_completed_process_poll_history(
        history,
        _poll_result(3, output_line_count=0),
    )

    assert folded is not None
    assert len(folded.history) == 2
    assert folded.metadata["terminal_status"] == "running"
    assert folded.metadata["polls"] == 3
    assert folded.metadata["polls_folded"] == 2
    assert folded.metadata["polls_retained"] == 1
    first = _summary_result(folded).content[0]
    assert isinstance(first, TextContent)
    assert "Current status: running." in first.text


def test_narrated_poll_suffix_folds_and_preserves_removed_updates_exactly() -> None:
    history = _history_before_terminal(4)
    narrations = [
        "Still compiling.",
        "Checking phase two.\nNo errors yet.",
        "Unicode remains exact: λ → ✓",
        "This retained update must not be archived.",
    ]
    for index, narration in enumerate(narrations, start=1):
        history[index * 2 - 1].content = [
            TextContent(type="text", text=narration)
        ]

    folded = fold_completed_process_poll_history(
        history,
        _poll_result(4, status="completed"),
    )

    assert folded is not None
    assert folded.metadata["assistant_updates"] == [
        {
            "call_id": f"call-{index}",
            "content": [{"type": "text", "text": narration}],
        }
        for index, narration in enumerate(narrations[:3], start=1)
    ]
    summary = _summary_result(folded).content[0]
    assert isinstance(summary, TextContent)
    for narration in narrations[:3]:
        assert narration in summary.text
    assert narrations[3] not in summary.text

    audit = folded.metadata["audit"]
    assert isinstance(audit, dict)
    audit_mapping = {str(key): value for key, value in audit.items()}
    removed_exchanges = audit_mapping["removed_exchanges"]
    assert isinstance(removed_exchanges, list)
    removed_narrations: list[str] = []
    for exchange in removed_exchanges:
        assert isinstance(exchange, dict)
        exchange_mapping = {
            str(key): value for key, value in exchange.items()
        }
        request = exchange_mapping["request"]
        assert isinstance(request, dict)
        request_mapping = {
            str(key): value for key, value in request.items()
        }
        content = request_mapping["content"]
        assert isinstance(content, list)
        block = content[0]
        assert isinstance(block, dict)
        block_mapping = {str(key): value for key, value in block.items()}
        text = block_mapping["text"]
        assert isinstance(text, str)
        removed_narrations.append(text)
    assert removed_narrations == narrations[:3]


def test_narration_does_not_weaken_zero_output_fold_guard() -> None:
    history = _history_before_terminal(4)
    history[1].content = [TextContent(type="text", text="Waiting.")]
    _update_result_metadata(history[2], output_line_count=1)

    assert (
        fold_completed_process_poll_history(
            history,
            _poll_result(4, status="completed", output_line_count=0),
        )
        is None
    )


def test_parallel_narrated_poll_call_stops_suffix_collection() -> None:
    history = _history_before_terminal(4)
    request = history[-3]
    request.content = [TextContent(type="text", text="Checking both tasks.")]
    assert request.tool_calls is not None
    request.tool_calls["other-call"] = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="other_tool", arguments={}),
    )

    assert (
        fold_completed_process_poll_history(
            history,
            _poll_result(4, status="completed"),
        )
        is None
    )


def test_different_process_id_stops_suffix_collection() -> None:
    history = _history_before_terminal(3)
    history[-3:-1] = [
        _poll_request(2, process_id="process-2"),
        _poll_result(2, process_id="process-2"),
    ]

    assert (
        fold_completed_process_poll_history(
            history,
            _poll_result(3, status="completed"),
        )
        is None
    )


def test_folded_terminal_result_does_not_fold_again() -> None:
    folded = fold_completed_process_poll_history(
        _history_before_terminal(5),
        _poll_result(5, status="completed"),
    )
    assert folded is not None

    assert (
        fold_completed_process_poll_history(
            folded.history,
            folded.tool_message,
        )
        is None
    )


def test_reads_durable_process_metadata_channel() -> None:
    history = _history_before_terminal(4)
    for message in history:
        if message.tool_results:
            _move_result_metadata_to_channel(message)
    terminal = _poll_result(4, status="completed")
    _move_result_metadata_to_channel(terminal)

    folded = fold_completed_process_poll_history(history, terminal)

    assert folded is not None
    assert folded.metadata["polls"] == 4
    assert folded.metadata["terminal_status"] == "completed"


def test_durable_process_metadata_is_authoritative_over_channel() -> None:
    history = _history_before_terminal(4)
    terminal = _poll_result(4, status="completed")
    for message in [*history, terminal]:
        results = message.tool_results or {}
        for call_id in results:
            message.channels = {
                FAST_AGENT_SHELL_PROCESS_METADATA: [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                call_id: {
                                    "process_id": "stale-process-id",
                                    "process_status": "running",
                                }
                            }
                        ),
                    )
                ]
            }

    folded = fold_completed_process_poll_history(history, terminal)

    assert folded is not None
    assert folded.metadata["process_id"] == "process-1"
    assert folded.metadata["terminal_status"] == "completed"


def test_tool_result_message_preserves_shell_process_metadata_in_channel() -> None:
    result = _poll_result(1).tool_results
    assert result is not None
    call_id, tool_result = next(iter(result.items()))
    tool_result.meta = {
        FAST_AGENT_SHELL_PROCESS_METADATA: {
            "process_id": "process-1",
            "process_status": "running",
        }
    }

    message = build_tool_result_message({call_id: tool_result})

    blocks = (message.channels or {})[FAST_AGENT_SHELL_PROCESS_METADATA]
    assert isinstance(blocks[0], TextContent)
    assert json.loads(blocks[0].text) == {
        call_id: {
            "process_id": "process-1",
            "process_status": "running",
        }
    }


def test_fold_archives_exact_usage_and_atif_restores_totals() -> None:
    history = _history_before_terminal(4)
    for index in range(1, 5):
        _add_usage(history[index * 2 - 1], index, cost_usd=index / 100)
        if index < 4:
            result_message = history[index * 2]
            _update_result_metadata(
                result_message,
                output_bytes_since_last_poll=0,
                seconds_since_last_output=index * 30.0,
                has_observed_output=False,
            )
            _move_result_metadata_to_channel(result_message)
            _add_tool_timing(result_message, index)
    terminal = _poll_result(4, status="completed")
    _update_result_metadata(
        terminal,
        output_bytes_since_last_poll=0,
        seconds_since_last_output=120.0,
        has_observed_output=False,
    )
    _move_result_metadata_to_channel(terminal)
    _add_tool_timing(terminal, 4)

    folded = fold_completed_process_poll_history(history, terminal)

    assert folded is not None
    archived = folded.metadata["folded_usage"]
    assert isinstance(archived, dict)
    archived_mapping = {str(key): value for key, value in archived.items()}
    turns = archived_mapping.pop("turns")
    assert archived_mapping == {
        "llm_calls": 3,
        "provider_attempts": 3,
        "prompt_tokens": 306,
        "uncached_tokens": 66,
        "cached_tokens": 240,
        "cache_write_tokens": 0,
        "tool_use_prompt_tokens": 6,
        "completion_tokens": 36,
        "reasoning_tokens": 6,
        "model_duration_ms": 3006.0,
        "tool_duration_ms": 90006.0,
        "request_modes": {"continuation": 3},
        "cost_usd": 0.06,
    }
    assert isinstance(turns, list)
    assert len(turns) == 3
    first_turn = turns[0]
    assert isinstance(first_turn, dict)
    first_turn_mapping = {str(key): value for key, value in first_turn.items()}
    assert first_turn_mapping["prompt_tokens"] == 101
    assert first_turn_mapping["request_mode"] == "continuation"
    assert first_turn_mapping["output_bytes_since_last_poll"] == 0
    assert first_turn_mapping["seconds_since_last_output"] == 30.0
    assert first_turn_mapping["has_observed_output"] is False
    fold_blocks = (folded.tool_message.channels or {})[
        FAST_AGENT_PROCESS_POLL_FOLD
    ]
    assert isinstance(fold_blocks[0], TextContent)

    final = PromptMessageExtended(role="assistant")
    _add_usage(final, 5, cost_usd=0.05)
    exported_history = [*folded.history, folded.tool_message, final]
    trajectory = build_atif_trajectory(
        AtifRunSource(
            session_id="session",
            agent_name="agent",
            model_name="model",
            provider="provider",
            history=exported_history,
            message_timestamps=(None,) * len(exported_history),
        )
    )

    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 515
    assert trajectory.final_metrics.total_cached_tokens == 400
    assert trajectory.final_metrics.total_completion_tokens == 65
    assert trajectory.final_metrics.total_cost_usd == pytest.approx(0.15)
    assert trajectory.final_metrics.total_steps == 7
    assert trajectory.final_metrics.extra == {
        "total_reasoning_tokens": 15,
        "total_tool_use_tokens": 15,
        "folded_process_poll_steps": 3,
        "process_poll_context_rewrites": 1,
    }
    assert len(trajectory.steps) == 7
    step_metrics = [
        step.metrics for step in trajectory.steps if step.metrics is not None
    ]
    assert sum(metric.prompt_tokens or 0 for metric in step_metrics) == 515
    assert sum(metric.completion_tokens or 0 for metric in step_metrics) == 65
    assert sum(metric.cached_tokens or 0 for metric in step_metrics) == 400
    assert sum(metric.cost_usd or 0 for metric in step_metrics) == pytest.approx(0.15)
    poll_steps = trajectory.steps[1:5]
    assert [step.llm_call_count for step in poll_steps] == [1, 1, 1, 1]
    assert [
        step.metrics.prompt_tokens if step.metrics is not None else None
        for step in poll_steps
    ] == [101, 102, 103, 104]
    assert [
        step.tool_calls[0].function_name if step.tool_calls else None
        for step in poll_steps
    ] == ["poll_process"] * 4
    assert [
        step.observation.results[0].content
        if step.observation is not None
        else None
        for step in poll_steps
    ] == ["poll output 1", "poll output 2", "poll output 3", "poll output 4"]
    assert [
        step.observation.results[0].extra["process_metadata"][
            "process_yield_reason"
        ]
        if step.observation is not None
        and step.observation.results[0].extra is not None
        else None
        for step in poll_steps
    ] == ["deadline", "deadline", "deadline", "completion"]
    boundary = trajectory.steps[5]
    assert boundary.source == "system"
    assert boundary.extra is not None
    assert boundary.extra["context_management"] == {
        "type": "compaction",
        "boundary": "truncate",
        "scope": "step_ids",
        "removed_step_ids": [2, 3, 4],
        "replacement_source": "observation",
        "replacement_position": "prepend_to_retained_observation",
        "strategy": "managed_process_poll_fold",
        "retained_step_ids": [5],
    }
    assert boundary.observation is not None
    assert "3 earlier polls folded" in str(boundary.observation.results[0].content)
    assert "No output observed for 120.0 seconds" in str(
        boundary.observation.results[0].content
    )
    assert trajectory.notes is not None
    assert "preserve every original LLM/tool step" in trajectory.notes


def test_fold_archives_usage_when_optional_token_fields_are_absent() -> None:
    history = _history_before_terminal(4)
    for index in range(1, 5):
        _add_sparse_usage(history[index * 2 - 1], index)
        if index < 4:
            _add_tool_timing(history[index * 2], index)
    terminal = _poll_result(4, status="completed")
    _add_tool_timing(terminal, 4)

    folded = fold_completed_process_poll_history(history, terminal)

    assert folded is not None
    archived = folded.metadata["folded_usage"]
    assert isinstance(archived, dict)
    archived_mapping = {str(key): value for key, value in archived.items()}
    assert archived_mapping["prompt_tokens"] == 306
    assert archived_mapping["completion_tokens"] == 36
    assert archived_mapping["cached_tokens"] == 240
    assert archived_mapping["uncached_tokens"] is None
    assert archived_mapping["reasoning_tokens"] is None

    final = PromptMessageExtended(role="assistant")
    _add_sparse_usage(final, 5)
    exported_history = [*folded.history, folded.tool_message, final]
    trajectory = build_atif_trajectory(
        AtifRunSource(
            session_id="session",
            agent_name="agent",
            model_name="model",
            provider="provider",
            history=exported_history,
            message_timestamps=(None,) * len(exported_history),
        )
    )

    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 515
    assert trajectory.final_metrics.total_completion_tokens == 65
    assert trajectory.final_metrics.total_cached_tokens == 400


def test_fold_keeps_partial_usage_when_one_poll_is_missing_usage() -> None:
    history = _history_before_terminal(4)
    for index in range(1, 5):
        if index != 2:
            _add_usage(history[index * 2 - 1], index, cost_usd=index / 100)
        if index < 4:
            _add_tool_timing(history[index * 2], index)
    terminal = _poll_result(4, status="completed")
    _add_tool_timing(terminal, 4)

    folded = fold_completed_process_poll_history(history, terminal)

    assert folded is not None
    archived = folded.metadata["folded_usage"]
    assert isinstance(archived, dict)
    archived_mapping = {str(key): value for key, value in archived.items()}
    assert archived_mapping["llm_calls"] == 3
    # Aggregates stay complete-or-None: the poll without usage poisons totals,
    # but fields observed for every poll are preserved.
    assert archived_mapping["provider_attempts"] is None
    assert archived_mapping["prompt_tokens"] is None
    assert "cost_usd" not in archived_mapping
    assert archived_mapping["model_duration_ms"] is None
    assert archived_mapping["tool_duration_ms"] == pytest.approx(90006.0)
    turns = archived_mapping["turns"]
    assert isinstance(turns, list)
    assert [turn["prompt_tokens"] for turn in turns] == [101, None, 103]
    assert [turn["cost_usd"] for turn in turns] == [0.01, None, 0.03]
    assert [turn["provider_attempts"] for turn in turns] == [1, None, 1]
    assert [turn["yield_reason"] for turn in turns] == ["deadline"] * 3


def test_atif_rejects_fold_without_audit_archive() -> None:
    history = _history_before_terminal(4)
    for index in range(1, 5):
        _add_usage(history[index * 2 - 1], index)
        if index < 4:
            _add_tool_timing(history[index * 2], index)
    terminal = _poll_result(4, status="completed")
    _add_tool_timing(terminal, 4)
    folded = fold_completed_process_poll_history(history, terminal)
    assert folded is not None

    invalid_metadata = {
        key: value for key, value in folded.metadata.items() if key != "audit"
    }
    channels = dict(folded.tool_message.channels or {})
    channels[FAST_AGENT_PROCESS_POLL_FOLD] = [
        TextContent(type="text", text=json.dumps(invalid_metadata, sort_keys=True))
    ]
    folded.tool_message.channels = channels

    with pytest.raises(ValueError, match="missing its audit archive"):
        build_atif_trajectory(
            AtifRunSource(
                session_id="session",
                agent_name="agent",
                model_name="model",
                provider="provider",
                history=[*folded.history, folded.tool_message],
                message_timestamps=(None,) * (len(folded.history) + 1),
            )
        )


def test_atif_rejects_context_rewrite_with_unknown_call_id() -> None:
    folded = fold_completed_process_poll_history(
        _history_before_terminal(4),
        _poll_result(4, status="completed"),
    )
    assert folded is not None
    invalid_metadata = json.loads(json.dumps(folded.metadata))
    invalid_metadata["audit"]["context_rewrites"][0][
        "removed_call_ids"
    ].append("unknown-call")
    channels = dict(folded.tool_message.channels or {})
    channels[FAST_AGENT_PROCESS_POLL_FOLD] = [
        TextContent(type="text", text=json.dumps(invalid_metadata, sort_keys=True))
    ]
    folded.tool_message.channels = channels

    with pytest.raises(
        ValueError,
        match="audit archive is invalid",
    ):
        build_atif_trajectory(
            AtifRunSource(
                session_id="session",
                agent_name="agent",
                model_name="model",
                provider="provider",
                history=[*folded.history, folded.tool_message],
                message_timestamps=(None,) * (len(folded.history) + 1),
            )
        )


def test_repeated_fold_requires_prior_audit_archive() -> None:
    history = _history_before_terminal(3)
    for index in range(1, 4):
        _add_usage(history[index * 2 - 1], index)
        if index < 3:
            _add_tool_timing(history[index * 2], index)
    running_three = _poll_result(3, output_line_count=0)
    _add_tool_timing(running_three, 3)
    first_fold = fold_completed_process_poll_history(history, running_three)
    assert first_fold is not None

    invalid_metadata = {
        key: value for key, value in first_fold.metadata.items() if key != "audit"
    }
    channels = dict(first_fold.tool_message.channels or {})
    channels[FAST_AGENT_PROCESS_POLL_FOLD] = [
        TextContent(type="text", text=json.dumps(invalid_metadata, sort_keys=True))
    ]
    first_fold.tool_message.channels = channels

    request_four = _poll_request(4)
    _add_usage(request_four, 4)
    result_four = _poll_result(4, output_line_count=0)
    _add_tool_timing(result_four, 4)
    request_five = _poll_request(5)
    _add_usage(request_five, 5)
    terminal = _poll_result(5, status="completed")
    _add_tool_timing(terminal, 5)

    assert (
        fold_completed_process_poll_history(
            [
                *first_fold.history,
                first_fold.tool_message,
                request_four,
                result_four,
                request_five,
            ],
            terminal,
        )
        is None
    )


def test_repeated_folds_preserve_cumulative_narration_and_boundary_history() -> None:
    history = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="build the project")],
        )
    ]
    narrations = [f"Update {index}\nλ{index}" for index in range(1, 6)]
    for index in range(1, 3):
        history.extend(
            [
                _poll_request(index, narration=narrations[index - 1]),
                _poll_result(index, output_line_count=0),
            ]
        )
    history.append(_poll_request(3, narration=narrations[2]))

    first_fold = fold_completed_process_poll_history(
        history,
        _poll_result(3, output_line_count=0),
    )

    assert first_fold is not None
    request_four = _poll_request(4, narration=narrations[3])
    result_four = _poll_result(4, output_line_count=0)
    request_five = _poll_request(5, narration=narrations[4])
    second_fold = fold_completed_process_poll_history(
        [
            *first_fold.history,
            first_fold.tool_message,
            request_four,
            result_four,
            request_five,
        ],
        _poll_result(5, output_line_count=0),
    )

    assert second_fold is not None
    updates = second_fold.metadata["assistant_updates"]
    assert isinstance(updates, list)
    update_mappings: list[dict[str, object]] = []
    for update in updates:
        assert isinstance(update, dict)
        update_mappings.append(
            {str(key): value for key, value in update.items()}
        )
    assert [update["call_id"] for update in update_mappings] == [
        "call-1",
        "call-2",
        "call-3",
        "call-4",
    ]
    archived_narrations: list[str] = []
    for update in update_mappings:
        content = update["content"]
        assert isinstance(content, list)
        block = content[0]
        assert isinstance(block, dict)
        block_mapping = {str(key): value for key, value in block.items()}
        text = block_mapping["text"]
        assert isinstance(text, str)
        archived_narrations.append(text)
    assert archived_narrations == narrations[:4]

    audit = second_fold.metadata["audit"]
    assert isinstance(audit, dict)
    audit_mapping = {str(key): value for key, value in audit.items()}
    boundaries = audit_mapping["context_rewrites"]
    assert isinstance(boundaries, list)
    assert len(boundaries) == 2
    first_boundary = boundaries[0]
    second_boundary = boundaries[1]
    assert isinstance(first_boundary, dict)
    assert isinstance(second_boundary, dict)
    first_boundary_mapping = {
        str(key): value for key, value in first_boundary.items()
    }
    second_boundary_mapping = {
        str(key): value for key, value in second_boundary.items()
    }
    first_summary = first_boundary_mapping["summary"]
    second_summary = second_boundary_mapping["summary"]
    assert isinstance(first_summary, str)
    assert isinstance(second_summary, str)
    assert narrations[0] in first_summary
    assert narrations[1] in first_summary
    assert narrations[2] not in first_summary
    for narration in narrations[:4]:
        assert narration in second_summary
    assert narrations[4] not in second_summary


def test_repeated_running_folds_preserve_exact_cumulative_usage() -> None:
    history = _history_before_terminal(3)
    for index in range(1, 4):
        _add_usage(history[index * 2 - 1], index)
        if index < 3:
            _add_tool_timing(history[index * 2], index)
    running_three = _poll_result(3, output_line_count=0)
    _add_tool_timing(running_three, 3)

    first_fold = fold_completed_process_poll_history(history, running_three)

    assert first_fold is not None
    request_four = _poll_request(4)
    _add_usage(request_four, 4)
    result_four = _poll_result(4, output_line_count=0)
    _add_tool_timing(result_four, 4)
    request_five = _poll_request(5)
    _add_usage(request_five, 5)
    second_history = [
        *first_fold.history,
        first_fold.tool_message,
        request_four,
        result_four,
        request_five,
    ]
    running_five = _poll_result(5, output_line_count=0)
    _add_tool_timing(running_five, 5)

    second_fold = fold_completed_process_poll_history(
        second_history,
        running_five,
    )

    assert second_fold is not None
    assert second_fold.metadata["polls"] == 5
    assert second_fold.metadata["polls_folded"] == 4
    archived = second_fold.metadata["folded_usage"]
    assert isinstance(archived, dict)
    archived_mapping = {str(key): value for key, value in archived.items()}
    assert archived_mapping["llm_calls"] == 4
    assert archived_mapping["prompt_tokens"] == 410

    request_six = _poll_request(6)
    _add_usage(request_six, 6)
    terminal_history = [
        *second_fold.history,
        second_fold.tool_message,
        request_six,
    ]
    terminal = _poll_result(6, status="completed", output_line_count=1)
    _add_tool_timing(terminal, 6)

    final_fold = fold_completed_process_poll_history(terminal_history, terminal)

    assert final_fold is not None
    assert final_fold.metadata["polls"] == 6
    assert final_fold.metadata["polls_folded"] == 5
    final_archived = final_fold.metadata["folded_usage"]
    assert isinstance(final_archived, dict)
    final_archived_mapping = {
        str(key): value for key, value in final_archived.items()
    }
    assert final_archived_mapping["llm_calls"] == 5
    assert final_archived_mapping["prompt_tokens"] == 515

    final = PromptMessageExtended(role="assistant")
    _add_usage(final, 7)
    exported_history = [*final_fold.history, final_fold.tool_message, final]
    trajectory = build_atif_trajectory(
        AtifRunSource(
            session_id="session",
            agent_name="agent",
            model_name="model",
            provider="provider",
            history=exported_history,
            message_timestamps=(None,) * len(exported_history),
        )
    )

    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 728
    assert trajectory.final_metrics.total_cached_tokens == 560
    assert trajectory.final_metrics.total_completion_tokens == 98
    assert trajectory.final_metrics.total_steps == 11
    assert trajectory.final_metrics.extra == {
        "total_reasoning_tokens": 28,
        "total_tool_use_tokens": 28,
        "folded_process_poll_steps": 5,
        "process_poll_context_rewrites": 3,
    }
    assert len(trajectory.steps) == 11
    boundaries = [trajectory.steps[index] for index in (4, 7, 9)]
    context_management = [
        boundary.extra["context_management"]
        for boundary in boundaries
        if boundary.extra is not None
    ]
    assert [item["removed_step_ids"] for item in context_management] == [
        [2, 3],
        [4, 6],
        [7],
    ]
    assert [item["retained_step_ids"] for item in context_management] == [
        [4],
        [7],
        [9],
    ]
    assert all(
        item["strategy"] == "managed_process_poll_fold"
        for item in context_management
    )


def test_runner_style_repeated_folds_keep_audit_linear_and_lossless() -> None:
    polls = 12
    history: list[PromptMessageExtended] = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="build the project")],
        )
    ]
    final_fold = None
    fold_count = 0
    for index in range(1, polls + 1):
        request = _poll_request(index)
        status = "completed" if index == polls else "running"
        result = _poll_result(index, status=status, output_line_count=0)
        folded = fold_completed_process_poll_history([*history, request], result)
        if folded is None:
            history = [*history, request, result]
            continue
        fold_count += 1
        final_fold = folded
        history = [*folded.history, folded.tool_message]

    assert final_fold is not None
    assert final_fold.metadata["polls"] == polls
    assert final_fold.metadata["polls_folded"] == polls - 1
    # Running refolds wait for another full fold's worth of quiet polls, so the
    # fold machinery does not run on every poll.
    assert fold_count < polls - 1

    audit = final_fold.metadata["audit"]
    assert isinstance(audit, dict)
    rewrites = audit["context_rewrites"]
    assert isinstance(rewrites, list)
    assert len(rewrites) == fold_count
    per_rewrite_turns: list[int] = []
    removed_total = 0
    for rewrite in rewrites:
        assert isinstance(rewrite, dict)
        removed_ids = rewrite["removed_call_ids"]
        assert isinstance(removed_ids, list)
        removed_total += len(removed_ids)
        fold_record = rewrite["fold"]
        assert isinstance(fold_record, dict)
        folded_usage = fold_record["folded_usage"]
        assert isinstance(folded_usage, dict)
        turns = folded_usage["turns"]
        assert isinstance(turns, list)
        per_rewrite_turns.append(len(turns))
    # Every folded poll is archived in exactly one rewrite delta, and each
    # rewrite stays bounded, so the audit grows linearly with poll count.
    assert sum(per_rewrite_turns) == polls - 1
    assert removed_total == polls - 1
    assert max(per_rewrite_turns) <= 2

    exported_history = [*final_fold.history, final_fold.tool_message]
    trajectory = build_atif_trajectory(
        AtifRunSource(
            session_id="session",
            agent_name="agent",
            model_name="model",
            provider="provider",
            history=exported_history,
            message_timestamps=(None,) * len(exported_history),
        )
    )
    poll_steps = [
        step
        for step in trajectory.steps
        if step.tool_calls and step.tool_calls[0].function_name == "poll_process"
    ]
    assert [
        step.tool_calls[0].tool_call_id for step in poll_steps if step.tool_calls
    ] == [f"call-{index}" for index in range(1, polls + 1)]
