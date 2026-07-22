"""Deterministic folding for completed managed-process polling exchanges."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from mcp.types import CallToolResult, TextContent

from fast_agent.constants import (
    FAST_AGENT_PROCESS_POLL_FOLD,
    FAST_AGENT_SHELL_PROCESS_METADATA,
    FAST_AGENT_TIMING,
    FAST_AGENT_TOOL_TIMING,
    FAST_AGENT_USAGE,
)
from fast_agent.history.process_poll_fold_audit import (
    ArchivedContextRewrite,
    ArchivedPollExchange,
    ProcessPollFoldAudit,
)
from fast_agent.utils.tool_names import (
    BASH_TOOL_NAME,
    EXECUTE_TOOL_NAME,
    POLL_PROCESS_TOOL_NAME,
    PROCESS_TOOL_NAME,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

_FOLDABLE_PROCESS_STATUSES = frozenset(
    {"running", "completed", "failed", "terminated", "cancelled", "already_exited"}
)
_SINGLE_RETAINED_POLL_STATUSES = frozenset({"running", "completed"})
_MIN_FOLDED_POLLS = 2
_RESPONSES_DIAGNOSTICS_CHANNEL = "fast-agent-provider-diagnostics"


@dataclass(frozen=True, slots=True)
class PollExchange:
    request_index: int
    call_id: str
    process_id: str
    wait_sec: int
    wake_on_output: bool
    request: PromptMessageExtended
    result_message: PromptMessageExtended
    result: CallToolResult
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProcessPollFold:
    history: list[PromptMessageExtended]
    tool_message: PromptMessageExtended
    metadata: dict[str, object]


def _single_poll_call(
    message: PromptMessageExtended,
) -> tuple[str, str, int | None, bool] | None:
    calls = message.tool_calls or {}
    if message.role != "assistant" or len(calls) != 1:
        return None
    call_id, request = next(iter(calls.items()))
    arguments = request.params.arguments or {}
    if request.params.name == POLL_PROCESS_TOOL_NAME:
        wait_sec = arguments.get("wait_sec")
        wake_on_output = arguments.get("wake_on_output", False)
    elif request.params.name == PROCESS_TOOL_NAME:
        action = arguments.get("action", "status")
        if action not in {"status", "wait"}:
            return None
        wait_sec = 0 if action == "status" else arguments.get("wait_sec")
        wake_on_output = False
    else:
        return None
    process_id = arguments.get("process_id")
    if (
        not isinstance(process_id, str)
        or ("wait_sec" in arguments and type(wait_sec) is not int)
        or type(wake_on_output) is not bool
    ):
        return None
    return call_id, process_id, wait_sec, wake_on_output


def _single_poll_result(
    message: PromptMessageExtended,
    call_id: str,
) -> CallToolResult | None:
    results = message.tool_results or {}
    if message.role != "user" or len(results) != 1:
        return None
    return results.get(call_id)


def _managed_process_start_id(
    request: PromptMessageExtended,
    result: PromptMessageExtended,
) -> str | None:
    if request.role != "assistant" or result.role != "user":
        return None
    calls = request.tool_calls or {}
    results = result.tool_results or {}
    for call_id, call in calls.items():
        if call.params.name not in {EXECUTE_TOOL_NAME, BASH_TOOL_NAME}:
            continue
        tool_result = results.get(call_id)
        if tool_result is None:
            continue
        metadata = _result_extra(tool_result, message=result, call_id=call_id)
        process_id = metadata.get("process_id")
        if metadata.get("process_status") == "running" and isinstance(process_id, str):
            return process_id
    return None


def _channel_process_metadata(
    message: PromptMessageExtended,
    call_id: str,
) -> dict[str, Any]:
    blocks = (message.channels or {}).get(FAST_AGENT_SHELL_PROCESS_METADATA)
    if blocks is None or len(blocks) != 1:
        return {}
    block = next(iter(blocks))
    if not isinstance(block, TextContent):
        return {}
    try:
        payload = json.loads(block.text)
    except json.JSONDecodeError:
        return {}
    call_metadata = payload.get(call_id) if isinstance(payload, dict) else None
    return call_metadata if isinstance(call_metadata, dict) else {}


def _result_extra(
    result: CallToolResult,
    *,
    message: PromptMessageExtended,
    call_id: str,
) -> dict[str, Any]:
    durable = (result.meta or {}).get(FAST_AGENT_SHELL_PROCESS_METADATA)
    if isinstance(durable, dict):
        return durable
    return _channel_process_metadata(message, call_id)


def _process_status(exchange: PollExchange) -> str | None:
    status = exchange.metadata.get("process_status")
    return status if isinstance(status, str) else None


def _has_resource_observation(exchange: PollExchange) -> bool:
    return isinstance(exchange.metadata.get("resource_observation"), str)


def _output_line_count(exchange: PollExchange) -> int | None:
    return _non_negative_int(exchange.metadata.get("output_line_count"))


def _exchange(
    request: PromptMessageExtended,
    result: PromptMessageExtended,
    *,
    request_index: int,
) -> PollExchange | None:
    parsed = _single_poll_call(request)
    if parsed is None:
        return None
    call_id, process_id, wait_sec, wake_on_output = parsed
    tool_result = _single_poll_result(result, call_id)
    if tool_result is None:
        return None
    metadata = _result_extra(tool_result, message=result, call_id=call_id)
    result_process_id = metadata.get("process_id")
    if result_process_id != process_id:
        return None
    if wait_sec is None:
        wait_sec = _non_negative_int(metadata.get("poll_wait_sec"))
        if wait_sec is None:
            return None
    return PollExchange(
        request_index=request_index,
        call_id=call_id,
        process_id=process_id,
        wait_sec=wait_sec,
        wake_on_output=wake_on_output,
        request=request,
        result_message=result,
        result=tool_result,
        metadata=metadata,
    )


def managed_process_poll_cache_boundary(
    messages: list[PromptMessageExtended],
) -> int | None:
    """Return the stable execute-result boundary for a contiguous polling suffix."""
    for result_index in range(len(messages) - 1, 0, -1):
        process_id = _managed_process_start_id(
            messages[result_index - 1], messages[result_index]
        )
        if process_id is None:
            continue

        cursor = result_index + 1
        while cursor + 1 < len(messages):
            exchange = _exchange(
                messages[cursor],
                messages[cursor + 1],
                request_index=cursor,
            )
            if exchange is None or exchange.process_id != process_id:
                break
            cursor += 2
        if cursor == len(messages):
            return result_index
    return None


def _non_negative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _non_negative_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _format_counts(counts: Mapping[Any, int], *, suffix: str = "") -> str:
    return ", ".join(
        f"{key}{suffix} × {count}"
        for key, count in sorted(counts.items(), key=lambda item: str(item[0]))
    )


def _json_channel(message: PromptMessageExtended, channel: str) -> object | None:
    blocks = (message.channels or {}).get(channel)
    if blocks is None or len(blocks) != 1:
        return None
    block = next(iter(blocks))
    if not isinstance(block, TextContent):
        return None
    try:
        return json.loads(block.text)
    except json.JSONDecodeError:
        return None


def _prior_fold_metadata(exchange: PollExchange) -> dict[str, object] | None:
    payload = _string_mapping(
        _json_channel(exchange.result_message, FAST_AGENT_PROCESS_POLL_FOLD)
    )
    if payload is None or payload.get("process_id") != exchange.process_id:
        return None
    return payload


def _string_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


_FOLDED_TOKEN_FIELDS = (
    "prompt_tokens",
    "uncached_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "tool_use_prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
)


def _attempt_token_fields(
    attempt_mapping: dict[str, object],
) -> dict[str, int | None] | None:
    prompt = _string_mapping(attempt_mapping.get("prompt"))
    completion = _string_mapping(attempt_mapping.get("completion"))
    if prompt is None or completion is None:
        return None
    raw = {
        "prompt_tokens": prompt.get("total"),
        "uncached_tokens": prompt.get("uncached"),
        "cached_tokens": prompt.get("cache_read"),
        "cache_write_tokens": prompt.get("cache_write"),
        "tool_use_prompt_tokens": prompt.get("tool_use"),
        "completion_tokens": completion.get("total"),
        "reasoning_tokens": completion.get("reasoning"),
    }
    return {field: value if type(value) is int else None for field, value in raw.items()}


def _folded_usage(exchanges: list[PollExchange]) -> dict[str, object]:
    """Summarize per-poll usage; unknown or malformed values degrade to None.

    Folds feed benchmarking exports, so a poll with missing usage must not
    discard what the other polls reported: each field is complete-or-None.
    """
    totals = Counter[str]()
    request_modes = Counter[str]()
    turns: list[dict[str, object]] = []
    unknown_fields: set[str] = set()
    provider_attempts_unknown = False
    model_duration_unknown = False
    tool_duration_unknown = False
    total_cost_usd = 0.0
    cost_unknown = False
    for exchange in exchanges:
        turn_totals = Counter[str]()
        turn_unknown_fields: set[str] = set()
        turn_provider_attempts: int | None = 0
        turn_cost_usd = 0.0
        turn_cost_unknown = False
        usage = _string_mapping(_json_channel(exchange.request, FAST_AGENT_USAGE))
        attempts = usage.get("provider_attempts") if usage is not None else None
        attempt_mappings = (
            [_string_mapping(attempt) for attempt in attempts]
            if isinstance(attempts, list) and attempts
            else None
        )
        if attempt_mappings is None or any(
            attempt_mapping is None for attempt_mapping in attempt_mappings
        ):
            turn_unknown_fields.update(_FOLDED_TOKEN_FIELDS)
            turn_provider_attempts = None
            turn_cost_unknown = True
        else:
            for attempt_mapping in attempt_mappings:
                assert attempt_mapping is not None
                fields = _attempt_token_fields(attempt_mapping) or dict.fromkeys(
                    _FOLDED_TOKEN_FIELDS
                )
                for field, value in fields.items():
                    if value is None:
                        turn_unknown_fields.add(field)
                    else:
                        turn_totals[field] += value
                turn_provider_attempts += 1
                cost = _non_negative_float(attempt_mapping.get("cost_usd"))
                if cost is None:
                    turn_cost_unknown = True
                else:
                    turn_cost_usd += cost

        unknown_fields |= turn_unknown_fields
        for field in _FOLDED_TOKEN_FIELDS:
            totals[field] += turn_totals[field]
        if turn_provider_attempts is None:
            provider_attempts_unknown = True
        else:
            totals["provider_attempts"] += turn_provider_attempts
        if turn_cost_unknown:
            cost_unknown = True
        else:
            total_cost_usd += turn_cost_usd

        timing = _string_mapping(_json_channel(exchange.request, FAST_AGENT_TIMING))
        raw_duration = timing.get("duration_ms") if timing is not None else None
        duration_ms = (
            float(raw_duration)
            if isinstance(raw_duration, (int, float))
            and not isinstance(raw_duration, bool)
            else None
        )
        if duration_ms is None:
            model_duration_unknown = True
        else:
            totals["model_duration_micros"] += round(duration_ms * 1000)

        tool_timings = _string_mapping(
            _json_channel(exchange.result_message, FAST_AGENT_TOOL_TIMING)
        )
        call_timing = (
            tool_timings.get(exchange.call_id)
            if tool_timings is not None
            else None
        )
        call_timing_mapping = _string_mapping(call_timing)
        raw_tool_duration = (
            call_timing_mapping.get("timing_ms")
            if call_timing_mapping is not None
            else None
        )
        tool_duration_ms = (
            float(raw_tool_duration)
            if isinstance(raw_tool_duration, (int, float))
            and not isinstance(raw_tool_duration, bool)
            else None
        )
        if tool_duration_ms is None:
            tool_duration_unknown = True
        else:
            totals["tool_duration_micros"] += round(tool_duration_ms * 1000)

        diagnostics = _string_mapping(
            _json_channel(exchange.request, _RESPONSES_DIAGNOSTICS_CHANNEL)
        )
        mode = None
        outcome = None
        if diagnostics is not None:
            mode = diagnostics.get("websocket_request_mode")
            if isinstance(mode, str):
                request_modes[mode] += 1
            else:
                mode = None
            raw_outcome = diagnostics.get("websocket_turn_outcome")
            outcome = raw_outcome if isinstance(raw_outcome, str) else None
        turns.append(
            {
                "wait_sec": exchange.wait_sec,
                "wake_on_output": exchange.wake_on_output,
                "yield_reason": exchange.metadata.get("process_yield_reason"),
                "poll_elapsed_seconds": exchange.metadata.get("poll_elapsed_seconds"),
                "deadline_overshoot_seconds": exchange.metadata.get(
                    "poll_deadline_overshoot_seconds"
                ),
                "output_bytes_since_last_poll": exchange.metadata.get(
                    "output_bytes_since_last_poll"
                ),
                "seconds_since_last_output": exchange.metadata.get(
                    "seconds_since_last_output"
                ),
                "has_observed_output": exchange.metadata.get(
                    "has_observed_output"
                ),
                "request_mode": mode,
                "request_outcome": outcome,
                "provider_attempts": turn_provider_attempts,
                **{
                    field: (
                        None
                        if field in turn_unknown_fields
                        else turn_totals[field]
                    )
                    for field in _FOLDED_TOKEN_FIELDS
                },
                "cost_usd": None if turn_cost_unknown else turn_cost_usd,
                "model_duration_ms": duration_ms,
                "tool_duration_ms": tool_duration_ms,
            }
        )

    folded_usage: dict[str, object] = {
        "llm_calls": len(exchanges),
        "provider_attempts": (
            None if provider_attempts_unknown else totals["provider_attempts"]
        ),
        **{
            field: None if field in unknown_fields else totals[field]
            for field in _FOLDED_TOKEN_FIELDS
        },
        "model_duration_ms": (
            None if model_duration_unknown else totals["model_duration_micros"] / 1000
        ),
        "tool_duration_ms": (
            None if tool_duration_unknown else totals["tool_duration_micros"] / 1000
        ),
        "request_modes": dict(sorted(request_modes.items())),
        "turns": turns,
    }
    if not cost_unknown:
        folded_usage["cost_usd"] = total_cost_usd
    return folded_usage


def _merge_folded_usage(
    prior: dict[str, object] | None,
    current: dict[str, object],
) -> dict[str, object]:
    """Accumulate a prior fold's usage; unmergeable fields degrade to None."""
    if prior is None:
        return current

    merged: dict[str, object] = {}
    for field in ("llm_calls", "provider_attempts", *_FOLDED_TOKEN_FIELDS):
        prior_value = prior.get(field)
        current_value = current.get(field)
        merged[field] = (
            prior_value + current_value
            if type(prior_value) is int and type(current_value) is int
            else None
        )

    for field in ("model_duration_ms", "tool_duration_ms"):
        prior_value = _non_negative_float(prior.get(field))
        current_value = _non_negative_float(current.get(field))
        merged[field] = (
            prior_value + current_value
            if prior_value is not None and current_value is not None
            else None
        )

    prior_cost = _non_negative_float(prior.get("cost_usd"))
    current_cost = _non_negative_float(current.get("cost_usd"))
    if prior_cost is not None and current_cost is not None:
        merged["cost_usd"] = prior_cost + current_cost

    request_modes = Counter[str]()
    for modes in (
        _string_mapping(prior.get("request_modes")),
        _string_mapping(current.get("request_modes")),
    ):
        for mode, count in (modes or {}).items():
            if type(count) is int:
                request_modes[mode] += count
    merged["request_modes"] = dict(sorted(request_modes.items()))

    prior_turns = prior.get("turns")
    current_turns = current.get("turns")
    merged["turns"] = [
        *(prior_turns if isinstance(prior_turns, list) else []),
        *(current_turns if isinstance(current_turns, list) else []),
    ]
    return merged


def _counter_from_metadata(
    metadata: dict[str, object],
    field: str,
    *,
    integer_keys: bool = False,
) -> Counter[object]:
    raw = _string_mapping(metadata.get(field))
    counter = Counter[object]()
    if raw is None:
        return counter
    for key, count in raw.items():
        if type(count) is not int:
            continue
        normalized: object = int(key) if integer_keys and key.isdigit() else key
        counter[normalized] += count
    return counter


def _archive_exchange(exchange: PollExchange) -> ArchivedPollExchange:
    return ArchivedPollExchange(
        call_id=exchange.call_id,
        request=exchange.request,
        result=exchange.result_message,
    )


def _assistant_update(exchange: PollExchange) -> dict[str, object] | None:
    if not exchange.request.content:
        return None
    return {
        "call_id": exchange.call_id,
        "content": [
            cast(
                "dict[str, object]",
                block.model_dump(mode="json", exclude_none=True),
            )
            for block in exchange.request.content
        ],
    }


def _assistant_updates(
    exchanges: list[PollExchange],
    *,
    prior_fold: dict[str, object] | None,
    prior_fold_exchange: PollExchange | None,
) -> list[dict[str, object]] | None:
    updates: list[dict[str, object]] = []
    for exchange in exchanges:
        if exchange is prior_fold_exchange:
            assert prior_fold is not None
            prior_updates = prior_fold.get("assistant_updates", [])
            if not isinstance(prior_updates, list) or not all(
                isinstance(update, dict) for update in prior_updates
            ):
                return None
            updates.extend(cast("list[dict[str, object]]", prior_updates))
        if update := _assistant_update(exchange):
            updates.append(update)
    return updates


def _fold_audit(
    exchanges: list[PollExchange],
    *,
    folded_polls: int,
    prior_fold: dict[str, object] | None,
    prior_fold_exchange: PollExchange | None,
) -> tuple[
    list[ArchivedPollExchange],
    list[ArchivedPollExchange],
    list[str],
    ProcessPollFoldAudit | None,
] | None:
    removed_exchanges: list[ArchivedPollExchange] = []
    current_removed_call_ids = [
        exchange.call_id for exchange in exchanges[:folded_polls]
    ]
    prior_audit: ProcessPollFoldAudit | None = None
    if prior_fold is not None:
        try:
            prior_audit = ProcessPollFoldAudit.model_validate(
                prior_fold.get("audit")
            )
        except ValueError:
            return None

    for exchange in exchanges[:folded_polls]:
        if exchange is prior_fold_exchange:
            if prior_audit is None:
                return None
            removed_exchanges.extend(prior_audit.removed_exchanges)
            removed_exchanges.extend(prior_audit.retained_exchanges)
            continue

        removed_exchanges.append(_archive_exchange(exchange))

    retained_exchanges = exchanges[folded_polls:]
    return (
        removed_exchanges,
        [_archive_exchange(exchange) for exchange in retained_exchanges],
        current_removed_call_ids,
        prior_audit,
    )


def _fold_metadata(
    exchanges: list[PollExchange],
    *,
    folded_polls: int,
    retained_polls: int,
    process_status: str,
    prior_fold: dict[str, object] | None,
    prior_fold_exchange: PollExchange | None,
) -> dict[str, object] | None:
    audit_result = _fold_audit(
        exchanges,
        folded_polls=folded_polls,
        prior_fold=prior_fold,
        prior_fold_exchange=prior_fold_exchange,
    )
    if audit_result is None:
        return None
    (
        removed_archive,
        retained_archive,
        current_removed_call_ids,
        prior_audit,
    ) = audit_result

    terminal_extra = exchanges[-1].metadata
    prior_folded_polls = 0
    if prior_fold is not None and prior_fold_exchange is not None:
        prior_folded_polls = _non_negative_int(prior_fold.get("polls_folded")) or 0
        wait_counts = cast(
            "Counter[int]",
            _counter_from_metadata(
                prior_fold,
                "requested_waits",
                integer_keys=True,
            ),
        )
        wake_counts = cast(
            "Counter[str]",
            _counter_from_metadata(prior_fold, "wake_reasons"),
        )
        output_lines = _non_negative_int(prior_fold.get("output_lines")) or 0
    else:
        wait_counts = Counter[int]()
        wake_counts = Counter[str]()
        output_lines = 0
    for exchange in exchanges:
        if exchange is prior_fold_exchange:
            continue
        wait_counts[exchange.wait_sec] += 1
        reason = exchange.metadata.get("process_yield_reason")
        if isinstance(reason, str):
            wake_counts[reason] += 1
        if (count := _output_line_count(exchange)) is not None:
            output_lines += count

    metadata: dict[str, object] = {
        "process_id": exchanges[-1].process_id,
        "polls": prior_folded_polls + len(exchanges),
        "polls_folded": prior_folded_polls + folded_polls,
        "polls_retained": retained_polls,
        "messages_removed": (prior_folded_polls + folded_polls) * 2,
        "requested_waits": dict(sorted(wait_counts.items())),
        "wake_reasons": dict(sorted(wake_counts.items())),
        "output_lines": output_lines,
        "terminal_status": process_status,
    }
    if (
        elapsed := _non_negative_float(terminal_extra.get("process_elapsed_seconds"))
    ) is not None:
        metadata["elapsed_seconds"] = round(elapsed, 3)
    if (
        output_bytes := _non_negative_int(terminal_extra.get("total_output_bytes"))
    ) is not None:
        metadata["output_bytes"] = output_bytes
    if (
        seconds_since_last_output := _non_negative_float(
            terminal_extra.get("seconds_since_last_output")
        )
    ) is not None:
        metadata["seconds_since_last_output"] = round(
            seconds_since_last_output,
            3,
        )
    has_observed_output = terminal_extra.get("has_observed_output")
    if type(has_observed_output) is bool:
        metadata["has_observed_output"] = has_observed_output
    exit_code = terminal_extra.get("exit_code")
    if type(exit_code) is int:
        metadata["exit_code"] = exit_code
    current_usage = _folded_usage(exchanges[:folded_polls])
    prior_usage = (
        _string_mapping(prior_fold.get("folded_usage"))
        if prior_fold is not None
        else None
    )
    metadata["folded_usage"] = _merge_folded_usage(prior_usage, current_usage)
    assistant_updates = _assistant_updates(
        exchanges[:folded_polls],
        prior_fold=prior_fold,
        prior_fold_exchange=prior_fold_exchange,
    )
    if assistant_updates is None:
        return None
    if assistant_updates:
        metadata["assistant_updates"] = assistant_updates
    context_rewrites = (
        list(prior_audit.context_rewrites)
        if prior_audit is not None and prior_fold_exchange is not None
        else []
    )
    retained_call_ids = [
        exchange.call_id for exchange in retained_archive
    ]
    # Each rewrite records only what this fold changed (its own usage and
    # narrations); the live channel keeps the cumulative view. Snapshotting
    # cumulative state per rewrite would make the audit quadratic in polls.
    rewrite_fold = {
        key: value
        for key, value in metadata.items()
        if key not in {"folded_usage", "assistant_updates"}
    }
    rewrite_fold["folded_usage"] = current_usage
    current_assistant_updates = [
        update
        for exchange in exchanges[:folded_polls]
        if (update := _assistant_update(exchange))
    ]
    if current_assistant_updates:
        rewrite_fold["assistant_updates"] = current_assistant_updates
    context_rewrites.append(
        ArchivedContextRewrite(
            after_call_id=retained_archive[-1].call_id,
            summary=_summary_text(metadata),
            fold=rewrite_fold,
            removed_call_ids=current_removed_call_ids,
            retained_call_ids=retained_call_ids,
        )
    )
    audit = ProcessPollFoldAudit(
        removed_exchanges=removed_archive,
        retained_exchanges=retained_archive,
        context_rewrites=context_rewrites,
    )
    metadata["audit"] = cast(
        "dict[str, object]",
        audit.model_dump(mode="json", exclude_none=True),
    )
    return metadata


def _summary_text(metadata: dict[str, object]) -> str:
    wait_counts = Counter(cast("dict[int, int]", metadata["requested_waits"]))
    wake_counts = Counter(cast("dict[str, int]", metadata["wake_reasons"]))
    lines = [
        "[MANAGED PROCESS POLLING FOLDED]",
        (
            f"{metadata['process_id']}: {metadata['polls']} polls; "
            f"{metadata['polls_folded']} earlier polls folded and "
            f"{metadata['polls_retained']} latest poll(s) retained."
        ),
        f"Requested waits: {_format_counts(wait_counts, suffix='s') or 'none'}.",
        f"Wake reasons: {_format_counts(wake_counts) or 'unavailable'}.",
        (
            f"Observed output: {metadata['output_lines']} lines"
            + (
                f" / {metadata['output_bytes']} bytes."
                if "output_bytes" in metadata
                else "."
            )
        ),
        (
            f"Current status: {metadata['terminal_status']}"
            + (
                f", exit code {metadata['exit_code']}."
                if "exit_code" in metadata
                else "."
            )
        ),
    ]
    if "elapsed_seconds" in metadata:
        lines.insert(2, f"Process elapsed time: {metadata['elapsed_seconds']} seconds.")
    if "seconds_since_last_output" in metadata:
        activity = (
            f"{metadata['seconds_since_last_output']} seconds since last output."
            if metadata.get("has_observed_output") is True
            else (
                "No output observed for "
                f"{metadata['seconds_since_last_output']} seconds."
            )
        )
        lines.insert(
            3,
            f"Output activity: {activity}",
        )
    assistant_updates = metadata.get("assistant_updates")
    if isinstance(assistant_updates, list) and assistant_updates:
        lines.append("Assistant updates from folded polls:")
        for raw_update in assistant_updates:
            update = _string_mapping(raw_update)
            if update is None:
                continue
            call_id = update.get("call_id")
            content = update.get("content")
            if not isinstance(call_id, str) or not isinstance(content, list):
                continue
            lines.append(f"[{call_id}]")
            for block in content:
                serialized = _string_mapping(block)
                text = serialized.get("text") if serialized is not None else None
                lines.append(
                    text
                    if isinstance(text, str)
                    else json.dumps(block, ensure_ascii=False, sort_keys=True)
                )
    lines.append("The retained latest poll result follows.")
    return "\n".join(lines)


def fold_managed_process_poll_history(
    history: list[PromptMessageExtended],
    tool_message: PromptMessageExtended,
) -> ProcessPollFold | None:
    """Fold a contiguous poll suffix while preserving the latest relevant pairs."""
    if not history:
        return None

    current = _exchange(history[-1], tool_message, request_index=len(history) - 1)
    if current is None:
        return None
    process_status = _process_status(current)
    if process_status not in _FOLDABLE_PROCESS_STATUSES:
        return None
    if _has_resource_observation(current):
        return None

    reverse_exchanges = [current]
    cursor = len(history) - 2
    while cursor >= 1:
        request = history[cursor - 1]
        result = history[cursor]
        exchange = _exchange(request, result, request_index=cursor - 1)
        if exchange is None or exchange.process_id != current.process_id:
            break
        if _has_resource_observation(exchange):
            break
        reverse_exchanges.append(exchange)
        cursor -= 2

    exchanges = list(reversed(reverse_exchanges))
    retained_polls = (
        1 if process_status in _SINGLE_RETAINED_POLL_STATUSES else 2
    )
    retained_polls = min(retained_polls, len(exchanges))
    folded_polls = len(exchanges) - retained_polls
    removed_exchanges = exchanges[:folded_polls]
    prior_folds = [
        (exchange, metadata)
        for exchange in removed_exchanges
        if (metadata := _prior_fold_metadata(exchange)) is not None
    ]
    if len(prior_folds) > 1:
        return None
    prior_fold_exchange, prior_fold = (
        prior_folds[0] if prior_folds else (None, None)
    )
    # A running process refolds only once another full fold's worth of quiet
    # polls has accumulated, amortizing the refold cost instead of paying it on
    # every poll; a terminal status compacts whatever remains.
    minimum_folded_polls = (
        1
        if prior_fold is not None and process_status != "running"
        else _MIN_FOLDED_POLLS
    )
    if folded_polls < minimum_folded_polls:
        return None
    if any(_output_line_count(exchange) != 0 for exchange in removed_exchanges):
        return None

    first_removed = exchanges[0].request_index
    first_retained = exchanges[folded_polls].request_index
    folded_history = [*history[:first_removed], *history[first_retained:]]

    metadata = _fold_metadata(
        exchanges,
        folded_polls=folded_polls,
        retained_polls=retained_polls,
        process_status=process_status,
        prior_fold=prior_fold,
        prior_fold_exchange=prior_fold_exchange,
    )
    if metadata is None:
        return None
    folded_tool_message = tool_message.model_copy(deep=True)
    folded_results = folded_tool_message.tool_results
    if folded_results is None:
        return None
    call_id = next(iter(folded_results))
    terminal_result = folded_results[call_id]
    terminal_result.content.insert(
        0,
        TextContent(type="text", text=_summary_text(metadata)),
    )
    channels = dict(folded_tool_message.channels or {})
    channels[FAST_AGENT_PROCESS_POLL_FOLD] = [
        TextContent(type="text", text=json.dumps(metadata, sort_keys=True))
    ]
    folded_tool_message.channels = channels
    return ProcessPollFold(
        history=folded_history,
        tool_message=folded_tool_message,
        metadata=metadata,
    )
