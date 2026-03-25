#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev

NUMERIC_FIELDS: tuple[str, ...] = (
    "input_tokens",
    "effective_input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "tool_calls",
    "cache_hit_tokens",
)


@dataclass(frozen=True, slots=True)
class TurnUsageRecord:
    session_id: str
    history_file: str
    message_index: int
    request_mode: str | None
    turn_outcome: str | None
    input_tokens: int | None
    effective_input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None
    tool_calls: int | None
    cache_hit_tokens: int | None


@dataclass(frozen=True, slots=True)
class FieldStats:
    name: str
    count: int
    minimum: int
    maximum: int
    mean: float
    cv_percent: float

    @property
    def spread(self) -> int:
        return self.maximum - self.minimum


@dataclass(frozen=True, slots=True)
class AccountingStats:
    checked_effective: int
    mismatched_effective: int
    max_effective_delta: int
    checked_total: int
    mismatched_total: int
    max_total_delta: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize token usage variability in recent fast-agent session histories. "
            "Useful for comparing websocket fresh vs reused turns."
        )
    )
    parser.add_argument(
        "sessions_root",
        nargs="?",
        default=".cdx/sessions",
        help="Session directory root (default: .cdx/sessions).",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=5,
        help="How many most-recent session directories to inspect (default: 5).",
    )
    parser.add_argument(
        "--top-fields",
        type=int,
        default=3,
        help="How many bouncy fields to print per session (default: 3).",
    )
    return parser.parse_args()


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_dict(value: object) -> dict[str, object] | None:
    return value if isinstance(value, dict) else None


def _as_list(value: object) -> list[object] | None:
    return value if isinstance(value, list) else None


def _extract_text_payload(channel_value: object) -> dict[str, object] | None:
    entries = _as_list(channel_value)
    if not entries:
        return None
    first_entry = _as_dict(entries[0])
    if first_entry is None:
        return None
    raw_text = first_entry.get("text")
    if not isinstance(raw_text, str):
        return None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    return _as_dict(parsed)


def _get_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _select_history_files(session_dir: Path) -> list[Path]:
    session_json = session_dir / "session.json"
    if session_json.exists():
        payload = _as_dict(_load_json(session_json))
        if payload is not None:
            history_files = _as_list(payload.get("history_files"))
            if history_files is not None:
                selected: list[Path] = []
                for item in history_files:
                    if not isinstance(item, str) or item.endswith("_previous.json"):
                        continue
                    candidate = session_dir / item
                    if candidate.exists():
                        selected.append(candidate)
                if selected:
                    return sorted(selected)

    return sorted(
        path
        for path in session_dir.glob("history_*.json")
        if not path.name.endswith("_previous.json")
    )


def _load_turn_records(session_dir: Path) -> list[TurnUsageRecord]:
    records: list[TurnUsageRecord] = []
    for history_path in _select_history_files(session_dir):
        payload = _as_dict(_load_json(history_path))
        if payload is None:
            continue
        messages = _as_list(payload.get("messages"))
        if messages is None:
            continue

        for index, message_obj in enumerate(messages):
            message = _as_dict(message_obj)
            if message is None:
                continue
            if message.get("role") != "assistant":
                continue

            channels = _as_dict(message.get("channels")) or {}
            usage_payload = _extract_text_payload(channels.get("fast-agent-usage"))
            if usage_payload is None:
                continue
            turn_payload = _as_dict(usage_payload.get("turn")) or {}
            cache_usage = _as_dict(turn_payload.get("cache_usage")) or {}
            diag_payload = _extract_text_payload(
                channels.get("fast-agent-provider-diagnostics")
            ) or {}

            records.append(
                TurnUsageRecord(
                    session_id=session_dir.name,
                    history_file=history_path.name,
                    message_index=index,
                    request_mode=(
                        diag_payload.get("websocket_request_mode")
                        if isinstance(diag_payload.get("websocket_request_mode"), str)
                        else None
                    ),
                    turn_outcome=(
                        diag_payload.get("websocket_turn_outcome")
                        if isinstance(diag_payload.get("websocket_turn_outcome"), str)
                        else None
                    ),
                    input_tokens=_get_int(turn_payload.get("input_tokens")),
                    effective_input_tokens=_get_int(
                        turn_payload.get("effective_input_tokens")
                    ),
                    output_tokens=_get_int(turn_payload.get("output_tokens")),
                    total_tokens=_get_int(turn_payload.get("total_tokens")),
                    reasoning_tokens=_get_int(turn_payload.get("reasoning_tokens")),
                    tool_calls=_get_int(turn_payload.get("tool_calls")),
                    cache_hit_tokens=_get_int(cache_usage.get("cache_hit_tokens")),
                )
            )
    return records


def _field_values(records: list[TurnUsageRecord], field_name: str) -> list[int]:
    values: list[int] = []
    for record in records:
        value = getattr(record, field_name)
        if isinstance(value, int):
            values.append(value)
    return values


def _compute_field_stats(records: list[TurnUsageRecord]) -> list[FieldStats]:
    stats: list[FieldStats] = []
    for field_name in NUMERIC_FIELDS:
        values = _field_values(records, field_name)
        if not values:
            continue
        mean_value = fmean(values)
        cv_percent = 0.0
        if len(values) > 1 and mean_value != 0:
            cv_percent = (pstdev(values) / mean_value) * 100.0
        stats.append(
            FieldStats(
                name=field_name,
                count=len(values),
                minimum=min(values),
                maximum=max(values),
                mean=mean_value,
                cv_percent=cv_percent,
            )
        )
    return sorted(stats, key=lambda item: (-item.cv_percent, -item.spread, item.name))


def _group_values(
    records: list[TurnUsageRecord],
    *,
    outcome: str,
    field_name: str,
) -> list[int]:
    values: list[int] = []
    for record in records:
        if record.turn_outcome != outcome:
            continue
        value = getattr(record, field_name)
        if isinstance(value, int):
            values.append(value)
    return values


def _avg_or_dash(values: list[int]) -> str:
    if not values:
        return "-"
    return f"{fmean(values):.1f}"


def _fmt_range(records: list[TurnUsageRecord], field_name: str) -> str:
    values = _field_values(records, field_name)
    if not values:
        return "-"
    return f"{min(values)}..{max(values)}"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header_line = "  ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    separator_line = "  ".join("-" * width for width in widths)
    print(header_line)
    print(separator_line)
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def _compute_accounting_stats(records: list[TurnUsageRecord]) -> AccountingStats:
    checked_effective = 0
    mismatched_effective = 0
    max_effective_delta = 0
    checked_total = 0
    mismatched_total = 0
    max_total_delta = 0

    for record in records:
        if (
            isinstance(record.input_tokens, int)
            and isinstance(record.cache_hit_tokens, int)
            and isinstance(record.effective_input_tokens, int)
        ):
            checked_effective += 1
            expected_effective = record.input_tokens - record.cache_hit_tokens
            delta = record.effective_input_tokens - expected_effective
            max_effective_delta = max(max_effective_delta, abs(delta))
            if delta != 0:
                mismatched_effective += 1

        if (
            isinstance(record.input_tokens, int)
            and isinstance(record.output_tokens, int)
            and isinstance(record.total_tokens, int)
        ):
            checked_total += 1
            expected_total = record.input_tokens + record.output_tokens
            delta = record.total_tokens - expected_total
            max_total_delta = max(max_total_delta, abs(delta))
            if delta != 0:
                mismatched_total += 1

    return AccountingStats(
        checked_effective=checked_effective,
        mismatched_effective=mismatched_effective,
        max_effective_delta=max_effective_delta,
        checked_total=checked_total,
        mismatched_total=mismatched_total,
        max_total_delta=max_total_delta,
    )


def main() -> int:
    args = _parse_args()
    sessions_root = Path(args.sessions_root)
    if not sessions_root.exists():
        print(f"session root not found: {sessions_root}")
        return 1

    session_dirs = sorted(
        (path for path in sessions_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )[: args.last]
    if not session_dirs:
        print(f"no session directories found in {sessions_root}")
        return 1

    session_records: list[tuple[Path, list[TurnUsageRecord], list[FieldStats]]] = []
    for session_dir in session_dirs:
        records = _load_turn_records(session_dir)
        if not records:
            continue
        session_records.append((session_dir, records, _compute_field_stats(records)))

    if not session_records:
        print("no assistant usage records found")
        return 0

    overview_rows: list[list[str]] = []
    effect_rows: list[list[str]] = []
    bounce_rows: list[list[str]] = []
    accounting_rows: list[list[str]] = []

    for session_dir, records, field_stats in session_records:
        accounting = _compute_accounting_stats(records)
        fresh_count = sum(1 for record in records if record.turn_outcome == "fresh")
        reused_count = sum(1 for record in records if record.turn_outcome == "reused")
        reconnected_count = sum(
            1 for record in records if record.turn_outcome == "reconnected"
        )
        top_field = field_stats[0] if field_stats else None
        overview_rows.append(
            [
                session_dir.name,
                str(len(records)),
                str(fresh_count),
                str(reused_count),
                str(reconnected_count),
                top_field.name if top_field is not None else "-",
                f"{top_field.cv_percent:.1f}" if top_field is not None else "-",
                _fmt_range(records, "output_tokens"),
                _fmt_range(records, "effective_input_tokens"),
                _fmt_range(records, "cache_hit_tokens"),
            ]
        )

        effect_rows.append(
            [
                session_dir.name,
                _avg_or_dash(_group_values(records, outcome="fresh", field_name="input_tokens")),
                _avg_or_dash(
                    _group_values(
                        records,
                        outcome="fresh",
                        field_name="effective_input_tokens",
                    )
                ),
                _avg_or_dash(_group_values(records, outcome="fresh", field_name="output_tokens")),
                _avg_or_dash(
                    _group_values(records, outcome="reused", field_name="input_tokens")
                ),
                _avg_or_dash(
                    _group_values(
                        records,
                        outcome="reused",
                        field_name="effective_input_tokens",
                    )
                ),
                _avg_or_dash(
                    _group_values(records, outcome="reused", field_name="output_tokens")
                ),
                _avg_or_dash(
                    _group_values(records, outcome="reused", field_name="cache_hit_tokens")
                ),
            ]
        )
        accounting_rows.append(
            [
                session_dir.name,
                f"{accounting.mismatched_effective}/{accounting.checked_effective}",
                str(accounting.max_effective_delta),
                f"{accounting.mismatched_total}/{accounting.checked_total}",
                str(accounting.max_total_delta),
            ]
        )

        for stat in field_stats[: args.top_fields]:
            bounce_rows.append(
                [
                    session_dir.name,
                    stat.name,
                    str(stat.count),
                    str(stat.minimum),
                    str(stat.maximum),
                    f"{stat.mean:.1f}",
                    str(stat.spread),
                    f"{stat.cv_percent:.1f}",
                ]
            )

    print()
    print("Session overview")
    _print_table(
        [
            "session",
            "turns",
            "fresh",
            "reused",
            "reconn",
            "top_field",
            "cv%",
            "output_range",
            "effective_in_range",
            "cache_hit_range",
        ],
        overview_rows,
    )

    print()
    print("Fresh vs reused websocket effect (averages)")
    _print_table(
        [
            "session",
            "fresh_in",
            "fresh_eff_in",
            "fresh_out",
            "reused_in",
            "reused_eff_in",
            "reused_out",
            "reused_cache_hit",
        ],
        effect_rows,
    )

    print()
    print("Accounting consistency checks")
    _print_table(
        [
            "session",
            "effective!=input-cache",
            "max_eff_delta",
            "total!=input+output",
            "max_total_delta",
        ],
        accounting_rows,
    )

    print()
    print(f"Most variable fields per session (top {args.top_fields})")
    _print_table(
        ["session", "field", "count", "min", "max", "mean", "spread", "cv%"],
        bounce_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
