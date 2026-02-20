"""Unit tests for the event replay script helpers."""

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

from fast_agent.core.logging.events import Event
from fast_agent.event_progress import ProgressAction, ProgressEvent


def _load_event_replay_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "event_replay.py"
    spec = importlib.util.spec_from_file_location("event_replay_script", script_path)
    assert spec is not None
    loader = spec.loader
    assert loader is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_event_replay_module = _load_event_replay_module()
_compute_replay_delay = _event_replay_module._compute_replay_delay
_select_events = _event_replay_module._select_events
load_events = _event_replay_module.load_events
replay_events = _event_replay_module.replay_events


class DisplayProbe:
    """Collects progress updates emitted by replay_events."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def update(self, event: ProgressEvent) -> None:
        self.events.append(event)


def _event(
    *,
    timestamp: datetime,
    action: ProgressAction | None,
    namespace: str = "fast_agent.llm.provider.openai.llm_openai",
) -> Event:
    data: dict[str, object] = {}
    if action is not None:
        data = {
            "data": {
                "progress_action": action,
                "agent_name": "assistant",
                "target": "assistant",
                "model": "gpt-5",
            }
        }
    return Event(
        type="info",
        namespace=namespace,
        message="event",
        timestamp=timestamp,
        data=data,
    )


def test_compute_replay_delay_scales_and_clamps() -> None:
    start = datetime(2026, 2, 20, 12, 0, 0)
    end = start + timedelta(seconds=4)

    assert _compute_replay_delay(start, end, speed=2.0) == 2.0
    assert _compute_replay_delay(end, start, speed=2.0) == 0.0


def test_select_events_applies_window_and_validates() -> None:
    now = datetime(2026, 2, 20, 12, 0, 0)
    events = [_event(timestamp=now + timedelta(seconds=i), action=None) for i in range(5)]

    selected = _select_events(events, start_at=1, end_at=4)
    assert len(selected) == 3

    with pytest.raises(ValueError):
        _select_events(events, start_at=-1, end_at=None)

    with pytest.raises(ValueError):
        _select_events(events, start_at=3, end_at=2)


def test_replay_events_replays_progress_with_scaled_timing() -> None:
    now = datetime(2026, 2, 20, 12, 0, 0)
    events = [
        _event(timestamp=now, action=ProgressAction.CHATTING),
        _event(timestamp=now + timedelta(seconds=1), action=None),
        _event(timestamp=now + timedelta(seconds=3), action=ProgressAction.THINKING),
    ]

    sleeps: list[float] = []

    def _record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    display = DisplayProbe()
    stats = replay_events(events, display=display, speed=2.0, sleep_func=_record_sleep)

    assert stats.total_events == 3
    assert stats.progress_events == 2
    assert len(display.events) == 2
    assert sleeps == [0.5, 1.0]


def test_load_events_parses_jsonl_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    log_path.write_text(
        "\n".join(
            [
                '{"level":"INFO","timestamp":"2026-02-20T12:00:00","namespace":"n","message":"m"}',
                '{"level":"ERROR","timestamp":"2026-02-20T12:00:01","namespace":"n2","message":"m2"}',
            ]
        )
    )

    events = load_events(log_path)

    assert [event.type for event in events] == ["info", "error"]
    assert [event.namespace for event in events] == ["n", "n2"]
