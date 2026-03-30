"""Optional lifecycle hooks for meeting events.

Design follows the ``ToolRunnerHooks`` pattern — a dataclass with
optional callables.  When no hooks are set (the default), all
callbacks are ``None`` and the library runs with zero overhead.

Usage::

    from fast_agent.spawn.servers.meeting_hooks import MeetingHooks

    hooks = MeetingHooks(
        on_transcript_entry=lambda mid, entry: broadcast(mid, entry),
        on_meeting_ended=lambda mid, outcome: log_end(mid, outcome),
    )

    from fast_agent.spawn.servers.meeting_room_server import configure_meeting_room
    configure_meeting_room(hooks=hooks)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class MeetingHooks:
    """Lifecycle hooks for meeting events.

    Every field is ``None`` by default — the library works without any
    hooks configured.  Host applications set specific callbacks to
    receive real-time events.

    Hook signatures:

    - ``on_meeting_created(meeting_id, config)``
    - ``on_participant_joined(meeting_id, agent_name, all_joined)``
    - ``on_meeting_started(meeting_id)``
    - ``on_transcript_entry(meeting_id, entry_dict)``
    - ``on_turn_advanced(meeting_id, next_speaker, round_num)``
    - ``on_verdict(meeting_id, verdict_str, by_agent)``
    - ``on_meeting_ended(meeting_id, outcome)``
    - ``on_participant_left(meeting_id, agent_name)``
    - ``on_participant_added(meeting_id, agent_name)``
    - ``on_state_changed(meeting_id, state_dict)``
    - ``on_audit(meeting_id, message)``
    """

    on_meeting_created: Callable[[str, dict], None] | None = None
    on_participant_joined: Callable[[str, str, bool], None] | None = None
    on_meeting_started: Callable[[str], None] | None = None
    on_transcript_entry: Callable[[str, dict], None] | None = None
    on_turn_advanced: Callable[[str, str, int], None] | None = None
    on_verdict: Callable[[str, str, str], None] | None = None
    on_meeting_ended: Callable[[str, str], None] | None = None
    on_participant_left: Callable[[str, str], None] | None = None
    on_participant_added: Callable[[str, str], None] | None = None
    on_state_changed: Callable[[str, dict], None] | None = None
    on_audit: Callable[[str, str], None] | None = None


def merge_hooks(a: MeetingHooks, b: MeetingHooks) -> MeetingHooks:
    """Compose two ``MeetingHooks`` instances.

    For each callback, if both *a* and *b* define it, the merged hook
    calls *a* first then *b*.  If only one defines it, that one is used.
    """

    def _merge_fn(fn_a, fn_b):  # type: ignore[no-untyped-def]
        if fn_a is None:
            return fn_b
        if fn_b is None:
            return fn_a

        def merged(*args, **kwargs):  # type: ignore[no-untyped-def]
            fn_a(*args, **kwargs)
            fn_b(*args, **kwargs)

        return merged

    fields = {
        f.name: _merge_fn(getattr(a, f.name), getattr(b, f.name))
        for f in MeetingHooks.__dataclass_fields__.values()
    }
    return MeetingHooks(**fields)
