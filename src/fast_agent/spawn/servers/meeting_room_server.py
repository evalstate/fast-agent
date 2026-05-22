"""Meeting Room MCP Server — multi-agent meetings with shared conversation.

Architecture: Each agent subprocess gets its OWN instance of this server.
Therefore all state is FILE-BASED (shared workspace directory). Turn
coordination uses file polling instead of ``asyncio.Event``, since agents
run as separate processes.

Storage & hooks are **pluggable** via ``configure_meeting_room()``.  By
default the library uses ``JsonFileMeetingStorage`` and no-op hooks so it
works standalone without any configuration.

Storage layout (default JSON backend)::

    workspace/meetings/{meeting_id}/
        config.json       — meeting configuration
        state.json        — current turn, round, joined set, ended flag
        transcript.json   — conversation messages
        audit.log         — append-only human-readable log

Concurrency: file locking via ``fcntl.flock()`` protects state mutations.
Turn wait: polling ``state.json`` every 2 s (acceptable for LLM interactions).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from fast_agent.spawn.servers.meeting_hooks import MeetingHooks
from fast_agent.spawn.servers.meeting_storage import (
    JsonFileMeetingStorage,
    MeetingStorage,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("meeting-room")


# ───────────────────────────────────────────────────────────────────
# Pluggable storage + hooks — module-level singletons
# ───────────────────────────────────────────────────────────────────

_storage: MeetingStorage = JsonFileMeetingStorage()
_hooks: MeetingHooks = MeetingHooks()  # NoOp by default


def configure_meeting_room(
    storage: MeetingStorage | None = None,
    hooks: MeetingHooks | None = None,
) -> None:
    """Override the storage backend and/or lifecycle hooks.

    This is **optional** — the library works without calling this.
    Host applications call this during startup to inject custom
    implementations (e.g. SQLite storage, SSE broadcast hooks).
    """
    global _storage, _hooks
    if storage is not None:
        _storage = storage
    if hooks is not None:
        _hooks = hooks


def _fire_hook(hook_name: str, *args: Any) -> None:
    """Safely invoke a hook callback if it is set."""
    fn = getattr(_hooks, hook_name, None)
    if fn is not None:
        try:
            fn(*args)
        except Exception as e:
            logger.warning("Meeting hook %s failed: %s", hook_name, e)


def _audit(meeting_id: str, message: str) -> None:
    """Append to the audit log (file-based, kept for backward compat)."""
    # For JsonFileMeetingStorage, also write the audit.log file
    if isinstance(_storage, JsonFileMeetingStorage):
        d = _storage._meeting_dir(meeting_id)
        audit_path = d / "audit.log"
        ts = datetime.now().isoformat(timespec="seconds")
        try:
            with open(audit_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {message}\n")
        except OSError:
            pass
    _fire_hook("on_audit", meeting_id, message)


def _notify_turn_agent(
    meeting_id: str, agent_name: str, agenda: str, round_num: int
) -> None:
    """Send YOUR_TURN notification with embedded unread transcript.

    Transcript is auto-pushed — agents never need to call get_transcript.
    Each notification clearly labels the meeting ID for multi-meeting support.
    """
    bus = _get_bus()
    if not bus:
        _auto_wake_if_idle(agent_name)
        return

    # Fetch and advance read cursor for this agent
    transcript = _storage.get_transcript(meeting_id)
    with _storage.acquire_lock(meeting_id) as conn:
        state = _storage.get_state(meeting_id, _conn=conn) or {}
        cursors: dict[str, int] = state.get("read_cursors", {})
        last_read = cursors.get(agent_name, 0)
        unread = transcript[last_read:]
        cursors[agent_name] = len(transcript)
        state["read_cursors"] = cursors
        _storage.update_state(meeting_id, state, _conn=conn)

    # Format unread transcript lines
    transcript_lines = []
    for entry in unread:
        transcript_lines.append(
            f"  [{entry['agent']}] (round {entry.get('round', '?')}): "
            f"{entry['message']}"
        )
    transcript_text = (
        "\n".join(transcript_lines) if transcript_lines else "(no new messages)"
    )

    msg = (
        f"🎙️ YOUR TURN TO SPEAK — Meeting [{meeting_id}]\n"
        f"Agenda: {agenda}\n"
        f"Round: {round_num}\n\n"
        f"📋 Transcript ({len(unread)} new messages):\n"
        f"{transcript_text}\n\n"
        f'→ speak(meeting_id="{meeting_id}", message="...")'
    )
    bus.send(
        from_name=f"Meeting [{meeting_id}]",
        to_name=agent_name,
        content=msg,
        message_type="meeting_turn",
    )
    _auto_wake_if_idle(agent_name)
    logger.info("🎙️ Notified %s: your turn in %s (round %d, %d unread)",
                agent_name, meeting_id, round_num, len(unread))


def _notify_meeting_started(
    meeting_id: str, agent_name: str, agenda: str, participants: list[str]
) -> None:
    """Inform non-first-speaker agents that the meeting has started."""
    bus = _get_bus()
    if not bus:
        return
    participant_names = ", ".join(participants)
    msg = (
        f"📋 Meeting [{meeting_id}] started.\n"
        f"Agenda: {agenda}\n"
        f"Participants: {participant_names}\n\n"
        f"You will receive a 🎙️ YOUR TURN notification when it's your turn to speak."
    )
    bus.send(
        from_name=f"Meeting [{meeting_id}]",
        to_name=agent_name,
        content=msg,
        message_type="meeting_started",
    )
    _auto_wake_if_idle(agent_name)


def _notify_meeting_ended(
    meeting_id: str, agent_name: str, agenda: str
) -> None:
    """Send meeting-ended notification with full transcript."""
    bus = _get_bus()
    if not bus:
        return
    transcript = _storage.get_transcript(meeting_id)
    transcript_lines = [
        f"  [{e['agent']}] (round {e.get('round', '?')}): {e['message']}"
        for e in transcript
    ]
    msg = (
        f"📋 Meeting [{meeting_id}] has ended.\n"
        f"Agenda: {agenda}\n\n"
        f"Full transcript ({len(transcript)} messages):\n"
        + "\n".join(transcript_lines)
    )
    bus.send(
        from_name=f"Meeting [{meeting_id}]",
        to_name=agent_name,
        content=msg,
        message_type="meeting_ended",
    )
    _auto_wake_if_idle(agent_name)


# ───────────────────────────────────────────────────────────────────
# MCP Tools
# ───────────────────────────────────────────────────────────────────


AGENDA_MAX_LEN = 120


@mcp.tool()
async def create_meeting(
    agenda: str,
    participants: str,
    description: str = "",
    max_rounds: int = 3,
    my_name: str = "",
    workspace_dir: str = "",
) -> str:
    """Create a meeting room — all participants are auto-joined immediately.

    🎙️ Meetings are for REAL-TIME discussion — faster than post_message.
    All participants are auto-joined and the first speaker is notified immediately.
    No need for participants to call join_meeting.

    YOU (the creator) are AUTO-INCLUDED as the first speaker. You don't need
    to repeat your own name in ``participants`` — just list the OTHERS you
    want at the table. As organizer you kick off the meeting and get a turn
    each round (typically to issue [DECISION] VERDICT to end it).

    Use meetings for: kickoff, design review, code review, blocker resolution.
    Use post_message for: task assignments, status updates, async notifications.

    Args:
        agenda: Short title (max 120 chars — STRICTLY ENFORCED, longer
                input is truncated). This is the meeting title shown in the
                dashboard. Examples: "Sprint 1 kickoff", "Blocker review
                cho SCRUM-5". Do NOT paste full project briefs here — use
                ``description`` for long-form context.
        participants: Comma-separated AGENT NAMES of the OTHERS to invite,
                      in speaking order after you. You are inserted at the
                      front automatically. E.g. "Devon [BA], Reese [Dev],
                      Devon [QE]" — final order will be ``[YOU, BA, Dev, QE]``.
                      Including yourself in this list is harmless (deduped).
        description: OPTIONAL long-form context (project brief, links,
                     deliverables, etc.). No length limit — render as
                     markdown in dashboard. Put ALL the long stuff here,
                     not in agenda. Empty by default.
        max_rounds: Maximum conversation rounds (default 3).
        my_name: YOUR agent name (auto-detected from env).
        workspace_dir: Workspace path (auto-detected from env).

    Returns:
        JSON with meeting_id, config, and a ``warning`` field if any input
        was truncated.
    """
    if not workspace_dir:
        workspace_dir = os.environ.get(
            "TEAM_WORKSPACE",
            str(Path.cwd() / ".runtime" / "cache" / "tmp" / "meeting-workspace"),
        )
    my_name = my_name or _get_my_name()
    participant_list = [p.strip() for p in participants.split(",") if p.strip()]
    # The creator is the meeting chair — auto-prepend so they speak first
    # and get a turn each round (typically to issue the verdict).
    if my_name and my_name not in participant_list:
        participant_list.insert(0, my_name)
    if len(participant_list) < 2:
        return json.dumps({"error": "Need at least 2 participants (including yourself)"})

    # Enforce agenda length so a wall-of-text doesn't break the dashboard
    # title (incident b61af7db: PM stuffed a 50-line markdown brief into
    # agenda). Caller still gets full info via the warning.
    agenda_warning = ""
    if len(agenda) > AGENDA_MAX_LEN:
        original_len = len(agenda)
        agenda = agenda[:AGENDA_MAX_LEN].rstrip() + "…"
        agenda_warning = (
            f"agenda truncated from {original_len} to {AGENDA_MAX_LEN} chars — "
            f"move long-form context into the 'description' parameter"
        )
        logger.warning(
            "Meeting create: agenda truncated %d→%d chars (move to description). Original prefix: %r",
            original_len, AGENDA_MAX_LEN, agenda[:60],
        )

    # Short 8-char meeting ID for easy identification
    meeting_id = uuid.uuid4().hex[:8]

    # ``config`` holds write-once setup metadata. Anything that mutates
    # during the meeting (participants, max_rounds, turn pointers, etc.)
    # lives in ``state`` so all writes funnel through ``update_state``
    # under a single ``acquire_lock`` — no dual-write deadlock window.
    config = {
        "meeting_id": meeting_id,
        "agenda": agenda,
        "description": description,
        "created_by": my_name,
        "created_at": datetime.now().isoformat(),
    }

    # Auto-join ALL participants immediately — no join_meeting step
    state: dict[str, Any] = {
        "participants": participant_list,         # mutable: add/leave_meeting
        "max_rounds": max_rounds,                 # mutable: extended on FAIL verdict
        "current_turn": 0,
        "current_round": 1,
        "joined": list(participant_list),  # Everyone auto-joined
        "ended": False,
        "outcome": None,
        "started": True,  # Meeting starts immediately
        "read_cursors": {},  # Per-agent transcript read cursors
        "turn_started_at": time.time(),
    }

    _storage.create_meeting(meeting_id, config, state)

    _audit(
        meeting_id,
        f"Meeting created by {my_name}: agenda='{agenda}', "
        f"participants={participant_list}, max_rounds={max_rounds}",
    )
    _audit(meeting_id, "All participants auto-joined. Meeting started.")

    _fire_hook("on_meeting_created", meeting_id, config)
    _fire_hook("on_meeting_started", meeting_id)
    _fire_hook("on_state_changed", meeting_id, state)

    # Notify first speaker with empty transcript
    first_speaker = participant_list[0]
    _notify_turn_agent(meeting_id, first_speaker, agenda, 1)

    # Inform other participants that meeting started
    for participant in participant_list:
        if participant != first_speaker:
            _notify_meeting_started(meeting_id, participant, agenda, participant_list)

    logger.info("Meeting created: %s — %s (auto-joined %d)",
                meeting_id, agenda, len(participant_list))
    response = {
        "meeting_id": meeting_id,
        "agenda": agenda,
        "participants": participant_list,
        "max_rounds": max_rounds,
        "status": "started",
        "note": "All participants auto-joined. First speaker notified.",
    }
    if agenda_warning:
        response["warning"] = agenda_warning
    return json.dumps(response)


# ── REMOVED TOOLS ──
# join_meeting: Replaced by auto-join in create_meeting
# wait_for_my_turn: Replaced by transcript auto-push via _notify_turn_agent
# get_transcript: Replaced by transcript embedded in notifications


def _assert_self_identity(agent_name: str) -> tuple[str, str | None]:
    """Resolve caller identity + refuse impersonation.

    The legacy contract was ``agent_name = agent_name or _get_my_name()``
    — i.e. trust whatever name the caller supplies. That let any agent
    `speak()` or `skip_turn()` ON BEHALF OF a teammate by simply passing
    their name as ``agent_name``, since the only downstream check was
    "is this name the current speaker", not "is the caller actually
    this agent" (production 2026-05-20: Taylor [PM] force-skipped all
    6 teammates in one meeting by impersonating each one's turn —
    transcript falsely attributed identical placeholder responses to
    BA/SA/Dev/Designer/QE/DSO).

    Authoritative caller identity is ``$TEAM_MY_NAME`` — set by the
    isolated_spawner at team-spawn time, immutable for the process's
    lifetime. We read the env var DIRECTLY (not via ``_get_my_name()``)
    because that helper falls back to ``$TEAM_MY_ROLE`` and then a
    generic "agent" string; only ``TEAM_MY_NAME`` proves the process
    was spawned as a specific team member. If it's unset, the process
    is NOT a team agent (CLI tests, dashboard direct calls, library
    use) and we preserve the legacy permissive contract.

    Returns ``(resolved_name, error_json)`` — exactly one is non-empty.
    Caller short-circuits on error_json.
    """
    caller_env = os.environ.get("TEAM_MY_NAME", "").strip()

    if not agent_name:
        # No claim made — use whatever the legacy helper resolves
        # (env name, env role, or "agent"). No impersonation possible.
        return _get_my_name(), None

    if caller_env and agent_name.strip().lower() != caller_env.lower():
        return "", json.dumps({
            "error": (
                f"Impersonation refused: caller is {caller_env!r} but "
                f"agent_name={agent_name!r}. You can only speak/skip on "
                f"YOUR own turn. If a teammate is unresponsive, use "
                f"end_meeting (or leave_meeting on your own turn) — do "
                f"not put words in their mouth."
            ),
            "caller": caller_env,
            "claimed_agent_name": agent_name,
        })

    return agent_name, None


@mcp.tool()
async def speak(meeting_id: str, message: str, agent_name: str = "") -> str:
    """Add your message to the meeting transcript.

    After speaking, the turn automatically advances to the next
    participant. To end the meeting, include the ``[DECISION]`` prefix
    followed by a verdict in your message, e.g.:
    ``[DECISION] VERDICT: PASS — summary of conclusion``
    ``[DECISION] VERDICT: FAIL — reason for failure``
    ``[DECISION] VERDICT: ESCALATE`` or ``ESCALATE_TO_USER``
    FAIL continues the meeting; PASS/ESCALATE/RESOLVED ends it.

    **Important**: ``VERDICT:`` WITHOUT the ``[DECISION]`` prefix is
    ignored, so you can safely discuss verdicts without triggering
    auto-end.

    Args:
        meeting_id: The meeting to speak in.
        agent_name: YOUR agent name (must be the current speaker). Auto-detected if empty.
            ⚠️ Cannot be used to speak on behalf of another agent — if you
            pass a name that disagrees with your spawn-pinned identity,
            the call is refused. Use ``end_meeting`` if a teammate stalls.
        message: What you want to say.

    Returns:
        JSON with confirmation and next turn info.
    """
    agent_name, _err = _assert_self_identity(agent_name)
    if _err:
        return _err

    if not _storage.meeting_exists(meeting_id):
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _storage.get_config(meeting_id) or {}

    # Collect hooks to fire AFTER releasing the lock (emit_event opens a
    # new connection which would deadlock against BEGIN IMMEDIATE).
    deferred_hooks: list[tuple] = []
    result_json: str | None = None
    turn_entry: dict = {}
    next_speaker: str = ""
    current_round: int = 1
    # Captured inside the lock for outside-lock notifications.
    participants: list[str] = []

    with _storage.acquire_lock(meeting_id) as conn:
        state = _storage.get_state(meeting_id, _conn=conn) or {}

        if state.get("ended"):
            return json.dumps({"error": "Meeting already ended"})

        # Mutable meeting metadata lives in state (post-B1 refactor) so
        # all writes funnel through update_state under this single lock.
        participants = state.get("participants", [])
        max_rounds = state.get("max_rounds", 3)

        current_turn = state.get("current_turn", 0)
        current_speaker = participants[current_turn] if current_turn < len(participants) else None
        if agent_name != current_speaker:
            return json.dumps(
                {
                    "error": (f"Not your turn. Current speaker: {current_speaker}"),
                    "your_name": agent_name,
                }
            )

        transcript = _storage.get_transcript(meeting_id, _conn=conn)

        turn_entry = {
            "turn": len(transcript) + 1,
            "round": state.get("current_round", 1),
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": message,
            "type": "speak",
        }
        _storage.append_transcript(meeting_id, turn_entry, _conn=conn)

        # Defer transcript hook
        deferred_hooks.append(("on_transcript_entry", meeting_id, turn_entry))

        # Check for verdict — requires [DECISION] prefix to avoid
        # false positives when agents mention verdicts in discussion.
        verdict_match = re.search(
            r"\[DECISION\]\s*VERDICT:\s*(PASS|FAIL|ESCALATE"
            r"|ESCALATE_TO_USER|RESOLVED)",
            message,
            re.IGNORECASE,
        )

        if verdict_match:
            verdict = verdict_match.group(1).lower()

            if verdict == "fail":
                # FAIL does NOT end the meeting — continues
                remaining = max_rounds - state.get("current_round", 1)
                if remaining < 3:
                    new_max = max_rounds + 3
                    state["max_rounds"] = new_max
                    max_rounds = new_max
                    # state is written below via update_state(_conn=conn) — single write path.
                    _audit(
                        meeting_id,
                        f"Extended max_rounds to {new_max} after FAIL verdict",
                    )

                action_items: list[dict[str, Any]] = state.get("action_items", [])
                reason_match = re.search(
                    r"\[DECISION\]\s*VERDICT:\s*FAIL\s*[-—:.]\s*(.+)",
                    message,
                    re.IGNORECASE,
                )
                action_items.append(
                    {
                        "from": agent_name,
                        "round": state.get("current_round", 1),
                        "reason": (
                            reason_match.group(1).strip()[:500] if reason_match else "Issues found"
                        ),
                    }
                )
                state["action_items"] = action_items
                state["last_fail_round"] = state.get("current_round", 1)

                next_turn = current_turn + 1
                current_round = state.get("current_round", 1)
                if next_turn >= len(participants):
                    next_turn = 0
                    current_round += 1
                state["current_turn"] = next_turn
                state["current_round"] = current_round
                _storage.update_state(meeting_id, state, _conn=conn)

                next_speaker = (
                    participants[next_turn] if next_turn < len(participants) else "unknown"
                )
                _audit(
                    meeting_id,
                    f"{agent_name} [round {turn_entry['round']}]: VERDICT FAIL — meeting continues",
                )

                deferred_hooks.append(("on_verdict", meeting_id, verdict, agent_name))
                deferred_hooks.append(("on_turn_advanced", meeting_id, next_speaker, current_round))
                deferred_hooks.append(("on_state_changed", meeting_id, state))

                result_json = json.dumps(
                    {
                        "status": "spoken",
                        "meeting_ended": False,
                        "verdict": "fail",
                        "action": "meeting_continues",
                        "message": ("FAIL recorded. Meeting continues."),
                        "turn": turn_entry["turn"],
                        "next_speaker": next_speaker,
                    }
                )
            else:
                # PASS, RESOLVED, ESCALATE, ESCALATE_TO_USER → end
                state["ended"] = True
                state["outcome"] = f"verdict_{verdict}"
                _storage.update_state(meeting_id, state, _conn=conn)
                _audit(
                    meeting_id,
                    f"{agent_name} [round {state.get('current_round', 1)}]: {message[:200]}",
                )
                _audit(
                    meeting_id,
                    f"Meeting ended: verdict_{verdict}",
                )

                deferred_hooks.append(("on_verdict", meeting_id, verdict, agent_name))
                deferred_hooks.append(("on_meeting_ended", meeting_id, f"verdict_{verdict}"))
                deferred_hooks.append(("on_state_changed", meeting_id, state))

                result_json = json.dumps(
                    {
                        "status": "spoken",
                        "meeting_ended": True,
                        "verdict": verdict,
                        "turn": turn_entry["turn"],
                    }
                )
        else:
            # Advance turn normally
            next_turn = current_turn + 1
            current_round = state.get("current_round", 1)

            if next_turn >= len(participants):
                next_turn = 0
                current_round += 1
                if current_round > max_rounds:
                    state["ended"] = True
                    state["outcome"] = "max_rounds_reached"
                    state["current_turn"] = next_turn
                    state["current_round"] = current_round
                    _storage.update_state(meeting_id, state, _conn=conn)
                    _audit(meeting_id, f"{agent_name}: {message[:200]}")
                    _audit(
                        meeting_id,
                        "Meeting ended: max_rounds_reached",
                    )

                    deferred_hooks.append(("on_meeting_ended", meeting_id, "max_rounds_reached"))
                    deferred_hooks.append(("on_state_changed", meeting_id, state))

                    result_json = json.dumps(
                        {
                            "status": "spoken",
                            "meeting_ended": True,
                            "reason": "max_rounds_reached",
                            "turn": turn_entry["turn"],
                        }
                    )

            if result_json is None:
                state["current_turn"] = next_turn
                state["current_round"] = current_round
                state["turn_started_at"] = time.time()
                _storage.update_state(meeting_id, state, _conn=conn)

                next_speaker = participants[next_turn]

    # ── Fire deferred hooks OUTSIDE the lock ──
    for hook_args in deferred_hooks:
        _fire_hook(*hook_args)

    # ── Send meeting-ended notifications if meeting just ended ──
    if result_json is not None and state.get("ended"):
        agenda = config.get("agenda", "")
        for participant in participants:
            _notify_meeting_ended(meeting_id, participant, agenda)

    # Early-return results (verdict / max_rounds)
    if result_json is not None:
        return result_json

    _audit(
        meeting_id,
        f"{agent_name} [round {turn_entry['round']}]: {message[:200]}",
    )
    _audit(
        meeting_id,
        f"Turn advanced → {next_speaker} (round {current_round})",
    )

    _fire_hook("on_turn_advanced", meeting_id, next_speaker, current_round)
    _fire_hook("on_state_changed", meeting_id, state)

    # Wake next speaker — ensures they know it's their turn
    _notify_turn_agent(
        meeting_id, next_speaker, config.get("agenda", ""), current_round
    )

    return json.dumps(
        {
            "status": "spoken",
            "meeting_ended": False,
            "turn": turn_entry["turn"],
            "next_speaker": next_speaker,
        }
    )


@mcp.tool()
async def skip_turn(meeting_id: str, agent_name: str = "", reason: str = "") -> str:
    """Skip your turn. The turn advances to the next participant.

    Args:
        meeting_id: The meeting.
        agent_name: YOUR agent name. Auto-detected if empty.
            ⚠️ Cannot be used to skip another agent's turn — if you pass
            a name that disagrees with your spawn-pinned identity, the
            call is refused. Use ``end_meeting`` if a teammate stalls.
        reason: Optional reason for skipping.

    Returns:
        JSON with confirmation.
    """
    agent_name, _err = _assert_self_identity(agent_name)
    if _err:
        return _err

    if not _storage.meeting_exists(meeting_id):
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _storage.get_config(meeting_id) or {}

    deferred_hooks: list[tuple] = []
    result_json: str | None = None
    turn_entry: dict = {}
    next_speaker: str = ""
    current_round: int = 1
    participants: list[str] = []  # captured in lock for outside-lock notifications

    with _storage.acquire_lock(meeting_id) as conn:
        state = _storage.get_state(meeting_id, _conn=conn) or {}

        if state.get("ended"):
            return json.dumps({"error": "Meeting already ended"})

        participants = state.get("participants", [])
        max_rounds = state.get("max_rounds", 3)

        current_turn = state.get("current_turn", 0)
        current_speaker = participants[current_turn] if current_turn < len(participants) else None
        if agent_name != current_speaker:
            return json.dumps(
                {
                    "error": (f"Not your turn. Current speaker: {current_speaker}"),
                    "your_name": agent_name,
                }
            )

        turn_entry = {
            "turn": len(_storage.get_transcript(meeting_id, _conn=conn)) + 1,
            "round": state.get("current_round", 1),
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": reason or "(skipped)",
            "type": "skip",
        }
        _storage.append_transcript(meeting_id, turn_entry, _conn=conn)

        deferred_hooks.append(("on_transcript_entry", meeting_id, turn_entry))

        next_turn = current_turn + 1
        current_round = state.get("current_round", 1)

        if next_turn >= len(participants):
            next_turn = 0
            current_round += 1
            if current_round > max_rounds:
                state["ended"] = True
                state["outcome"] = "max_rounds_reached"
                state["current_turn"] = next_turn
                state["current_round"] = current_round
                _storage.update_state(meeting_id, state, _conn=conn)
                _audit(
                    meeting_id,
                    f"{agent_name} skipped: {reason}",
                )
                _audit(
                    meeting_id,
                    "Meeting ended: max_rounds_reached",
                )

                deferred_hooks.append(("on_meeting_ended", meeting_id, "max_rounds_reached"))
                deferred_hooks.append(("on_state_changed", meeting_id, state))

                result_json = json.dumps(
                    {
                        "status": "skipped",
                        "meeting_ended": True,
                        "reason": "max_rounds_reached",
                        "turn": turn_entry["turn"],
                    }
                )

        if result_json is None:
            state["current_turn"] = next_turn
            state["current_round"] = current_round
            state["turn_started_at"] = time.time()
            _storage.update_state(meeting_id, state, _conn=conn)

            next_speaker = participants[next_turn]

    # ── Fire deferred hooks OUTSIDE the lock ──
    for hook_args in deferred_hooks:
        _fire_hook(*hook_args)

    # ── Send meeting-ended notifications if meeting just ended ──
    if result_json is not None and state.get("ended"):
        agenda = config.get("agenda", "")
        for participant in participants:
            _notify_meeting_ended(meeting_id, participant, agenda)

    if result_json is not None:
        return result_json

    _audit(
        meeting_id,
        f"{agent_name} skipped turn: {reason or 'no reason'}",
    )

    _fire_hook("on_turn_advanced", meeting_id, next_speaker, current_round)
    _fire_hook("on_state_changed", meeting_id, state)

    # Wake next speaker
    _notify_turn_agent(
        meeting_id, next_speaker, config.get("agenda", ""), current_round
    )

    return json.dumps(
        {
            "status": "skipped",
            "meeting_ended": False,
            "turn": turn_entry["turn"],
            "next_speaker": next_speaker,
        }
    )


@mcp.tool()
async def get_meeting_status(meeting_id: str) -> str:
    """Get the current status of a meeting.

    Args:
        meeting_id: The meeting to check.

    Returns:
        JSON with round, turn, participants, ended flag.
    """
    if not _storage.meeting_exists(meeting_id):
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _storage.get_config(meeting_id) or {}
    state = _storage.get_state(meeting_id) or {}
    participants: list[str] = state.get("participants", [])
    current_turn = state.get("current_turn", 0)
    transcript = _storage.get_transcript(meeting_id)

    current_speaker = None
    if not state.get("ended") and current_turn < len(participants):
        current_speaker = participants[current_turn]

    return json.dumps(
        {
            "meeting_id": meeting_id,
            "agenda": config.get("agenda", ""),
            "ended": state.get("ended", False),
            "outcome": state.get("outcome"),
            "started": state.get("started", False),
            "current_round": state.get("current_round", 1),
            "max_rounds": state.get("max_rounds", 3),
            "current_speaker": current_speaker,
            "participants": participants,
            "joined": state.get("joined", []),
            "transcript_length": len(transcript),
        }
    )


@mcp.tool()
async def add_participant(meeting_id: str, agent_name: str) -> str:
    """Add a new participant to an ongoing meeting (for escalation).

    Also sends them a meeting invite via post_message and wakes them if idle.

    Args:
        meeting_id: The meeting.
        agent_name: Agent name to add (e.g. "Agent B - QE").

    Returns:
        JSON with updated participant list.
    """
    if not _storage.meeting_exists(meeting_id):
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _storage.get_config(meeting_id) or {}

    # Mutate participants and auto-join in a SINGLE lock pass — both fields
    # live in state_json now so one update_state writes everything.
    with _storage.acquire_lock(meeting_id) as conn:
        state = _storage.get_state(meeting_id, _conn=conn) or {}

        if state.get("ended"):
            return json.dumps({"error": ("Cannot add participant — meeting already ended")})

        participants: list[str] = state.get("participants", [])
        if agent_name in participants:
            return json.dumps({"status": "already_participant", "agent_name": agent_name})

        participants.append(agent_name)
        state["participants"] = participants

        joined = state.get("joined", [])
        if agent_name not in joined:
            joined.append(agent_name)
            state["joined"] = joined

        _storage.update_state(meeting_id, state, _conn=conn)

    bus = _get_bus()
    if bus:
        agenda = config.get("agenda", "")
        invite_content = (
            f"📋 You've been added to Meeting [{meeting_id}] (mid-meeting).\n"
            f"Agenda: {agenda}\n"
            f"Participants: {', '.join(participants)}\n\n"
            f"You will receive a 🎙️ YOUR TURN notification when it's your turn to speak."
        )
        bus.send(
            from_name=_get_my_name(),
            to_name=agent_name,
            content=invite_content,
            message_type="meeting_invite",
        )
        _auto_wake_if_idle(agent_name)

    _audit(
        meeting_id,
        f"Participant added mid-meeting: {agent_name}",
    )

    _fire_hook("on_participant_added", meeting_id, agent_name)

    return json.dumps(
        {
            "status": "added",
            "agent_name": agent_name,
            "participants": participants,
        }
    )


@mcp.tool()
async def leave_meeting(meeting_id: str, agent_name: str = "", reason: str = "") -> str:
    """Leave a meeting. Your turns will be skipped.

    🛑 **You CANNOT leave when it's currently your turn**. Meetings are
    synchronous turn-based — bailing mid-turn strands the next speaker
    and is the failure pattern that hung incident b61af7db. If you're
    the current speaker, choose one instead:

    * ``speak(...)`` — say what you intended to do then advance.
    * ``speak("[DECISION] VERDICT: PASS — ...")`` — close the meeting if
      you're the chair and discussion is done.
    * ``skip_turn(...)`` — advance to the next speaker without speaking.

    Once the turn is no longer yours, ``leave_meeting`` is allowed.

    Args:
        meeting_id: The meeting to leave.
        agent_name: YOUR agent name. Auto-detected if empty.
        reason: Why you're leaving.

    Returns:
        JSON confirming you've left, or an error with ``next_action``
        guidance if you tried to leave on your own turn.
    """
    agent_name = agent_name or _get_my_name()

    if not _storage.meeting_exists(meeting_id):
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    participants: list[str] = []

    with _storage.acquire_lock(meeting_id) as conn:
        state = _storage.get_state(meeting_id, _conn=conn) or {}

        if state.get("ended"):
            return json.dumps({"error": "Meeting already ended"})

        participants = state.get("participants", [])
        if agent_name not in participants:
            return json.dumps({"error": f"'{agent_name}' not in meeting"})

        current_turn = state.get("current_turn", 0)
        current_speaker = (
            participants[current_turn] if current_turn < len(participants) else None
        )
        # R3 guard: refuse to leave mid-turn. The b61af7db incident hung
        # for 24h because BA's LLM tried `leave_meeting` instead of
        # `speak()` to "go work on Jira". Force the LLM back into the
        # protocol — speak / verdict / skip — so the next speaker is
        # always cued correctly.
        if current_speaker == agent_name:
            return json.dumps({
                "error": (
                    "Cannot leave_meeting while it's your turn. Other "
                    "participants are blocked waiting for you to act."
                ),
                "current_speaker": agent_name,
                "round": state.get("current_round", 1),
                "max_rounds": state.get("max_rounds", 0),
                "next_action": "Use speak(...) to contribute, "
                               "speak('[DECISION] VERDICT: PASS — ...') "
                               "to close the meeting, or skip_turn(...) "
                               "to pass without speaking. Then "
                               "leave_meeting is allowed.",
            })

        my_index = participants.index(agent_name)
        participants.remove(agent_name)
        state["participants"] = participants

        if my_index < current_turn:
            state["current_turn"] = current_turn - 1
        elif my_index == current_turn:
            if len(participants) > 0:
                state["current_turn"] = current_turn % len(participants)
            else:
                state["ended"] = True
                state["outcome"] = "all_left"

        leave_entry = {
            "turn": len(_storage.get_transcript(meeting_id, _conn=conn)) + 1,
            "round": state.get("current_round", 1),
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": (f"Left the meeting: {reason}" if reason else "Left the meeting"),
            "type": "leave",
        }
        _storage.append_transcript(meeting_id, leave_entry, _conn=conn)
        _storage.update_state(meeting_id, state, _conn=conn)

    _audit(
        meeting_id,
        f"{agent_name} left the meeting: {reason or 'no reason given'}",
    )

    _fire_hook("on_transcript_entry", meeting_id, leave_entry)
    _fire_hook("on_participant_left", meeting_id, agent_name)
    _fire_hook("on_state_changed", meeting_id, state)

    if state.get("ended"):
        _fire_hook("on_meeting_ended", meeting_id, "all_left")

    return json.dumps(
        {
            "status": "left",
            "agent_name": agent_name,
            "remaining_participants": participants,
        }
    )


# ───────────────────────────────────────────────────────────────────
# Shared helpers — imported from _team_helpers (shared with email server)
# ───────────────────────────────────────────────────────────────────

from fast_agent.spawn.servers._team_helpers import (
    auto_wake_if_idle as _auto_wake_if_idle,
    get_bus as _get_bus,
    get_my_name as _get_my_name,
    get_team_config as _get_team_config,
    parse_recipients as _parse_recipients,
)


if __name__ == "__main__":
    import os

    db_path = os.environ.get("JARVIS_DB_PATH", "")
    if db_path:
        from fast_agent.spawn.servers.meeting_storage import SqliteMeetingStorage
        from fast_agent.spawn.servers.meeting_hooks import MeetingHooks

        storage = SqliteMeetingStorage(db_path=db_path)

        def _emit(event_type, meeting_id, data):
            try:
                storage.emit_event(event_type, meeting_id, data)
            except Exception:
                pass  # Never let hook errors crash the MCP server

        hooks = MeetingHooks(
            on_meeting_created=lambda mid, cfg: _emit(
                "meeting_created", mid, {"config": cfg}
            ),
            on_participant_joined=lambda mid, name, all_j: _emit(
                "participant_joined", mid,
                {"agent_name": name, "all_joined": all_j},
            ),
            on_meeting_started=lambda mid: _emit(
                "meeting_started", mid, {}
            ),
            on_transcript_entry=lambda mid, entry: _emit(
                "transcript_entry", mid, {"entry": entry}
            ),
            on_turn_advanced=lambda mid, spk, rnd: _emit(
                "turn_advanced", mid,
                {"next_speaker": spk, "round": rnd},
            ),
            on_verdict=lambda mid, v, agent: _emit(
                "verdict", mid, {"verdict": v, "by_agent": agent}
            ),
            on_meeting_ended=lambda mid, outcome: _emit(
                "meeting_ended", mid, {"outcome": outcome}
            ),
            on_state_changed=lambda mid, state: _emit(
                "state_changed", mid, {"state": state}
            ),
        )
        configure_meeting_room(storage=storage, hooks=hooks)

    mcp.run()
