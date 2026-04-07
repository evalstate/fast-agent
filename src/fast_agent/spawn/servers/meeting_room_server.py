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
    """Send YOUR_TURN inbox notification + wake signal to the current speaker.

    This ensures agents are notified even if their wait_for_my_turn poll
    has already timed out.
    """
    bus = _get_bus()
    if bus:
        msg = (
            f"🎙️ YOUR TURN TO SPEAK\n"
            f"Meeting: {meeting_id}\n"
            f"Agenda: {agenda}\n"
            f"Round: {round_num}\n\n"
            f'→ get_transcript(meeting_id="{meeting_id}")\n'
            f'→ speak(meeting_id="{meeting_id}", message="...")'
        )
        bus.send(
            from_name="Meeting Room",
            to_name=agent_name,
            content=msg,
            message_type="meeting_turn",
        )
    _auto_wake_if_idle(agent_name)
    logger.info("🎙️ Notified %s: your turn in %s (round %d)", agent_name, meeting_id, round_num)


# ───────────────────────────────────────────────────────────────────
# MCP Tools
# ───────────────────────────────────────────────────────────────────


@mcp.tool()
async def create_meeting(
    agenda: str,
    participants: str,
    max_rounds: int = 3,
    my_name: str = "",
    workspace_dir: str = "",
) -> str:
    """Create a meeting room and auto-invite all participants.

    🎙️ Meetings are for REAL-TIME discussion — faster than post_message.
    All participants receive a 🔔 MEETING INVITE and are auto-woken if idle.

    Use meetings for: kickoff, design review, code review, blocker resolution.
    Use post_message for: task assignments, status updates, async notifications.

    Args:
        agenda: Short title for this meeting (max 120 chars). Displayed as the
                meeting title in the dashboard. Keep it concise, e.g.
                "Sprint 1 kickoff" or "Blocker review cho SCRUM-5".
                Do NOT put detailed discussion points here.
        participants: Comma-separated AGENT NAMES in speaking order.
                      E.g. "Linh - PM, Hoa - BA, Khang - SA"
        max_rounds: Maximum conversation rounds (default 3).
        my_name: YOUR agent name (auto-detected from env).
        workspace_dir: Workspace path (auto-detected from env).

    Returns:
        JSON with meeting_id and config.
    """
    if not workspace_dir:
        workspace_dir = os.environ.get(
            "TEAM_WORKSPACE",
            str(Path.cwd() / ".runtime" / "cache" / "tmp" / "meeting-workspace"),
        )
    my_name = my_name or _get_my_name()
    participant_list = [p.strip() for p in participants.split(",") if p.strip()]
    if len(participant_list) < 2:
        return json.dumps({"error": "Need at least 2 participants"})

    meeting_id = f"mtg_{uuid.uuid4().hex[:8]}"

    config = {
        "meeting_id": meeting_id,
        "agenda": agenda,
        "participants": participant_list,
        "max_rounds": max_rounds,
        "created_by": my_name,
        "created_at": datetime.now().isoformat(),
    }

    state: dict[str, Any] = {
        "current_turn": 0,
        "current_round": 1,
        "joined": [my_name],  # Auto-join the creator
        "ended": False,
        "outcome": None,
        "started": False,
    }

    _storage.create_meeting(meeting_id, config, state)

    _audit(
        meeting_id,
        f"Meeting created by {my_name}: agenda='{agenda}', "
        f"participants={participant_list}, max_rounds={max_rounds}",
    )

    _fire_hook("on_meeting_created", meeting_id, config)
    _fire_hook("on_state_changed", meeting_id, state)

    # ── Auto-invite all participants ──────────────────────────────
    bus = _get_bus()
    invited: list[str] = []
    if bus:
        participant_names = ", ".join(participant_list)
        for participant in participant_list:
            if participant == my_name:
                continue  # Don't invite self
            invite_content = (
                f"🔔 MEETING INVITE\n"
                f"Agenda: {agenda}\n"
                f"Meeting ID: {meeting_id}\n"
                f"Participants: {participant_names}\n"
                f"Created by: {my_name}\n\n"
                f'→ To join: join_meeting(meeting_id="{meeting_id}", '
                f'agent_name="{participant}")\n'
                f"You will receive a 🎙️ YOUR TURN notification when it's your turn to speak."
            )
            bus.send(
                from_name=my_name,
                to_name=participant,
                content=invite_content,
                message_type="meeting_invite",
            )
            _auto_wake_if_idle(participant)
            invited.append(participant)
            logger.info("📨 Sent meeting invite to %s for %s", participant, meeting_id)

    logger.info("Meeting created: %s — %s (invited %d)", meeting_id, agenda, len(invited))
    return json.dumps(
        {
            "meeting_id": meeting_id,
            "agenda": agenda,
            "participants": participant_list,
            "max_rounds": max_rounds,
            "status": "invites_sent",
            "invited": invited,
        }
    )


@mcp.tool()
async def join_meeting(meeting_id: str, agent_name: str = "") -> str:
    """Join an existing meeting. Call after receiving a 🔔 MEETING INVITE.

    When all expected participants have joined, the meeting starts.

    Args:
        meeting_id: The meeting to join.
        agent_name: YOUR agent name (e.g. "Hoa - BA"). Auto-detected if empty.

    Returns:
        JSON with meeting config and join status.
    """
    agent_name = agent_name or _get_my_name()

    if not _storage.meeting_exists(meeting_id):
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _storage.get_config(meeting_id) or {}

    with _storage.acquire_lock(meeting_id) as conn:
        state = _storage.get_state(meeting_id, _conn=conn) or {}

        if state.get("ended"):
            return json.dumps(
                {
                    "error": "Meeting already ended",
                    "meeting_id": meeting_id,
                }
            )

        joined: list[str] = state.get("joined", [])
        if agent_name not in joined:
            joined.append(agent_name)
            state["joined"] = joined

        participants: list[str] = config.get("participants", [])
        if agent_name not in participants:
            participants.append(agent_name)
            config["participants"] = participants
            _storage.update_config(meeting_id, config)

        all_joined = all(p in joined for p in participants)
        if all_joined:
            state["started"] = True
            state["turn_started_at"] = time.time()

        _storage.update_state(meeting_id, state, _conn=conn)

    _audit(meeting_id, f"{agent_name} joined the meeting")
    _fire_hook("on_participant_joined", meeting_id, agent_name, all_joined)
    _fire_hook("on_state_changed", meeting_id, state)

    if all_joined:
        first_speaker = participants[0]
        _audit(
            meeting_id,
            f"All participants joined. Meeting started. First turn: {first_speaker}",
        )
        _fire_hook("on_meeting_started", meeting_id)
        # Wake first speaker — even if their wait_for_my_turn already timed out
        _notify_turn_agent(
            meeting_id, first_speaker, config.get("agenda", ""), 1
        )

    return json.dumps(
        {
            "status": "joined",
            "meeting_id": meeting_id,
            "agenda": config.get("agenda", ""),
            "participants": config.get("participants", []),
            "max_rounds": config.get("max_rounds", 3),
            "all_joined": all_joined,
            "your_name": agent_name,
            "note": "You will receive a 🎙️ YOUR TURN inbox notification when it's your turn. Continue other work.",
        }
    )


@mcp.tool()
async def wait_for_my_turn(meeting_id: str, agent_name: str = "") -> str:
    """Check if it's your turn to speak in the meeting.

    This is an INSTANT check — no blocking or polling.
    You will also receive a 🎙️ YOUR TURN inbox notification
    automatically when it's your turn, so you don't need to
    call this repeatedly.

    Args:
        meeting_id: The meeting to check.
        agent_name: YOUR agent name. Auto-detected if empty.

    Returns:
        JSON with status: "your_turn", "not_your_turn", or "meeting_ended".
    """
    agent_name = agent_name or _get_my_name()

    if not _storage.meeting_exists(meeting_id):
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _storage.get_config(meeting_id) or {}
    participants: list[str] = config.get("participants", [])

    if agent_name not in participants:
        return json.dumps({"error": (f"'{agent_name}' not in meeting participants: {participants}")})

    my_index = participants.index(agent_name)
    state = _storage.get_state(meeting_id) or {}

    # Meeting ended?
    if state.get("ended"):
        transcript = _storage.get_transcript(meeting_id)
        cursors: dict[str, int] = state.get("read_cursors", {})
        last_read = cursors.get(agent_name, 0)
        unread = transcript[last_read:]
        return json.dumps(
            {
                "status": "meeting_ended",
                "outcome": state.get("outcome"),
                "meeting_id": meeting_id,
                "unread_messages": unread,
                "unread_count": len(unread),
            },
            ensure_ascii=False,
        )

    # Not started yet?
    if not state.get("started"):
        joined = state.get("joined", [])
        return json.dumps(
            {
                "status": "waiting_for_participants",
                "meeting_id": meeting_id,
                "joined": joined,
                "joined_count": len(joined),
                "total_participants": len(participants),
                "note": "You will receive a 🎙️ YOUR TURN notification when meeting starts and it's your turn.",
            }
        )

    # Is it my turn?
    current_turn = state.get("current_turn", 0)
    if current_turn == my_index:
        return json.dumps(
            {
                "status": "your_turn",
                "round": state.get("current_round", 1),
                "turn": current_turn,
                "meeting_id": meeting_id,
            }
        )

    # Not my turn
    current_speaker = (
        participants[current_turn]
        if current_turn < len(participants)
        else "unknown"
    )
    return json.dumps(
        {
            "status": "not_your_turn",
            "meeting_id": meeting_id,
            "current_speaker": current_speaker,
            "round": state.get("current_round", 1),
            "note": "You will receive a 🎙️ YOUR TURN notification when it's your turn.",
        }
    )


@mcp.tool()
async def get_transcript(meeting_id: str, agent_name: str = "") -> str:
    """Get NEW (unread) transcript messages since your last read.

    Each agent has a read cursor. First call returns the full transcript;
    subsequent calls return only messages you haven't seen yet.
    No messages are ever skipped or missed.

    Args:
        meeting_id: The meeting to get transcript for.
        agent_name: YOUR agent name. Auto-detected if empty.

    Returns:
        JSON with new_messages (unread) and total_count.
    """
    agent_name = agent_name or _get_my_name()

    if not _storage.meeting_exists(meeting_id):
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _storage.get_config(meeting_id) or {}
    transcript = _storage.get_transcript(meeting_id)

    with _storage.acquire_lock(meeting_id) as conn:
        state = _storage.get_state(meeting_id, _conn=conn) or {}

        cursors: dict[str, int] = state.get("read_cursors", {})
        last_read = cursors.get(agent_name, 0)
        new_messages = transcript[last_read:]

        # Advance cursor — agent has now "seen" everything up to this point
        cursors[agent_name] = len(transcript)
        state["read_cursors"] = cursors
        _storage.update_state(meeting_id, state, _conn=conn)

    return json.dumps(
        {
            "meeting_id": meeting_id,
            "agenda": config.get("agenda", ""),
            "current_round": state.get("current_round", 1),
            "new_messages": new_messages,
            "new_count": len(new_messages),
            "total_count": len(transcript),
        },
        ensure_ascii=False,
    )


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
        message: What you want to say.

    Returns:
        JSON with confirmation and next turn info.
    """
    agent_name = agent_name or _get_my_name()

    if not _storage.meeting_exists(meeting_id):
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _storage.get_config(meeting_id) or {}
    participants: list[str] = config.get("participants", [])
    max_rounds = config.get("max_rounds", 3)

    # Collect hooks to fire AFTER releasing the lock (emit_event opens a
    # new connection which would deadlock against BEGIN IMMEDIATE).
    deferred_hooks: list[tuple] = []
    result_json: str | None = None
    turn_entry: dict = {}
    next_speaker: str = ""
    current_round: int = 1

    with _storage.acquire_lock(meeting_id) as conn:
        state = _storage.get_state(meeting_id, _conn=conn) or {}

        if state.get("ended"):
            return json.dumps({"error": "Meeting already ended"})

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
                    config["max_rounds"] = new_max
                    _storage.update_config(meeting_id, config)
                    max_rounds = new_max
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
        reason: Optional reason for skipping.

    Returns:
        JSON with confirmation.
    """
    agent_name = agent_name or _get_my_name()

    if not _storage.meeting_exists(meeting_id):
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _storage.get_config(meeting_id) or {}
    participants: list[str] = config.get("participants", [])
    max_rounds = config.get("max_rounds", 3)

    deferred_hooks: list[tuple] = []
    result_json: str | None = None
    turn_entry: dict = {}
    next_speaker: str = ""
    current_round: int = 1

    with _storage.acquire_lock(meeting_id) as conn:
        state = _storage.get_state(meeting_id, _conn=conn) or {}

        if state.get("ended"):
            return json.dumps({"error": "Meeting already ended"})

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
    participants: list[str] = config.get("participants", [])
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
            "max_rounds": config.get("max_rounds", 3),
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
        agent_name: Agent name to add (e.g. "Tuan - QE").

    Returns:
        JSON with updated participant list.
    """
    if not _storage.meeting_exists(meeting_id):
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    with _storage.acquire_lock(meeting_id) as conn:
        config = _storage.get_config(meeting_id) or {}
        state = _storage.get_state(meeting_id, _conn=conn) or {}

        if state.get("ended"):
            return json.dumps({"error": ("Cannot add participant — meeting already ended")})

        participants: list[str] = config.get("participants", [])
        if agent_name in participants:
            return json.dumps({"status": "already_participant", "agent_name": agent_name})

        participants.append(agent_name)
        config["participants"] = participants
        _storage.update_config(meeting_id, config)

    # Auto-invite the new participant
    bus = _get_bus()
    if bus:
        agenda = config.get("agenda", "")
        invite_content = (
            f"🔔 MEETING INVITE (mid-meeting)\n"
            f"Agenda: {agenda}\n"
            f"Meeting ID: {meeting_id}\n"
            f"You were added to an ongoing meeting.\n\n"
            f'→ join_meeting(meeting_id="{meeting_id}", '
            f'agent_name="{agent_name}")\n'
            f'→ wait_for_my_turn(meeting_id="{meeting_id}", '
            f'agent_name="{agent_name}")'
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

    Args:
        meeting_id: The meeting to leave.
        agent_name: YOUR agent name. Auto-detected if empty.
        reason: Why you're leaving.

    Returns:
        JSON confirming you've left.
    """
    agent_name = agent_name or _get_my_name()

    if not _storage.meeting_exists(meeting_id):
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _storage.get_config(meeting_id) or {}
    participants: list[str] = config.get("participants", [])

    with _storage.acquire_lock(meeting_id) as conn:
        state = _storage.get_state(meeting_id, _conn=conn) or {}

        if state.get("ended"):
            return json.dumps({"error": "Meeting already ended"})

        if agent_name not in participants:
            return json.dumps({"error": f"'{agent_name}' not in meeting"})

        current_turn = state.get("current_turn", 0)
        my_index = participants.index(agent_name)
        participants.remove(agent_name)
        config["participants"] = participants
        _storage.update_config(meeting_id, config)

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
