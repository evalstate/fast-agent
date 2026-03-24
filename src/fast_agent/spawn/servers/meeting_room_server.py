"""Meeting Room MCP Server — multi-agent meetings with shared conversation.

Architecture: Each agent subprocess gets its OWN instance of this server.
Therefore all state is FILE-BASED (shared workspace directory). Turn
coordination uses file polling instead of ``asyncio.Event``, since agents
run as separate processes.

Storage layout::

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
import fcntl
import json
import logging
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("meeting-room")


# ───────────────────────────────────────────────────────────────────
# File-based state helpers
# ───────────────────────────────────────────────────────────────────


def _get_meeting_dir(meeting_id: str) -> Path:
    """Get meeting directory from env-based workspace."""
    workspace = os.environ.get("TEAM_WORKSPACE", "")
    if not workspace:
        # Fallback: use .runtime/ under current working directory
        workspace = str(Path.cwd() / ".runtime" / "cache" / "tmp" / "meeting-workspace")
    return Path(workspace) / "meetings" / meeting_id


@contextmanager
def _file_lock(meeting_dir: Path):  # type: ignore[no-untyped-def]
    """Acquire exclusive file lock on the meeting directory."""
    lock_path = meeting_dir / ".lock"
    lock_path.touch(exist_ok=True)
    fd = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    """Read a JSON file, return empty dict if not found."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    """Write data as JSON."""
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _audit(meeting_dir: Path, message: str) -> None:
    """Append to the audit log."""
    audit_path = meeting_dir / "audit.log"
    ts = datetime.now().isoformat(timespec="seconds")
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


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
        agenda: What this meeting is about.
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
    meeting_dir = Path(workspace_dir) / "meetings" / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "meeting_id": meeting_id,
        "agenda": agenda,
        "participants": participant_list,
        "max_rounds": max_rounds,
        "created_by": my_name,
        "created_at": datetime.now().isoformat(),
    }
    _write_json(meeting_dir / "config.json", config)

    state: dict[str, Any] = {
        "current_turn": 0,
        "current_round": 1,
        "joined": [my_name],  # Auto-join the creator
        "ended": False,
        "outcome": None,
        "started": False,
    }
    _write_json(meeting_dir / "state.json", state)
    _write_json(meeting_dir / "transcript.json", [])

    _audit(
        meeting_dir,
        f"Meeting created by {my_name}: agenda='{agenda}', "
        f"participants={participant_list}, max_rounds={max_rounds}",
    )

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
                f"→ To join: join_meeting(meeting_id=\"{meeting_id}\", "
                f"agent_name=\"{participant}\")\n"
                f"→ Then: wait_for_my_turn(meeting_id=\"{meeting_id}\", "
                f"agent_name=\"{participant}\")"
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
    meeting_dir = _get_meeting_dir(meeting_id)

    if not meeting_dir.exists():
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _read_json(meeting_dir / "config.json")
    if not isinstance(config, dict):
        config = {}

    with _file_lock(meeting_dir):
        state = _read_json(meeting_dir / "state.json")
        if not isinstance(state, dict):
            state = {}

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
            _write_json(meeting_dir / "config.json", config)

        all_joined = all(p in joined for p in participants)
        if all_joined:
            state["started"] = True

        _write_json(meeting_dir / "state.json", state)

    _audit(meeting_dir, f"{agent_name} joined the meeting")
    if all_joined:
        _audit(
            meeting_dir,
            f"All participants joined. Meeting started. First turn: {participants[0]}",
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
        }
    )


@mcp.tool()
async def wait_for_my_turn(meeting_id: str, agent_name: str = "", timeout_seconds: int = 180) -> str:
    """Wait (poll) until it's your turn to speak or the meeting ends.

    Call after ``join_meeting``. When this returns ``your_turn``, call
    ``get_transcript`` then ``speak`` or ``skip_turn``.

    Args:
        meeting_id: The meeting to wait in.
        agent_name: YOUR agent name. Auto-detected if empty.
        timeout_seconds: Max seconds to wait (default 180).

    Returns:
        JSON with status: "your_turn", "meeting_ended", or "timeout".
    """
    agent_name = agent_name or _get_my_name()
    meeting_dir = _get_meeting_dir(meeting_id)
    if not meeting_dir.exists():
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _read_json(meeting_dir / "config.json")
    if not isinstance(config, dict):
        config = {}
    participants: list[str] = config.get("participants", [])

    if agent_name not in participants:
        return json.dumps({"error": (f"'{agent_name}' not in meeting participants: {participants}")})

    my_index = participants.index(agent_name)
    start_time = time.time()
    poll_interval = 2

    while True:
        state = _read_json(meeting_dir / "state.json")
        if not isinstance(state, dict):
            state = {}

        if state.get("ended"):
            # Include unread transcript so agent doesn't miss anything
            transcript = _read_json(meeting_dir / "transcript.json")
            if not isinstance(transcript, list):
                transcript = []
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

        if not state.get("started"):
            if time.time() - start_time > timeout_seconds:
                joined = state.get("joined", [])
                return json.dumps(
                    {
                        "status": "timeout",
                        "reason": "waiting_for_participants",
                        "meeting_state": {
                            "started": False,
                            "participants": participants,
                            "joined": joined,
                            "joined_count": len(joined),
                            "total_participants": len(participants),
                            "current_turn": None,
                            "current_round": None,
                        },
                    }
                )
            await asyncio.sleep(poll_interval)
            continue

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

        if time.time() - start_time > timeout_seconds:
            joined = state.get("joined", [])
            return json.dumps(
                {
                    "status": "timeout",
                    "reason": "waited_too_long",
                    "meeting_state": {
                        "started": state.get("started", False),
                        "participants": participants,
                        "joined": joined,
                        "joined_count": len(joined),
                        "total_participants": len(participants),
                        "current_turn": state.get("current_turn"),
                        "current_round": state.get("current_round"),
                    },
                }
            )

        await asyncio.sleep(poll_interval)


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
    meeting_dir = _get_meeting_dir(meeting_id)
    if not meeting_dir.exists():
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _read_json(meeting_dir / "config.json")
    if not isinstance(config, dict):
        config = {}
    transcript = _read_json(meeting_dir / "transcript.json")
    if not isinstance(transcript, list):
        transcript = []

    with _file_lock(meeting_dir):
        state = _read_json(meeting_dir / "state.json")
        if not isinstance(state, dict):
            state = {}

        cursors: dict[str, int] = state.get("read_cursors", {})
        last_read = cursors.get(agent_name, 0)
        new_messages = transcript[last_read:]

        # Advance cursor — agent has now "seen" everything up to this point
        cursors[agent_name] = len(transcript)
        state["read_cursors"] = cursors
        _write_json(meeting_dir / "state.json", state)

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
    meeting_dir = _get_meeting_dir(meeting_id)
    if not meeting_dir.exists():
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _read_json(meeting_dir / "config.json")
    if not isinstance(config, dict):
        config = {}
    participants: list[str] = config.get("participants", [])
    max_rounds = config.get("max_rounds", 3)

    with _file_lock(meeting_dir):
        state = _read_json(meeting_dir / "state.json")
        if not isinstance(state, dict):
            state = {}

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

        transcript = _read_json(meeting_dir / "transcript.json")
        if not isinstance(transcript, list):
            transcript = []

        turn_entry = {
            "turn": len(transcript) + 1,
            "round": state.get("current_round", 1),
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": message,
            "type": "speak",
        }
        transcript.append(turn_entry)
        _write_json(meeting_dir / "transcript.json", transcript)

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
                    _write_json(meeting_dir / "config.json", config)
                    max_rounds = new_max
                    _audit(
                        meeting_dir,
                        f"Extended max_rounds to {new_max} after FAIL verdict",
                    )

                action_items: list[dict[str, Any]] = state.get("action_items", [])
                reason_match = re.search(
                    r"\[DECISION\]\s*VERDICT:\s*FAIL\s*[-—:.]?\s*(.+)",
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
                _write_json(meeting_dir / "state.json", state)

                next_speaker = (
                    participants[next_turn] if next_turn < len(participants) else "unknown"
                )
                _audit(
                    meeting_dir,
                    f"{agent_name} [round {turn_entry['round']}]: VERDICT FAIL — meeting continues",
                )

                return json.dumps(
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

            # PASS, RESOLVED, ESCALATE, ESCALATE_TO_USER → end
            state["ended"] = True
            state["outcome"] = f"verdict_{verdict}"
            _write_json(meeting_dir / "state.json", state)
            _audit(
                meeting_dir,
                f"{agent_name} [round {state.get('current_round', 1)}]: {message[:200]}",
            )
            _audit(
                meeting_dir,
                f"Meeting ended: verdict_{verdict}",
            )
            return json.dumps(
                {
                    "status": "spoken",
                    "meeting_ended": True,
                    "verdict": verdict,
                    "turn": turn_entry["turn"],
                }
            )

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
                _write_json(meeting_dir / "state.json", state)
                _audit(meeting_dir, f"{agent_name}: {message[:200]}")
                _audit(
                    meeting_dir,
                    "Meeting ended: max_rounds_reached",
                )
                return json.dumps(
                    {
                        "status": "spoken",
                        "meeting_ended": True,
                        "reason": "max_rounds_reached",
                        "turn": turn_entry["turn"],
                    }
                )

        state["current_turn"] = next_turn
        state["current_round"] = current_round
        _write_json(meeting_dir / "state.json", state)

        next_speaker = participants[next_turn]

    _audit(
        meeting_dir,
        f"{agent_name} [round {turn_entry['round']}]: {message[:200]}",
    )
    _audit(
        meeting_dir,
        f"Turn advanced → {next_speaker} (round {current_round})",
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
    meeting_dir = _get_meeting_dir(meeting_id)
    if not meeting_dir.exists():
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _read_json(meeting_dir / "config.json")
    if not isinstance(config, dict):
        config = {}
    participants: list[str] = config.get("participants", [])
    max_rounds = config.get("max_rounds", 3)

    with _file_lock(meeting_dir):
        state = _read_json(meeting_dir / "state.json")
        if not isinstance(state, dict):
            state = {}

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

        transcript = _read_json(meeting_dir / "transcript.json")
        if not isinstance(transcript, list):
            transcript = []

        turn_entry = {
            "turn": len(transcript) + 1,
            "round": state.get("current_round", 1),
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": reason or "(skipped)",
            "type": "skip",
        }
        transcript.append(turn_entry)
        _write_json(meeting_dir / "transcript.json", transcript)

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
                _write_json(meeting_dir / "state.json", state)
                _audit(
                    meeting_dir,
                    f"{agent_name} skipped: {reason}",
                )
                _audit(
                    meeting_dir,
                    "Meeting ended: max_rounds_reached",
                )
                return json.dumps(
                    {
                        "status": "skipped",
                        "meeting_ended": True,
                        "reason": "max_rounds_reached",
                        "turn": turn_entry["turn"],
                    }
                )

        state["current_turn"] = next_turn
        state["current_round"] = current_round
        _write_json(meeting_dir / "state.json", state)

        next_speaker = participants[next_turn]

    _audit(
        meeting_dir,
        f"{agent_name} skipped turn: {reason or 'no reason'}",
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
    meeting_dir = _get_meeting_dir(meeting_id)
    if not meeting_dir.exists():
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _read_json(meeting_dir / "config.json")
    if not isinstance(config, dict):
        config = {}
    state = _read_json(meeting_dir / "state.json")
    if not isinstance(state, dict):
        state = {}
    participants: list[str] = config.get("participants", [])
    current_turn = state.get("current_turn", 0)

    transcript = _read_json(meeting_dir / "transcript.json")
    if not isinstance(transcript, list):
        transcript = []

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
    meeting_dir = _get_meeting_dir(meeting_id)
    if not meeting_dir.exists():
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    with _file_lock(meeting_dir):
        config = _read_json(meeting_dir / "config.json")
        if not isinstance(config, dict):
            config = {}
        state = _read_json(meeting_dir / "state.json")
        if not isinstance(state, dict):
            state = {}

        if state.get("ended"):
            return json.dumps({"error": ("Cannot add participant — meeting already ended")})

        participants: list[str] = config.get("participants", [])
        if agent_name in participants:
            return json.dumps({"status": "already_participant", "agent_name": agent_name})

        participants.append(agent_name)
        config["participants"] = participants
        _write_json(meeting_dir / "config.json", config)

    # Auto-invite the new participant
    bus = _get_bus()
    if bus:
        agenda = config.get("agenda", "")
        invite_content = (
            f"🔔 MEETING INVITE (mid-meeting)\n"
            f"Agenda: {agenda}\n"
            f"Meeting ID: {meeting_id}\n"
            f"You were added to an ongoing meeting.\n\n"
            f"→ join_meeting(meeting_id=\"{meeting_id}\", "
            f"agent_name=\"{agent_name}\")\n"
            f"→ wait_for_my_turn(meeting_id=\"{meeting_id}\", "
            f"agent_name=\"{agent_name}\")"
        )
        bus.send(
            from_name=_get_my_name(),
            to_name=agent_name,
            content=invite_content,
            message_type="meeting_invite",
        )
        _auto_wake_if_idle(agent_name)

    _audit(
        meeting_dir,
        f"Participant added mid-meeting: {agent_name}",
    )

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
    meeting_dir = _get_meeting_dir(meeting_id)
    if not meeting_dir.exists():
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _read_json(meeting_dir / "config.json")
    if not isinstance(config, dict):
        config = {}
    participants: list[str] = config.get("participants", [])

    with _file_lock(meeting_dir):
        state = _read_json(meeting_dir / "state.json")
        if not isinstance(state, dict):
            state = {}

        if state.get("ended"):
            return json.dumps({"error": "Meeting already ended"})

        if agent_name not in participants:
            return json.dumps({"error": f"'{agent_name}' not in meeting"})

        current_turn = state.get("current_turn", 0)
        my_index = participants.index(agent_name)
        participants.remove(agent_name)
        config["participants"] = participants
        _write_json(meeting_dir / "config.json", config)

        if my_index < current_turn:
            state["current_turn"] = current_turn - 1
        elif my_index == current_turn:
            if len(participants) > 0:
                state["current_turn"] = current_turn % len(participants)
            else:
                state["ended"] = True
                state["outcome"] = "all_left"

        transcript = _read_json(meeting_dir / "transcript.json")
        if not isinstance(transcript, list):
            transcript = []
        transcript.append(
            {
                "turn": len(transcript) + 1,
                "round": state.get("current_round", 1),
                "agent": agent_name,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "message": (f"Left the meeting: {reason}" if reason else "Left the meeting"),
                "type": "leave",
            }
        )
        _write_json(meeting_dir / "transcript.json", transcript)
        _write_json(meeting_dir / "state.json", state)

    _audit(
        meeting_dir,
        f"{agent_name} left the meeting: {reason or 'no reason given'}",
    )

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
    mcp.run()
