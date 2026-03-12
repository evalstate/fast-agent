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
    workspace_dir: str = "",
) -> str:
    """Create a new meeting room. Call BEFORE spawning participants.

    Args:
        agenda: What this meeting is about.
        participants: Comma-separated roles in speaking order.
        max_rounds: Maximum conversation rounds (default 3).
        workspace_dir: Workspace path (auto-detected from env).

    Returns:
        JSON with meeting_id and config.
    """
    if not workspace_dir:
        workspace_dir = os.environ.get(
            "TEAM_WORKSPACE",
            str(Path.cwd() / ".runtime" / "cache" / "tmp" / "meeting-workspace"),
        )
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
        "created_at": datetime.now().isoformat(),
    }
    _write_json(meeting_dir / "config.json", config)

    state: dict[str, Any] = {
        "current_turn": 0,
        "current_round": 1,
        "joined": [],
        "ended": False,
        "outcome": None,
        "started": False,
    }
    _write_json(meeting_dir / "state.json", state)
    _write_json(meeting_dir / "transcript.json", [])

    _audit(
        meeting_dir,
        f"Meeting created: agenda='{agenda}', "
        f"participants={participant_list}, max_rounds={max_rounds}",
    )

    logger.info("Meeting created: %s — %s", meeting_id, agenda)
    return json.dumps(
        {
            "meeting_id": meeting_id,
            "agenda": agenda,
            "participants": participant_list,
            "max_rounds": max_rounds,
            "status": "waiting_for_participants",
        }
    )


@mcp.tool()
async def join_meeting(meeting_id: str, role: str) -> str:
    """Join an existing meeting. Call before ``wait_for_my_turn``.

    When all expected participants have joined, the meeting starts.

    Args:
        meeting_id: The meeting to join.
        role: Your role (e.g. "qe", "dev").

    Returns:
        JSON with meeting config and join status.
    """
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
        if role not in joined:
            joined.append(role)
            state["joined"] = joined

        participants: list[str] = config.get("participants", [])
        if role not in participants:
            participants.append(role)
            config["participants"] = participants
            _write_json(meeting_dir / "config.json", config)

        all_joined = all(p in joined for p in participants)
        if all_joined:
            state["started"] = True

        _write_json(meeting_dir / "state.json", state)

    _audit(meeting_dir, f"{role} joined the meeting")
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
            "your_role": role,
        }
    )


@mcp.tool()
async def wait_for_my_turn(meeting_id: str, role: str, timeout_seconds: int = 180) -> str:
    """Wait (poll) until it's your turn to speak or the meeting ends.

    Call after ``join_meeting``. When this returns ``your_turn``, call
    ``get_transcript`` then ``speak`` or ``skip_turn``.

    Args:
        meeting_id: The meeting to wait in.
        role: Your role.
        timeout_seconds: Max seconds to wait (default 180).

    Returns:
        JSON with status: "your_turn", "meeting_ended", or "timeout".
    """
    meeting_dir = _get_meeting_dir(meeting_id)
    if not meeting_dir.exists():
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _read_json(meeting_dir / "config.json")
    if not isinstance(config, dict):
        config = {}
    participants: list[str] = config.get("participants", [])

    if role not in participants:
        return json.dumps({"error": (f"Role '{role}' not in meeting participants: {participants}")})

    my_index = participants.index(role)
    start_time = time.time()
    poll_interval = 2

    while True:
        state = _read_json(meeting_dir / "state.json")
        if not isinstance(state, dict):
            state = {}

        if state.get("ended"):
            return json.dumps(
                {
                    "status": "meeting_ended",
                    "outcome": state.get("outcome"),
                    "meeting_id": meeting_id,
                }
            )

        if not state.get("started"):
            if time.time() - start_time > timeout_seconds:
                return json.dumps(
                    {
                        "status": "timeout",
                        "reason": "waiting_for_participants",
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
            return json.dumps({"status": "timeout", "reason": "waited_too_long"})

        await asyncio.sleep(poll_interval)


@mcp.tool()
async def get_transcript(meeting_id: str) -> str:
    """Get the full conversation transcript of a meeting.

    Args:
        meeting_id: The meeting to get transcript for.

    Returns:
        JSON with transcript messages in chronological order.
    """
    meeting_dir = _get_meeting_dir(meeting_id)
    if not meeting_dir.exists():
        return json.dumps({"error": f"Meeting '{meeting_id}' not found"})

    config = _read_json(meeting_dir / "config.json")
    if not isinstance(config, dict):
        config = {}
    transcript = _read_json(meeting_dir / "transcript.json")
    if not isinstance(transcript, list):
        transcript = []

    state = _read_json(meeting_dir / "state.json")
    if not isinstance(state, dict):
        state = {}

    return json.dumps(
        {
            "meeting_id": meeting_id,
            "agenda": config.get("agenda", ""),
            "current_round": state.get("current_round", 1),
            "transcript": transcript,
            "message_count": len(transcript),
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def speak(meeting_id: str, role: str, message: str) -> str:
    """Add your message to the meeting transcript.

    After speaking, the turn automatically advances to the next
    participant. If your message contains ``VERDICT: PASS``,
    ``VERDICT: FAIL``, ``VERDICT: ESCALATE``, or
    ``VERDICT: ESCALATE_TO_USER``, the meeting ends automatically
    (except FAIL, which continues the meeting).

    Args:
        meeting_id: The meeting to speak in.
        role: Your role (must be the current speaker).
        message: What you want to say.

    Returns:
        JSON with confirmation and next turn info.
    """
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
        if role != current_speaker:
            return json.dumps(
                {
                    "error": (f"Not your turn. Current speaker: {current_speaker}"),
                    "your_role": role,
                }
            )

        transcript = _read_json(meeting_dir / "transcript.json")
        if not isinstance(transcript, list):
            transcript = []

        turn_entry = {
            "turn": len(transcript) + 1,
            "round": state.get("current_round", 1),
            "agent": role,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": message,
            "type": "speak",
        }
        transcript.append(turn_entry)
        _write_json(meeting_dir / "transcript.json", transcript)

        # Check for verdict
        verdict_match = re.search(
            r"VERDICT:\s*(PASS|FAIL|ESCALATE"
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
                    r"VERDICT:\s*FAIL\s*[-—:.]?\s*(.+)",
                    message,
                    re.IGNORECASE,
                )
                action_items.append(
                    {
                        "from": role,
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
                    f"{role} [round {turn_entry['round']}]: VERDICT FAIL — meeting continues",
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
                f"{role} [round {state.get('current_round', 1)}]: {message[:200]}",
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
                _audit(meeting_dir, f"{role}: {message[:200]}")
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
        f"{role} [round {turn_entry['round']}]: {message[:200]}",
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
async def skip_turn(meeting_id: str, role: str, reason: str = "") -> str:
    """Skip your turn. The turn advances to the next participant.

    Args:
        meeting_id: The meeting.
        role: Your role.
        reason: Optional reason for skipping.

    Returns:
        JSON with confirmation.
    """
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
        if role != current_speaker:
            return json.dumps(
                {
                    "error": (f"Not your turn. Current speaker: {current_speaker}"),
                    "your_role": role,
                }
            )

        transcript = _read_json(meeting_dir / "transcript.json")
        if not isinstance(transcript, list):
            transcript = []

        turn_entry = {
            "turn": len(transcript) + 1,
            "round": state.get("current_round", 1),
            "agent": role,
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
                    f"{role} skipped: {reason}",
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
        f"{role} skipped turn: {reason or 'no reason'}",
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
async def add_participant(meeting_id: str, role: str) -> str:
    """Add a new participant to an ongoing meeting (for escalation).

    Args:
        meeting_id: The meeting.
        role: Role to add.

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
        if role in participants:
            return json.dumps({"status": "already_participant", "role": role})

        participants.append(role)
        config["participants"] = participants
        _write_json(meeting_dir / "config.json", config)

    _audit(
        meeting_dir,
        f"Participant added mid-meeting: {role}",
    )

    return json.dumps(
        {
            "status": "added",
            "role": role,
            "participants": participants,
        }
    )


@mcp.tool()
async def leave_meeting(meeting_id: str, role: str, reason: str = "") -> str:
    """Leave a meeting. Your turns will be skipped.

    Args:
        meeting_id: The meeting to leave.
        role: Your role.
        reason: Why you're leaving.

    Returns:
        JSON confirming you've left.
    """
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

        if role not in participants:
            return json.dumps({"error": f"Role '{role}' not in meeting"})

        current_turn = state.get("current_turn", 0)
        my_index = participants.index(role)
        participants.remove(role)
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
                "agent": role,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "message": (f"Left the meeting: {reason}" if reason else "Left the meeting"),
                "type": "leave",
            }
        )
        _write_json(meeting_dir / "transcript.json", transcript)
        _write_json(meeting_dir / "state.json", state)

    _audit(
        meeting_dir,
        f"{role} left the meeting: {reason or 'no reason given'}",
    )

    return json.dumps(
        {
            "status": "left",
            "role": role,
            "remaining_participants": participants,
        }
    )


if __name__ == "__main__":
    mcp.run()
