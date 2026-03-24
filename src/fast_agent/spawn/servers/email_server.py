"""Email MCP Server — async inter-agent messaging.

Provides send_email, read_email, and check_teammate_status tools.
Extracted from meeting_room_server to separate meeting (synchronous,
turn-based) from messaging (async, fire-and-forget) concerns.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP

from fast_agent.spawn.servers._team_helpers import (
    auto_wake_if_idle,
    get_bus,
    get_my_name,
    get_project_registry,
    get_team_config,
    parse_recipients,
    resolve_agent_name,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("email")


@mcp.tool()
def send_email(
    to: str,
    body: str,
    subject: str = "",
    my_name: str = "",
    cc: str = "",
    priority: str = "normal",
) -> str:
    """Send an email to one or more teammates. No meeting needed.

    Unlike meetings (which require turn-taking), this is fire-and-forget.
    The email is queued and the recipient reads it when available.
    If the recipient is idle, they are auto-woken.

    Use this for: quick questions, notifications, status updates,
    blocker alerts, or any message that doesn't need a formal meeting.

    Args:
        to: Recipient name(s). Single: "Minh - Dev".
            Multiple (comma-separated): "Minh - Dev, Tuan - QE".
            Broadcast: "all".
        body: Your email content.
        subject: Short summary of the email (shown in timeline).
        my_name: YOUR agent name (for sender tracking).
        cc: Optional CC recipients (comma-separated). They receive an
            informational copy prefixed with [CC]. Use for keeping
            stakeholders informed without direct action needed.
        priority: "normal" | "high" | "low"
    """
    bus = get_bus()
    if not bus:
        return json.dumps({"error": "No workspace configured. Cannot send emails."})

    my_name = my_name or get_my_name()
    recipients = parse_recipients(to)
    if not recipients:
        return json.dumps({"error": "'to' must specify at least one recipient."})

    # Guard: reject self-messaging
    recipients = [r for r in recipients if r != my_name]
    if not recipients:
        teammates = [
            cfg.get("agent_name", "")
            for cfg in get_team_config().values()
            if cfg.get("agent_name") != my_name
        ]
        return json.dumps({
            "error": "Cannot send email to yourself. Use send_email to contact teammates.",
            "available_teammates": teammates,
        })

    # Generate batch_id for grouping multi-recipient sends in timeline
    batch_id = f"batch_{uuid.uuid4().hex[:8]}"

    sent: list[dict[str, str]] = []
    # Primary recipients
    for recipient in recipients:
        msg = bus.send(
            from_name=my_name,
            to_name=recipient,
            content=body,
            message_type="email",
            priority=priority,
            context={
                "subject": subject,
                "batch_id": batch_id,
                "recipient_type": "to",
            },
        )
        auto_wake_if_idle(recipient)
        sent.append({"to": recipient, "message_id": msg.message_id, "type": "to"})

    # CC recipients — informational copy
    cc_recipients = parse_recipients(cc) if cc else []
    cc_recipients = [r for r in cc_recipients if r != my_name and r not in recipients]
    cc_sent: list[dict[str, str]] = []
    if cc_recipients:
        to_names = ", ".join(recipients)
        cc_content = f"[CC — originally to: {to_names}]\n{body}"
        for recipient in cc_recipients:
            msg = bus.send(
                from_name=my_name,
                to_name=recipient,
                content=cc_content,
                message_type="notification",
                priority=priority,
                context={
                    "subject": subject,
                    "batch_id": batch_id,
                    "recipient_type": "cc",
                },
            )
            auto_wake_if_idle(recipient)
            cc_sent.append({"to": recipient, "message_id": msg.message_id, "type": "cc"})

    result: dict[str, Any] = {
        "status": "sent",
        "from": my_name,
        "batch_id": batch_id,
        "sent": sent,
        "note": (
            f"Email delivered to {', '.join(r['to'] for r in sent)}. "
            f"They will read it when available."
        ),
    }
    if cc_sent:
        result["cc_sent"] = cc_sent
        result["note"] += f" CC: {', '.join(r['to'] for r in cc_sent)}."
    return json.dumps(result)


@mcp.tool()
async def read_email(
    my_name: str = "",
    from_agent: str = "",
    wait: bool = False,
    timeout_seconds: int = 60,
) -> str:
    """Read your unread emails (inbox).

    Use this to check for emails from teammates — status updates,
    questions, notifications, or completion signals.

    When from_agent is specified with wait=True:
    - If there are emails from that agent → returns them immediately.
    - If there are OTHER unread emails (not from that agent) → returns
      those with status "has_other_emails" so you can process them
      instead of blocking indefinitely.
    - If inbox is completely empty → polls until timeout.

    Args:
        my_name: YOUR agent name (to identify your inbox).
        from_agent: Optional — filter to only show emails from this agent.
                    Leave empty to see all emails.
        wait: If True, poll every 3s until an email arrives or timeout.
              If False (default), check once and return immediately.
        timeout_seconds: Max time to wait when wait=True. Default 60s.
    """
    import asyncio
    import time as _time

    bus = get_bus()
    if not bus:
        return json.dumps({"error": "No workspace configured."})

    my_name = my_name or get_my_name()
    poll_interval = 3.0
    start = _time.time()

    while True:
        all_unread = bus.read_unread(my_name)

        if from_agent:
            resolved = resolve_agent_name(from_agent) or from_agent
            from_sender = [m for m in all_unread if m.from_name == resolved]
            other_msgs = [m for m in all_unread if m.from_name != resolved]

            if from_sender:
                # Found emails from the requested sender → return them
                result = []
                for msg in from_sender:
                    result.append({
                        "message_id": msg.message_id,
                        "from": msg.from_name,
                        "type": msg.message_type,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                    })
                    bus.mark_done(my_name, msg.message_id)
                return json.dumps({
                    "status": "has_emails",
                    "count": len(result),
                    "messages": result,
                })

            if other_msgs:
                # No email from requested sender, but OTHER unread exist.
                # Return them so the agent can process and move on.
                result = []
                for msg in other_msgs:
                    result.append({
                        "message_id": msg.message_id,
                        "from": msg.from_name,
                        "type": msg.message_type,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                    })
                    bus.mark_done(my_name, msg.message_id)
                return json.dumps({
                    "status": "has_other_emails",
                    "count": len(result),
                    "messages": result,
                    "note": (
                        f"No email from {from_agent} yet, but you have "
                        f"{len(result)} other unread email(s). "
                        f"Process these and decide your next steps."
                    ),
                })

            # Inbox truly empty for this filter → poll or break
        else:
            # No filter — return all unread
            if all_unread:
                result = []
                for msg in all_unread:
                    result.append({
                        "message_id": msg.message_id,
                        "from": msg.from_name,
                        "type": msg.message_type,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                    })
                    bus.mark_done(my_name, msg.message_id)
                return json.dumps({
                    "status": "has_emails",
                    "count": len(result),
                    "messages": result,
                })

        if not wait or (_time.time() - start) >= timeout_seconds:
            break

        # ── Idle detection: avoid blocking forever when all teammates idle ──
        if from_agent and wait:
            try:
                registry = get_project_registry()
                if registry:
                    resolved_from = resolve_agent_name(from_agent) or from_agent
                    record = registry.find_by_name(resolved_from)
                    if record and record.status in ("idle", "completed"):
                        # Check if agent is alive via AgentChannel
                        try:
                            from fast_agent.spawn.agent_channel import AgentChannel
                            alive = AgentChannel.is_alive(resolved_from)
                        except Exception:
                            alive = False

                        if not alive:
                            return json.dumps({
                                "status": "teammate_idle",
                                "agent": resolved_from,
                                "agent_status": record.status,
                                "message": (
                                    f"{resolved_from} is {record.status} and not actively processing. "
                                    f"They won't send new emails until woken. "
                                    f"Consider: send_email to wake them, or proceed with your own work."
                                ),
                            })
            except Exception:
                pass  # Best-effort; continue polling

        await asyncio.sleep(poll_interval)

    filter_note = f" from {from_agent}" if from_agent else ""
    if wait:
        return json.dumps({
            "status": "timeout",
            "message": f"No emails{filter_note} after {timeout_seconds}s.",
        })

    return json.dumps({
        "status": "empty",
        "message": f"No unread emails{filter_note}.",
    })


def _get_recent_activities(run_id: str, limit: int = 3) -> list[dict]:
    """Read last N tool activities from spawn_events.jsonl for a given run_id."""
    import os
    import time as _time
    from pathlib import Path

    workspace = os.environ.get("TEAM_WORKSPACE", "")
    if not workspace:
        return []

    # Find .runtime root → spawn_events.jsonl
    cur = Path(workspace)
    events_file = None
    while cur != cur.parent:
        if cur.name == ".runtime":
            events_file = cur / "state" / "spawn_events.jsonl"
            break
        cur = cur.parent

    if not events_file or not events_file.exists():
        return []

    try:
        # Read last ~100 lines efficiently
        with open(events_file, "rb") as f:
            # Seek to near end of file
            f.seek(0, 2)  # End of file
            file_size = f.tell()
            read_size = min(file_size, 32768)  # Last 32KB
            f.seek(max(0, file_size - read_size))
            tail_bytes = f.read()

        lines = tail_bytes.decode("utf-8", errors="replace").strip().split("\n")

        # Filter for this run_id's tool events
        now = _time.time()
        activities: list[dict] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                evt = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            evt_run_id = evt.get("run_id") or evt.get("data", {}).get("run_id")
            if evt_run_id != run_id:
                continue

            evt_type = evt.get("event_type", "")
            if evt_type not in ("tool_call", "tool_result", "thinking"):
                continue

            data = evt.get("data", {})
            ts = evt.get("timestamp") or data.get("timestamp")
            ago = ""
            if ts:
                try:
                    diff = int(now - float(ts))
                    if diff < 60:
                        ago = f"{diff}s ago"
                    elif diff < 3600:
                        ago = f"{diff // 60}m ago"
                    else:
                        ago = f"{diff // 3600}h ago"
                except (ValueError, TypeError):
                    pass

            activity = {"event": evt_type}
            if evt_type in ("tool_call", "tool_result"):
                activity["tool"] = data.get("tool_name", "unknown")
            if ago:
                activity["ago"] = ago

            activities.append(activity)
            if len(activities) >= limit:
                break

        return activities
    except Exception:
        return []


@mcp.tool()
def check_teammate_status(agent_name: str) -> str:
    """Check a teammate's current status and recent activity.

    Use this to check if a dependency is met (e.g., "is SA done with
    architecture?") before starting your own work. Also shows recent
    tool activity so you can tell if the agent is actively working.

    Args:
        agent_name: The teammate to check (e.g. "Khang - SA").
                    Use "all" to check all teammates at once.

    Returns:
        JSON with status: "not_spawned" | "running" | "idle" |
                          "completed" | "error"
        Plus: result summary if completed, recent_activities if running.
        When agent_name="all", returns a dict of all teammate statuses.
    """
    try:
        registry = get_project_registry()
        if not registry:
            logger.warning(
                "check_teammate_status(%s): no registry found", agent_name
            )
            return json.dumps({
                "agent_name": agent_name,
                "status": "unknown",
                "error": "No spawn registry found. Check SPAWN_PROJECT_DIR env.",
            })

        # Batch mode: check all teammates
        if agent_name.strip().lower() == "all":
            team_config = get_team_config()
            my_name = get_my_name()
            all_status: dict[str, dict] = {}
            for _role, cfg in team_config.items():
                name = cfg.get("agent_name", "")
                if not name or name == my_name:
                    continue
                record = registry.find_by_name(name)
                if not record:
                    all_status[name] = {"status": "not_spawned"}
                else:
                    info: dict = {"status": record.status, "run_id": record.run_id}
                    if record.status == "completed" and record.result:
                        info["result_preview"] = record.result[:500]
                    elif record.status in ("running", "idle"):
                        activities = _get_recent_activities(record.run_id, limit=3)
                        if activities:
                            info["recent_activities"] = activities
                            info["last_active"] = activities[0].get("ago", "unknown")
                    all_status[name] = info
            return json.dumps({"teammates": all_status, "count": len(all_status)})

        # Single agent mode
        record = registry.find_by_name(agent_name)

        if not record:
            return json.dumps({
                "agent_name": agent_name,
                "status": "not_spawned",
            })

        result_info: dict = {
            "agent_name": agent_name,
            "status": record.status,
            "run_id": record.run_id,
        }

        if record.status == "completed" and record.result:
            result_info["result_preview"] = record.result[:2000]
        elif record.status in ("running", "idle"):
            activities = _get_recent_activities(record.run_id, limit=3)
            if activities:
                result_info["recent_activities"] = activities
                result_info["last_active"] = activities[0].get("ago", "unknown")

        return json.dumps(result_info)

    except Exception as e:
        logger.error(
            "check_teammate_status(%s) failed: %s", agent_name, e, exc_info=True
        )
        return json.dumps({
            "agent_name": agent_name,
            "status": "unknown",
            "error": str(e),
        })


if __name__ == "__main__":
    mcp.run()
