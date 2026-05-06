"""Email MCP Server — async inter-agent messaging.

Provides send_email tool for inter-agent messaging.
Teammate status is auto-delivered via consolidated team notifications.
Email delivery to agents is handled automatically via RTAC (InboxWatcherHook)
and the keep-alive loop — agents never need to poll their inbox.
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
    no_reply: bool = False,
) -> str:
    """Send an email to one or more teammates. No meeting needed.

    Unlike meetings (which require turn-taking), this is fire-and-forget.
    The email is queued and the recipient reads it when available.
    If the recipient is idle, they are auto-woken.

    Use this for: quick questions, notifications, status updates,
    blocker alerts, or any message that doesn't need a formal meeting.

    Args:
        to: Recipient name(s). Single: "Agent A - Dev".
            Multiple (comma-separated): "Agent A - Dev, Agent B - QE".
            Broadcast: "all".
        body: Your email content.
        subject: Short summary of the email (shown in timeline).
        my_name: YOUR agent name (for sender tracking).
        cc: Optional CC recipients (comma-separated). They receive an
            informational copy prefixed with [CC]. Use for keeping
            stakeholders informed without direct action needed.
        priority: "normal" | "high" | "low"
        no_reply: If True, marks this email as informational only.
            Recipients will see [NO REPLY NEEDED] and should NOT reply.
            Use for: deliverable notifications, FYI updates, status broadcasts.
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

    # Apply no-reply prefix
    email_body = f"[NO REPLY NEEDED]\n{body}" if no_reply else body

    # Generate batch_id for grouping multi-recipient sends in timeline
    batch_id = f"batch_{uuid.uuid4().hex[:8]}"

    sent: list[dict[str, str]] = []
    # Primary recipients
    for recipient in recipients:
        msg = bus.send(
            from_name=my_name,
            to_name=recipient,
            content=email_body,
            message_type="email",
            priority=priority,
            context={
                "subject": subject,
                "batch_id": batch_id,
                "recipient_type": "to",
                "no_reply": no_reply,
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
        cc_content = f"[CC — originally to: {to_names}]\n{email_body}"
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
                    "no_reply": no_reply,
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



def _get_recent_activities(run_id: str, limit: int = 3) -> list[dict]:
    """Read last N tool activities from agent_activities DB for a given run_id."""
    import os
    import sqlite3
    import time as _time

    db_path = os.environ.get("SPAWN_REGISTRY_DB")
    if not db_path:
        return []

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        rows = conn.execute(
            """SELECT event_type, data_json, created_at
               FROM agent_activities
               WHERE run_id = ? AND event_type IN ('tool_call', 'tool_result')
               ORDER BY created_at DESC LIMIT ?""",
            (run_id, limit),
        ).fetchall()
        conn.close()

        now = _time.time()
        activities: list[dict] = []
        for event_type, data_json, created_at in rows:
            activity: dict = {"event": event_type}
            if data_json:
                try:
                    data = json.loads(data_json)
                    activity["tool"] = data.get("tool_name", "unknown")
                except (json.JSONDecodeError, ValueError):
                    pass
            try:
                diff = int(now - float(created_at))
                if diff < 60:
                    activity["ago"] = f"{diff}s ago"
                elif diff < 3600:
                    activity["ago"] = f"{diff // 60}m ago"
                else:
                    activity["ago"] = f"{diff // 3600}h ago"
            except (ValueError, TypeError):
                pass
            activities.append(activity)
        return activities
    except Exception:
        return []




if __name__ == "__main__":
    mcp.run()
