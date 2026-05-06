"""Team Communicate MCP Server — agent-initiated inter-agent messaging.

Provides tools for agents to send messages to teammates and check responses.
Messages go through the MessageBus queue — no sub-agent clones are spawned.

Tools:
- ``team_communicate`` — send a message to a teammate's inbox queue
- ``check_responses`` — read your inbox for responses from teammates
- ``reply_to_message`` — reply to a specific message
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from fast_agent.spawn.message_bus import MessageBus
from fast_agent.spawn.servers._team_helpers import (
    auto_wake_if_idle as _auto_wake_if_idle,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("team-communicate")


def _get_bus() -> MessageBus | None:
    """Get MessageBus from TEAM_WORKSPACE env var.

    TEAM_WORKSPACE is injected into fastagent.config.yaml's env section
    by get_server_env() during child config generation.
    """
    from pathlib import Path

    workspace = os.environ.get("TEAM_WORKSPACE", "")
    if not workspace:
        return None
    # Workspace is like: .runtime/data/workspaces/agile-team_xxx
    # We need:           .runtime/state/messages
    # Walk up until we find .runtime root
    cur = Path(workspace)
    while cur != cur.parent:
        if cur.name == ".runtime":
            state_dir = cur / "state" / "messages"
            state_dir.mkdir(parents=True, exist_ok=True)
            return MessageBus(messages_dir=str(state_dir))
        cur = cur.parent
    return None


def _get_my_name() -> str:
    """Get current agent's name."""
    return os.environ.get("TEAM_MY_NAME", os.environ.get("TEAM_MY_ROLE", "agent"))


def _get_team_config() -> dict[str, Any]:
    """Load team roles config from env."""
    try:
        return json.loads(os.environ.get("TEAM_ROLES_CONFIG", "{}"))
    except json.JSONDecodeError:
        return {}


def _resolve_agent_name(to: str) -> str | None:
    """Resolve target agent name — supports both name and role key lookup."""
    team_config = _get_team_config()

    # Direct match by agent_name
    for _role_key, cfg in team_config.items():
        if isinstance(cfg, dict) and cfg.get("agent_name") == to:
            return to

    # Fallback: match by role key → return agent_name
    if to in team_config:
        cfg = team_config[to]
        if isinstance(cfg, dict):
            return cfg.get("agent_name", to)

    return None


# _auto_wake_if_idle imported from _team_helpers (AgentChannel-first wake)


def _parse_recipients(value: str) -> list[str]:
    """Parse a recipient string into a list of names.

    Accepts:
      - Single name: "Agent A - Dev"
      - Comma-separated: "Agent A - Dev, Agent C - PM"
    """
    if not value:
        return []
    return [name.strip() for name in value.split(",") if name.strip()]


@mcp.tool()
def team_communicate(
    to: str,
    message: str,
    my_name: str = "",
    message_type: str = "question",
    cc: str = "",
) -> str:
    """Send a message to one or more teammates' inbox queues.

    This is NON-BLOCKING — your message is queued and teammates
    will process it when available. If they are idle, they will be
    woken up automatically. Use ``check_responses`` to read replies.

    Args:
        to: Primary recipient(s) who should take action.
            Single name: "Agent A - Dev"
            Multiple names (comma-separated): "Agent A - Dev, Agent B - QE"
        message: Your message or question.
        my_name: YOUR agent name (e.g. "Agent D - BA"). Required for proper sender tracking.
        message_type: "question" | "task" | "review_request" | "feedback" | "response"
        cc: Optional FYI recipient(s) — they receive the message tagged as [CC]
            so they know it was sent directly to the 'to' recipients.
            Comma-separated: "Agent C - PM" or "Agent C - PM, Agent D - BA"
    """
    bus = _get_bus()
    if not bus:
        return json.dumps({"error": "No workspace configured. Cannot send messages."})

    my_name = my_name or _get_my_name()
    to_list = _parse_recipients(to)
    cc_list = _parse_recipients(cc)

    if not to_list:
        return json.dumps({"error": "'to' must specify at least one recipient."})

    sent: list[dict[str, str]] = []

    # 1. Send to primary (direct) recipients
    for recipient in to_list:
        ctx: dict[str, Any] = {"delivery_type": "direct"}
        if cc_list:
            ctx["cc"] = cc_list

        msg = bus.send(
            from_name=my_name,
            to_name=recipient,
            content=message,
            message_type=message_type,
            context=ctx,
        )
        _auto_wake_if_idle(recipient)
        sent.append({"to": recipient, "message_id": msg.message_id, "delivery": "direct"})

    # 2. Send to CC (FYI) recipients
    for recipient in cc_list:
        ctx = {
            "delivery_type": "cc",
            "direct_to": to_list,
        }
        msg = bus.send(
            from_name=my_name,
            to_name=recipient,
            content=message,
            message_type=message_type,
            context=ctx,
        )
        _auto_wake_if_idle(recipient)
        sent.append({"to": recipient, "message_id": msg.message_id, "delivery": "cc"})

    # Build response
    all_recipients = [s["to"] for s in sent]
    return json.dumps(
        {
            "status": "queued",
            "sent": sent,
            "from": my_name,
            "to": to_list,
            "cc": cc_list,
            "note": (
                f"Message queued for {', '.join(all_recipients)}. "
                f"Use check_responses(wait=True) to wait for replies."
            ),
        }
    )


@mcp.tool()
def check_responses(
    my_name: str,
    from_agent: str = "",
    wait: bool = False,
    timeout_seconds: int = 120,
) -> str:
    """Check your inbox for messages from teammates.

    Args:
        my_name: YOUR agent name (e.g. "Agent A - Dev"). Required to identify your inbox.
        from_agent: Optional — filter to only show messages from this agent.
                    Leave empty to see all messages.
        wait: If True, poll every 3s until a message arrives or timeout.
              If False (default), check once and return immediately.
        timeout_seconds: Max time to wait when wait=True. Default 120s.
    """
    import time as _time

    bus = _get_bus()
    if not bus:
        return json.dumps({"error": "No workspace configured."})

    my_name = my_name or _get_my_name()

    poll_interval = 3.0
    start = _time.time()

    while True:
        messages = bus.read_unread(my_name)

        if from_agent:
            resolved = _resolve_agent_name(from_agent)
            if resolved:
                messages = [m for m in messages if m.from_name == resolved]
            else:
                messages = [m for m in messages if m.from_name == from_agent]

        if messages:
            result = []
            for msg in messages:
                entry: dict[str, Any] = {
                    "message_id": msg.message_id,
                    "from": msg.from_name,
                    "type": msg.message_type,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "reply_to": msg.reply_to,
                }
                # Add delivery metadata if available
                ctx = msg.context or {}
                delivery = ctx.get("delivery_type", "direct")
                entry["delivery"] = delivery
                if delivery == "cc":
                    direct_to = ctx.get("direct_to", [])
                    entry["direct_to"] = direct_to
                    entry["note"] = (
                        f"[CC] This message was sent directly to "
                        f"{', '.join(direct_to)}. You are CC'd for awareness only."
                    )
                elif ctx.get("cc"):
                    entry["cc"] = ctx["cc"]

                result.append(entry)
                # Mark message as done after reading
                bus.mark_done(my_name, msg.message_id)

            return json.dumps({
                "status": "has_messages",
                "count": len(result),
                "messages": result,
            })

        # No messages found
        if not wait or (_time.time() - start) >= timeout_seconds:
            break

        _time.sleep(poll_interval)

    # No messages after waiting (or immediate check)
    filter_note = f" from {from_agent}" if from_agent else ""
    if wait:
        return json.dumps({
            "status": "waiting",
            "message": (
                f"No reply{filter_note} after {timeout_seconds}s. "
                f"The agent may still be working. You can try again later."
            ),
        })

    return json.dumps({
        "status": "empty",
        "message": f"No unread messages{filter_note} in your inbox.",
    })


@mcp.tool()
def reply_to_message(
    to: str,
    message: str,
    my_name: str = "",
    original_message_id: str = "",
) -> str:
    """Reply to a message from a teammate.

    Args:
        to: Agent name to reply to.
        message: Your reply content.
        my_name: YOUR agent name (e.g. "Agent D - BA"). Required for proper sender tracking.
        original_message_id: The message_id you're replying to (optional).
    """
    bus = _get_bus()
    if not bus:
        return json.dumps({"error": "No workspace configured."})

    my_name = my_name or _get_my_name()
    resolved = to

    msg = bus.send(
        from_name=my_name,
        to_name=resolved,
        content=message,
        message_type="response",
        reply_to=original_message_id,
    )

    # Mark the original message as done in our inbox
    if original_message_id:
        bus.mark_done(my_name, original_message_id)

    # Auto-wake if receiver is idle
    _auto_wake_if_idle(resolved)

    return json.dumps({
        "status": "sent",
        "message_id": msg.message_id,
        "from": my_name,
        "to": resolved,
        "reply_to": original_message_id,
    })


if __name__ == "__main__":
    mcp.run()
