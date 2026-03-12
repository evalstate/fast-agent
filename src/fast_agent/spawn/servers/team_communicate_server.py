"""Team Communicate MCP Server — agent-initiated inter-role messaging.

Provides a single tool ``team_communicate`` that allows an agent to
directly contact another role during execution. Internally spawns a
fresh agent of the target role with the message + workspace context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("team-communicate")


def _get_workspace_context(workspace_dir: str) -> str:
    """Build context string from workspace files."""
    workspace = Path(workspace_dir)
    if not workspace.exists():
        return ""

    context_parts = [f"## Shared Workspace: {workspace}"]

    for subdir in ["specs", "src", "tests", "reviews", "docs"]:
        dir_path = workspace / subdir
        if dir_path.exists():
            files = list(dir_path.rglob("*"))
            file_list = [str(f.relative_to(workspace)) for f in files if f.is_file()]
            if file_list:
                context_parts.append(f"\n### {subdir}/\n" + "\n".join(f"- {f}" for f in file_list))

    changelog = workspace / "changelog.md"
    if changelog.exists():
        content = changelog.read_text()
        if content:
            context_parts.append(f"\n### Team Changelog\n{content[:2000]}")

    return "\n".join(context_parts)


@mcp.tool()
def team_communicate(
    to_role: str,
    message: str,
    message_type: str = "question",
) -> str:
    """Communicate with another team member by spawning them.

    This spawns a REAL agent of the target role who will read your
    message, review relevant workspace files, and give you a
    genuine response.

    Use this when you need:
    - Clarification on requirements (ask BA)
    - Architecture guidance (ask SA)
    - Code review feedback (ask QE)
    - Task status or priority info (ask PM)

    Args:
        to_role: Target role (e.g. "ba", "dev", "qe", "sa", "pm").
        message: Your message or question.
        message_type: "question" | "review_request" | "feedback"
    """
    workspace_dir = os.environ.get("TEAM_WORKSPACE", "")
    team_config_json = os.environ.get("TEAM_ROLES_CONFIG", "{}")
    my_role = os.environ.get("TEAM_MY_ROLE", "agent")
    project_dir = os.environ.get("SPAWN_PROJECT_DIR", os.getcwd())

    try:
        team_config: dict[str, Any] = json.loads(team_config_json)
    except json.JSONDecodeError:
        team_config = {}

    target_config = team_config.get(to_role, {})
    if not target_config and to_role not in team_config:
        available = list(team_config.keys())
        return json.dumps(
            {"error": (f"Role '{to_role}' not found in team. Available roles: {available}")}
        )

    task = (
        f"You received a {message_type} from the "
        f"{my_role} on your team:\n\n"
        f'"{message}"\n\n'
        f"Please respond thoughtfully based on your role "
        f"as {to_role}. "
        f"Review relevant workspace files if needed."
    )

    instruction = target_config.get(
        "instruction",
        f"You are the {to_role} on an agile team.",
    )

    workspace_context = _get_workspace_context(workspace_dir) if workspace_dir else ""
    context = (
        f"## Communication from {my_role}\n"
        f"Type: {message_type}\n"
        f"Message: {message}\n\n"
        f"{workspace_context}"
    )

    servers = target_config.get("servers", ["filesystem"])
    if isinstance(servers, str):
        servers = [s.strip() for s in servers.split(",") if s.strip()]

    from fast_agent.spawn.isolated_spawner import (
        run_isolated_agent,
    )

    current_depth = int(os.environ.get("TEAM_SPAWN_DEPTH", "1"))

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    lambda: asyncio.run(
                        run_isolated_agent(
                            task=task,
                            project_dir=project_dir,
                            instruction=instruction,
                            context=context,
                            servers=servers,
                            timeout_seconds=120,
                            role=to_role,
                            lifecycle="oneshot",
                            depth=current_depth + 1,
                        )
                    )
                ).result(timeout=180)
        else:
            result = loop.run_until_complete(
                run_isolated_agent(
                    task=task,
                    project_dir=project_dir,
                    instruction=instruction,
                    context=context,
                    servers=servers,
                    timeout_seconds=120,
                    role=to_role,
                    lifecycle="oneshot",
                    depth=current_depth + 1,
                )
            )
    except Exception as e:
        logger.error("team_communicate to %s failed: %s", to_role, e)
        return json.dumps(
            {
                "error": f"Failed to reach {to_role}: {e!s}",
                "from": my_role,
                "to": to_role,
            }
        )

    response_text = result.get("result", result.get("formatted_result", "No response"))

    return json.dumps(
        {
            "status": "received",
            "from": to_role,
            "to": my_role,
            "message_type": f"{message_type}_response",
            "response": response_text,
            "source": "agent",
        }
    )


if __name__ == "__main__":
    mcp.run()
