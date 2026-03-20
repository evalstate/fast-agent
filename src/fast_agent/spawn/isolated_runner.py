"""Isolated Agent Runner — standalone child FastAgent entrypoint.

This script runs as a SEPARATE PROCESS, spawned by the orchestrator.
It receives a handoff config JSON, creates a temporary FastAgent instance,
runs the task, and writes the result JSON.

Usage::

    uv run python -m fast_agent.spawn.isolated_runner --config /tmp/run_<uuid>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


from fast_agent.spawn.config_reader import (
    _load_config,
    get_default_model,
    get_server_commands,
    get_server_env,
)
from fast_agent.spawn.runtime_paths import get_runtime_paths
from fast_agent.spawn.spawn_events import emit_event

logger = logging.getLogger(__name__)


def build_child_system_prompt(
    task: str,
    context: str = "",
    depth: int = 1,
    max_depth: int = 3,
    workspace_dir: str = "",
    has_filesystem: bool = False,
    team_workspace: str = "",
) -> str:
    """Build system prompt for the child agent.

    Inspired by OpenClaw's buildSubagentSystemPrompt().
    """
    lines = [
        "# Subagent Context",
        "",
        "You are a **sub-agent** spawned to handle a specific task.",
        "",
        "## Your Role",
        f"- You were created to handle: {task}",
        "- Follow your instruction to complete this task.",
        "",
    ]

    if context and context.strip():
        lines.extend(
            [
                "## Context from Parent",
                context.strip(),
                "",
            ]
        )

    # Workspace awareness (only when filesystem server is available)
    if has_filesystem and workspace_dir:
        # Check if this is a team workspace (inside .runtime/data/workspaces/)
        is_team_workspace = "/workspaces/" in workspace_dir
        if is_team_workspace:
            lines.extend(
                [
                    "## Workspace & File Access",
                    f"Your workspace root is: `{workspace_dir}`",
                    "The filesystem server is scoped to this directory.",
                    "",
                    "### Rules",
                    "- Write ALL output files inside this workspace (use relative paths like `src/`, `docs/`, `tests/`)",
                    "- You CANNOT access files outside this workspace",
                    "- Read the ROSTER.md file to see your team members",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "## Workspace & File Access",
                    f"Your filesystem server root is: `{workspace_dir}`",
                    "All filesystem tool paths are relative to this root.",
                    "",
                    "### Free Zones (read/write freely — no confirmation needed)",
                    "- `.runtime/` — agent runtime data",
                    "  - `.runtime/state/` — persistent (signals, messages, runs)",
                    "  - `.runtime/cache/` — ephemeral (tmp, logs)",
                    "  - `.runtime/data/` — output (agent_cards, workspaces)",
                ]
            )
            if team_workspace:
                lines.append(f"- `{team_workspace}` — team shared workspace")
            lines.extend(
                [
                    "",
                    "### Protected Zones (read OK, modify only if your task requires it)",
                    "- Source code (`*.py`), configs (`*.yaml`), templates",
                    "",
                    "### Forbidden",
                    "- `fastagent.secrets.yaml` — never read or modify",
                    "",
                    "### Path Convention",
                    "- Use relative paths from project root (e.g., `.runtime/data/agent_cards/`)",
                    "",
                ]
            )

    lines.extend(
        [
            "## Rules",
            "1. **Stay focused** — Do your assigned task, nothing else",
            "2. **Be thorough** — Your output is the ONLY thing the orchestrator sees",
            "3. **Be concise** — Include all relevant info, skip filler",
            "4. **Don't initiate** — No proactive actions, no side quests",
            "5. **No user interaction** — You cannot talk to the user directly",
            "",
            "## Output Format",
            "When complete, respond with:",
            "- What you accomplished or found",
            "- Any relevant details the orchestrator should know",
            "- File paths for any files you created/modified",
            "- Keep it concise but informative",
        ]
    )

    if depth < max_depth - 1:
        lines.extend(
            [
                "",
                "## Sub-Agent Spawning",
                f"You are at depth {depth}/{max_depth}. "
                "You can spawn your own sub-agents if needed.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Depth Limit",
                f"You are at depth {depth}/{max_depth}. You CANNOT spawn further sub-agents.",
            ]
        )

    return "\n".join(lines)


def create_child_config(
    project_dir: str | Path,
    workspace_dir: str,
    servers: list[str],
    model: str = "",
    depth: int = 1,
    run_id: str = "",
    agent_name: str = "",
) -> str:
    """Create a temporary fastagent.config.yaml for the child.

    Returns the path to the temp directory containing the config.
    """
    paths = get_runtime_paths(project_dir)

    # Use project-local .runtime/cache/tmp/ dir instead of system /tmp
    project_tmp = paths["tmp"] / "child_configs"
    project_tmp.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="fastagent_child_", dir=str(project_tmp))

    # Centralized log dir in .runtime/cache/logs/
    logs_dir = str(paths["logs"])
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = f"child_d{depth}_{run_id or 'unknown'}.jsonl"

    # Build config YAML
    config_lines: list[str] = []

    if model:
        config_lines.append(f"default_model: {model}")
    else:
        config_lines.append(f"default_model: {get_default_model(project_dir)}")

    config_lines.extend(
        [
            "",
            "logger:",
            "  type: file",
            f"  path: {logs_dir}/{log_filename}",
            "  level: info",
            "  truncate_tools: true",
        ]
    )

    if servers:
        # Read parent config to get non-command fields (url, env)
        parent_config = _load_config(project_dir)
        parent_servers = parent_config.get("mcp", {}).get("servers", {})
        # server_commands has workspace-substituted command strings
        server_commands = get_server_commands(project_dir, workspace_dir)

        config_lines.extend(["", "mcp:", "  servers:"])
        for srv in servers:
            if srv not in server_commands:
                continue

            config_lines.append(f"    {srv}:")
            parent_srv = parent_servers.get(srv, {}) if isinstance(parent_servers, dict) else {}

            # Parse command/args from server_commands string (has correct paths)
            cmd_str = server_commands[srv]
            parts = shlex.split(cmd_str)
            if parts:
                config_lines.append(f'      command: "{parts[0]}"')
                if len(parts) > 1:
                    args_str = ", ".join(f'"{a}"' for a in parts[1:])
                    config_lines.append(f"      args: [{args_str}]")

            # URL from parent config (not in command string)
            if isinstance(parent_srv, dict):
                url = parent_srv.get("url", "")
                if url:
                    config_lines.append(f'      url: "{url}"')

            # Merge env: parent config env + team-aware env
            parent_env = (parent_srv.get("env", {}) or {}) if isinstance(parent_srv, dict) else {}
            srv_env = get_server_env(srv, workspace_dir, agent_name=agent_name) or {}
            merged_env = {**parent_env, **srv_env}
            if merged_env:
                config_lines.append("      env:")
                for k, v in merged_env.items():
                    config_lines.append(f'        {k}: "{v}"')

    config_content = "\n".join(config_lines) + "\n"
    config_path = os.path.join(temp_dir, "fastagent.config.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)

    # Copy secrets as-is. Since child now uses mcp.servers format (same as
    # parent), deep_merge correctly merges secrets' env into server entries
    # without losing command/args.
    project_root = str(Path(project_dir).resolve())
    secrets_src = os.path.join(workspace_dir, "fastagent.secrets.yaml")
    if not os.path.exists(secrets_src):
        secrets_src = os.path.join(project_root, "fastagent.secrets.yaml")
    secrets_dst = os.path.join(temp_dir, "fastagent.secrets.yaml")
    if os.path.exists(secrets_src):
        shutil.copy2(secrets_src, secrets_dst)

    return temp_dir


async def run_child_agent(
    config: dict[str, Any],
    project_dir: str | Path,
) -> dict[str, Any]:
    """Create and run a FastAgent child for a single task.

    Args:
        config: Handoff config dict.
        project_dir: Root directory of the host application.

    Returns:
        Result dict with status, result, summary, etc.
    """
    task = config["task"]
    instruction = config.get("instruction", "You are a helpful sub-agent.")
    context = config.get("context", "")
    servers = config.get("servers", [])
    model = config.get("model", "")
    depth = config.get("depth", 1)
    max_depth = config.get("max_depth", 3)
    parent_run_id = config.get("parent_run_id", "")
    role = config.get("role", "agent")
    skills = config.get("skills", [])

    # Build system prompt with workspace awareness
    system_prompt = build_child_system_prompt(
        task,
        context,
        depth,
        max_depth,
        workspace_dir=config.get("workspace_dir", ""),
        has_filesystem="filesystem" in servers,
        team_workspace=config.get("team_workspace", ""),
    )
    full_instruction = f"{instruction}\n\n{{{{agentSkills}}}}\n\n{system_prompt}"

    # Create temp config directory
    import uuid as _uuid

    run_id = _uuid.uuid4().hex[:8]
    workspace_dir = config.get("workspace_dir", str(project_dir))
    temp_dir = create_child_config(
        project_dir=project_dir,
        workspace_dir=workspace_dir,
        servers=servers,
        model=model,
        depth=depth,
        run_id=run_id,
        agent_name=os.environ.get("TEAM_MY_NAME", ""),
    )

    # Emit started event for TUI
    event_run_id = parent_run_id or run_id
    emit_event(
        "started",
        event_run_id,
        role,
        model=model or get_default_model(project_dir),
        servers=servers,
    )

    start_time = time.time()
    result: dict[str, Any] = {
        "status": "error",
        "result": "",
        "summary": "",
        "artifacts": [],
        "metadata": {},
        "error": None,
    }

    original_dir = os.getcwd()
    try:
        # Phase 1: chdir to temp_dir so FastAgent loads the child config
        os.chdir(temp_dir)

        # Import FastAgent here (after chdir so it picks up the right config)
        from fast_agent import FastAgent

        logger.info("[SKILLS DEBUG] role=%s, skills_paths=%s", role, skills)
        logger.info(
            "[SKILLS DEBUG] instruction has {agentSkills}: %s",
            "{agentSkills}" in full_instruction,
        )

        fast = FastAgent("Isolated Child Agent")

        # Resume support: load previous conversation history if available
        history_file = config.get("history_file")

        @fast.agent(
            name="child",
            instruction=full_instruction,
            servers=servers if servers else [],
            skills=skills if skills else [],
        )
        async def child_main() -> str | None:
            # Phase 2: chdir to workspace_dir BEFORE fast.run()
            os.chdir(workspace_dir)

            async with fast.run() as agent:
                _install_tool_hooks(agent, event_run_id, role)

                # If resuming, load previous session history into agent
                # This uses FastAgent's native API — no LLM call, just
                # restores message_history so the next send() has full context
                if history_file and Path(history_file).exists():
                    from fast_agent.mcp.prompts.prompt_load import (
                        load_history_into_agent,
                    )

                    try:
                        load_history_into_agent(agent["child"], Path(history_file))
                        logger.info(
                            "📂 Loaded previous history from %s",
                            history_file,
                        )
                    except Exception as exc:
                        logger.warning(
                            "⚠️ Failed to load history: %s — continuing fresh",
                            exc,
                        )

                response = await agent.send(task)
                return response

        response = await child_main()

        duration = time.time() - start_time
        emit_event(
            "result",
            event_run_id,
            role,
            summary=f"Task completed in {duration:.1f}s",
            duration_seconds=round(duration, 1),
        )
        result = {
            "status": "completed",
            "result": str(response) if response else "(no output)",
            "summary": f"Task completed in {duration:.1f}s",
            "artifacts": [],
            "metadata": {
                "model_used": model or get_default_model(project_dir),
                "duration_seconds": round(duration, 1),
            },
            "error": None,
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error("Child agent failed: %s", e, exc_info=True)
        emit_event("error", event_run_id, role, message=str(e))
        result = {
            "status": "error",
            "result": "",
            "summary": "Child agent execution failed",
            "artifacts": [],
            "metadata": {"duration_seconds": round(duration, 1)},
            "error": str(e),
        }

    finally:
        os.chdir(original_dir)
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    return result


# ── Tool-call event hooks ──────────────────────────────────────────


def _install_tool_hooks(agent_app: Any, run_id: str, role: str) -> None:
    """Set ToolRunnerHooks on the child agent for TUI event emission.

    Uses fast-agent's built-in hook mechanism — zero fork changes needed.
    """
    try:
        child_agent = agent_app["child"]
    except (KeyError, TypeError):
        return

    if not hasattr(child_agent, "tool_runner_hooks"):
        return

    from fast_agent.agents.tool_runner import ToolRunnerHooks

    _tool_start_times: dict[str, float] = {}

    async def before_tool_call(runner: Any, request: Any) -> None:
        tool_calls = getattr(request, "tool_calls", None) or {}
        cwd = os.getcwd()
        for _corr_id, tool_request in tool_calls.items():
            tool_name = tool_request.params.name
            tool_args = tool_request.params.arguments or {}

            # Shorten absolute paths: replace workspace root with ./
            def _shorten(v: object) -> str:
                s = str(v)
                if cwd and s.startswith(cwd):
                    s = "." + s[len(cwd):]
                return s[:40]

            args_preview = ", ".join(
                f"{k}={_shorten(v)}" for k, v in list(tool_args.items())[:3]
            )
            _tool_start_times[tool_name] = time.time()
            emit_event(
                "tool_call",
                run_id,
                role,
                tool_name=tool_name,
                args_preview=args_preview,
            )

    async def after_tool_call(runner: Any, tool_message: Any) -> None:
        tool_results = getattr(tool_message, "tool_results", None) or {}
        now = time.time()
        for tool_name, start_t in list(_tool_start_times.items()):
            duration_ms = (now - start_t) * 1000
            status = "ok"
            for _corr_id, result in tool_results.items():
                if getattr(result, "isError", False):
                    status = "error"
                    break
            emit_event(
                "tool_result",
                run_id,
                role,
                tool_name=tool_name,
                status=status,
                duration_ms=round(duration_ms, 1),
            )
        _tool_start_times.clear()

    existing = child_agent.tool_runner_hooks
    if existing is not None:
        orig_before = existing.before_tool_call
        orig_after = existing.after_tool_call

        async def merged_before(runner: Any, request: Any) -> None:
            if orig_before:
                await orig_before(runner, request)
            await before_tool_call(runner, request)

        async def merged_after(runner: Any, result: Any) -> None:
            if orig_after:
                await orig_after(runner, result)
            await after_tool_call(runner, result)

        child_agent.tool_runner_hooks = ToolRunnerHooks(
            before_llm_call=existing.before_llm_call,
            after_llm_call=existing.after_llm_call,
            before_tool_call=merged_before,
            after_tool_call=merged_after,
            after_turn_complete=existing.after_turn_complete,
        )
    else:
        child_agent.tool_runner_hooks = ToolRunnerHooks(
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
        )


async def main() -> None:
    """CLI entrypoint for running as a subprocess."""
    parser = argparse.ArgumentParser(description="Isolated Agent Runner")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to handoff config JSON file",
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project root directory",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"Config file not found: {config_path}",
                }
            )
        )
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    result = await run_child_agent(config, project_dir=args.project_dir)

    result_file = config.get("result_file")
    if result_file:
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
