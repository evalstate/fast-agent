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
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


from fast_agent.spawn.config_reader import (
    _load_config,
    get_default_model,
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

    Provides execution context (task, workspace, depth).
    Behavioral rules come from the agent's instruction (template YAML).
    """
    lines = [
        "# Agent Context",
        "",
        "You are an AI agent. Follow your instruction to complete the assignment.",
        "",
        "## Your Assignment",
        task,
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
                    "- Read `team_roster.json` to see your team members",
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
            "## General",
            "- You cannot talk to the user directly",
            "- Be thorough and concise in your outputs",
            "- Include file paths for any files you created/modified",
            "",
        ]
    )

    if depth < max_depth - 1:
        lines.extend(
            [
                "## Sub-Agent Spawning",
                f"You are at depth {depth}/{max_depth}. "
                "You can spawn your own sub-agents if needed.",
            ]
        )
    else:
        lines.extend(
            [
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
    server_overrides: dict[str, dict] | None = None,
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
        # Read parent config to get server definitions directly
        parent_config = _load_config(project_dir)
        parent_servers = parent_config.get("mcp", {}).get("servers", {})

        config_lines.extend(["", "mcp:", "  servers:"])
        for srv in servers:
            parent_srv = parent_servers.get(srv, {}) if isinstance(parent_servers, dict) else {}
            if not isinstance(parent_srv, dict):
                continue

            # Read command and args DIRECTLY from parent config to preserve
            # argument integrity (avoid shlex.split which breaks quoted args
            # like '--header "Authorization: Bearer ${KEY}"')
            cmd = parent_srv.get("command", "")
            raw_args = parent_srv.get("args", [])
            srv_url = parent_srv.get("url", "")

            if not cmd and not srv_url:
                continue

            config_lines.append(f"    {srv}:")

            if cmd:
                config_lines.append(f'      command: "{cmd}"')

                # Apply workspace path substitutions on each arg individually
                project = str(Path(project_dir).resolve())
                is_team_workspace = workspace_dir and "/workspaces/" in workspace_dir

                # For team workspaces, filesystem server should serve just
                # the workspace root — not project-specific subdirs
                if is_team_workspace and srv == "filesystem":
                    resolved_args: list[str] = [str(a) for a in raw_args
                                     if not (str(a).startswith(".") or str(a).startswith("/"))]
                    resolved_args.append(workspace_dir)
                else:
                    # Check if template provides an override for this server's args
                    srv_override = (server_overrides or {}).get(srv, {})
                    override_args = srv_override.get("args") if isinstance(srv_override, dict) else None

                    # Use override args if provided, otherwise use parent args
                    source_args = override_args if override_args is not None else raw_args

                    resolved_args: list[str] = []
                    for a in source_args:
                        a_str = str(a)
                        # Substitute placeholders
                        a_str = a_str.replace("{workspace_dir}", workspace_dir)
                        a_str = a_str.replace("{project_dir}", project)
                        # Replace relative "." with workspace dir
                        if a_str == ".":
                            a_str = workspace_dir
                        elif a_str.startswith("./"):
                            a_str = f"{workspace_dir}{a_str[1:]}"
                        # Replace relative "servers/" paths with absolute project paths
                        if a_str.startswith("servers/"):
                            a_str = f"{project}/{a_str}"
                        resolved_args.append(a_str)

                # Add --directory for uv run commands
                if cmd == "uv" and "run" in resolved_args and "--directory" not in resolved_args:
                    run_idx = resolved_args.index("run")
                    resolved_args.insert(run_idx + 1, "--directory")
                    resolved_args.insert(run_idx + 2, project)

                if resolved_args:
                    args_str = ", ".join(f'"{a}"' for a in resolved_args)
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
    instruction = config.get("instruction", "You are a helpful team member.")
    context = config.get("context", "")
    servers = config.get("servers", [])
    model = config.get("model", "")
    depth = config.get("depth", 1)
    max_depth = config.get("max_depth", 3)
    parent_run_id = config.get("parent_run_id", "")
    role = config.get("role", "agent")
    agent_name = os.environ.get("TEAM_MY_NAME", "") or role
    skill_names = config.get("skills", [])

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
        server_overrides=config.get("server_overrides"),
    )

    # Emit started event for TUI
    event_run_id = parent_run_id or run_id
    lifecycle = config.get("lifecycle", "oneshot")
    team_name = config.get("team_name", "")
    emit_event(
        "started",
        event_run_id,
        agent_name,
        model=model or get_default_model(project_dir),
        servers=servers,
        lifecycle=lifecycle,
        team_name=team_name,
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
        from fast_agent.spawn.config_reader import get_skills

        # Convert skill names to SkillManifest objects using shared helper
        skills_dir = Path(project_dir) / ".fast-agent" / "skills"
        skill_manifests = get_skills(skills_dir, *skill_names) if skill_names else []
        logger.info("[SKILLS DEBUG] role=%s, skill_names=%s, manifests=%d", role, skill_names, len(skill_manifests))
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
            skills=skill_manifests,
        )
        async def child_main() -> str | None:
            # Phase 2: chdir to workspace_dir BEFORE fast.run()
            os.chdir(workspace_dir)

            async with fast.run() as agent:
                _install_tool_hooks(agent, event_run_id, agent_name)

                # Signal that agent is ready (MCP servers loaded, hooks installed)
                emit_event("agent_ready", event_run_id, agent_name)

                # Emit runtime config for monitoring dashboard
                _emit_runtime_config(agent, event_run_id, agent_name)
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

                # ── Keep-alive: wait for inbox messages via AgentChannel ──
                is_team_agent = bool(os.environ.get("TEAM_MY_NAME") and os.environ.get("TEAM_WORKSPACE"))

                if is_team_agent:
                    from fast_agent.spawn.agent_channel import AgentChannel
                    from fast_agent.spawn.message_bus import MessageBus

                    idle_timeout = int(os.environ.get("AGENT_IDLE_TIMEOUT", "7200"))
                    channel = AgentChannel(agent_name)
                    await channel.start_server()

                    # Resolve messages dir for inbox reads
                    _msgs_dir = os.environ.get("TEAM_MESSAGES_DIR", "")
                    if not _msgs_dir:
                        _ws = os.environ.get("TEAM_WORKSPACE", "")
                        if _ws:
                            _cur = Path(_ws)
                            while _cur != _cur.parent:
                                if _cur.name == ".runtime":
                                    _msgs_dir = str(_cur / "state" / "messages")
                                    break
                                _cur = _cur.parent

                    emit_event("idle", event_run_id, agent_name)
                    logger.info(
                        "📡 %s entering keep-alive mode (timeout=%ds)",
                        agent_name,
                        idle_timeout,
                    )

                    try:
                        while True:
                            signal = await channel.listen(timeout=idle_timeout)
                            if signal is None:
                                logger.info(
                                    "⏰ %s idle timeout (%ds) — exiting",
                                    agent_name,
                                    idle_timeout,
                                )
                                break

                            logger.info(
                                "📬 %s woke up on signal '%s'",
                                agent_name,
                                signal,
                            )

                            # Read inbox
                            if _msgs_dir:
                                bus = MessageBus(messages_dir=_msgs_dir)
                                unread = bus.read_unread(agent_name)
                            else:
                                unread = []

                            if not unread:
                                continue

                            # Format inbox messages as follow-up
                            inbox_lines = [
                                f"\n━━━ 📬 NEW MESSAGES ({len(unread)} unread) ━━━\n"
                            ]
                            for msg in unread:
                                inbox_lines.append(
                                    f"[{msg.message_type.upper()}] from {msg.from_name}:\n"
                                    f"  {msg.content}\n"
                                )
                                bus.mark_done(agent_name, msg.message_id)

                            inbox_lines.append(
                                "→ Handle these messages: take action and/or reply.\n"
                                "━━━━━━━━━━━━━━━━━━━━\n"
                            )
                            follow_up = "\n".join(inbox_lines)

                            emit_event(
                                "resumed",
                                event_run_id,
                                agent_name,
                                message_count=len(unread),
                            )
                            response = await agent.send(follow_up)
                            emit_event(
                                "idle",
                                event_run_id,
                                agent_name,
                            )
                    finally:
                        await channel.stop()

                return response

        response = await child_main()

        duration = time.time() - start_time
        emit_event(
            "result",
            event_run_id,
            agent_name,
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
        emit_event("error", event_run_id, agent_name, message=str(e))
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

# ── Runtime config emission ────────────────────────────────────────


def _emit_runtime_config(agent_app: Any, run_id: str, role: str) -> None:
    """Emit resolved runtime config from the live agent for dashboard monitoring.

    Reads the fully-initialized agent to extract:
    - resolved instruction (with skills + server instructions injected)
    - skill manifests (name + description)
    - per-server tool lists (name + description)
    """
    try:
        child_agent = agent_app["child"]
    except (KeyError, TypeError):
        return

    try:
        # Resolved instruction (after _apply_instruction_templates)
        resolved_instruction = getattr(child_agent, "_instruction", "") or ""

        # Skill manifests
        skills_data = []
        for m in getattr(child_agent, "_skill_manifests", []):
            skills_data.append({
                "name": m.name,
                "description": getattr(m, "description", "") or "",
            })

        # Per-server tool map
        tools_data = {}
        aggregator = getattr(child_agent, "_aggregator", None)
        if aggregator:
            server_tool_map = getattr(aggregator, "_server_to_tool_map", {})
            for server_name, namespaced_tools in server_tool_map.items():
                tools_data[server_name] = [
                    {
                        "name": nt.tool.name,
                        "description": getattr(nt.tool, "description", "") or "",
                    }
                    for nt in namespaced_tools
                ]

        emit_event(
            "runtime_config",
            run_id,
            role,
            resolved_instruction=resolved_instruction,
            skills=skills_data,
            tools=tools_data,
        )
    except Exception:
        pass  # Never crash the child for monitoring


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
                return s[:80]

            args_preview = ", ".join(
                f"{k}={_shorten(v)}" for k, v in list(tool_args.items())[:5]
            )

            # Serialize full args (with safety truncation for large values)
            def _safe_arg(v: object) -> object:
                s = str(v)
                return s[:2000] if len(s) > 2000 else v

            args_full = {k: _safe_arg(v) for k, v in tool_args.items()}

            _tool_start_times[tool_name] = time.time()
            emit_event(
                "tool_call",
                run_id,
                role,
                tool_name=tool_name,
                args_preview=args_preview,
                args_full=args_full,
            )

    async def after_tool_call(runner: Any, tool_message: Any) -> None:
        tool_results = getattr(tool_message, "tool_results", None) or {}
        now = time.time()
        for tool_name, start_t in list(_tool_start_times.items()):
            duration_ms = (now - start_t) * 1000
            status = "ok"
            is_error = False
            result_preview = ""

            # Extract result content from tool_results
            for _corr_id, result in tool_results.items():
                if getattr(result, "isError", False):
                    status = "error"
                    is_error = True
                # Extract text content from result
                content_list = getattr(result, "content", None) or []
                for content_item in content_list:
                    text = getattr(content_item, "text", None)
                    if text:
                        result_preview += text[:500]
                        break  # take first text content

            emit_event(
                "tool_result",
                run_id,
                role,
                tool_name=tool_name,
                status=status,
                duration_ms=round(duration_ms, 1),
                result_preview=result_preview[:500],
                is_error=is_error,
            )
        _tool_start_times.clear()

    # ── Centralized activity hooks: thinking + response ──────────────

    async def spawn_before_llm(runner: Any, messages: Any) -> None:
        """Emit 'thinking' event before each LLM call."""
        model = ""
        rp = getattr(runner, "request_params", None)
        if rp:
            model = getattr(rp, "model", "") or ""
        emit_event("thinking", run_id, role, model=model)

    async def spawn_after_llm(runner: Any, message: Any) -> None:
        """Emit 'response' event after each LLM reply, including reasoning."""
        import re
        from fast_agent.mcp.helpers.content_helpers import get_text

        # Extract stop reason
        stop_reason = str(getattr(message, "stop_reason", "") or "")

        # Extract response text from content blocks
        text = ""
        if hasattr(message, "first_text"):
            raw = message.first_text() or ""
            # filter out the "<no text>" sentinel from PromptMessageExtended
            text = "" if raw == "<no text>" else raw
        else:
            content = getattr(message, "content", None) or []
            for item in content:
                t = get_text(item)
                if t:
                    text = t
                    break

        # Strip <think>...</think> tags that qwen3 models embed in content
        if text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Extract model reasoning from channels (Anthropic/OpenAI/Kimi)
        reasoning_text = ""
        channels = getattr(message, "channels", None) or {}
        reasoning_blocks = channels.get("reasoning", [])
        for block in reasoning_blocks:
            t = get_text(block)
            if t:
                reasoning_text = t[:2000]
                break

        # Fallback: if no text but reasoning exists (e.g. qwen3 thinking-only),
        # use a truncated version of reasoning as the response text
        if not text and reasoning_text and "TOOL_USE" not in stop_reason:
            # Strip think tags from reasoning too
            clean_reasoning = re.sub(
                r"<think>|</think>", "", reasoning_text
            ).strip()
            if clean_reasoning:
                text = clean_reasoning[:500] + ("..." if len(clean_reasoning) > 500 else "")

        # Truncate final text
        text = text[:1000] if text else ""

        # Skip emitting response for pure TOOL_USE with no text
        # (tool_call events already cover this use case)
        if "TOOL_USE" in stop_reason and not text:
            pass  # still emit token_usage below
        else:
            emit_event(
                "response", run_id, role,
                text=text,
                reasoning=reasoning_text,
                stop_reason=stop_reason,
            )

        # ── Emit token_usage event for cost tracking ──
        try:
            _agent = getattr(runner, "_agent", None)
            _acc = getattr(_agent, "usage_accumulator", None) if _agent else None
            if _acc and _acc.turns:
                _last = _acc.turns[-1]
                _cache = getattr(_last, "cache_usage", None)
                emit_event(
                    "token_usage", run_id, role,
                    model=getattr(_last, "model", "") or "",
                    input_tokens=getattr(_last, "input_tokens", 0),
                    output_tokens=getattr(_last, "output_tokens", 0),
                    total_tokens=getattr(_last, "total_tokens", 0),
                    cache_hit_tokens=getattr(_cache, "cache_hit_tokens", 0) if _cache else 0,
                    cache_read_tokens=getattr(_cache, "cache_read_tokens", 0) if _cache else 0,
                    cache_write_tokens=getattr(_cache, "cache_write_tokens", 0) if _cache else 0,
                    reasoning_tokens=getattr(_last, "reasoning_tokens", 0),
                )
        except Exception:
            pass  # never crash child for token tracking

    # RTAC: Real-time Agent Communication — inbox watcher hook
    rtac_before_llm: Any = None
    try:
        from fast_agent.spawn.inbox_watcher_hook import create_inbox_watcher

        watcher = create_inbox_watcher()
        if watcher is not None:
            rtac_before_llm = watcher.before_llm_call
    except Exception:
        pass  # RTAC is optional — don't break spawn if it fails

    # Pause/Resume: signal-based pause checkpoint hook
    pause_before_llm: Any = None
    try:
        from fast_agent.spawn.pause_signal_handler import pause_checkpoint
        pause_before_llm = pause_checkpoint
    except Exception:
        pass  # Pause is optional — don't break spawn if it fails

    # ── Merge all hooks: spawn events + RTAC + card-defined hooks ──

    existing = child_agent.tool_runner_hooks
    if existing is not None:
        orig_before_tool = existing.before_tool_call
        orig_after_tool = existing.after_tool_call
        orig_before_llm = existing.before_llm_call
        orig_after_llm = existing.after_llm_call

        async def merged_before_tool(runner: Any, request: Any) -> None:
            if orig_before_tool:
                await orig_before_tool(runner, request)
            await before_tool_call(runner, request)

        async def merged_after_tool(runner: Any, result: Any) -> None:
            if orig_after_tool:
                await orig_after_tool(runner, result)
            await after_tool_call(runner, result)

        async def merged_before_llm(runner: Any, messages: Any) -> None:
            await spawn_before_llm(runner, messages)
            if orig_before_llm:
                await orig_before_llm(runner, messages)
            if rtac_before_llm:
                await rtac_before_llm(runner, messages)
            if pause_before_llm:
                await pause_before_llm(runner, messages)

        async def merged_after_llm(runner: Any, message: Any) -> None:
            await spawn_after_llm(runner, message)
            if orig_after_llm:
                await orig_after_llm(runner, message)

        child_agent.tool_runner_hooks = ToolRunnerHooks(
            before_llm_call=merged_before_llm,
            after_llm_call=merged_after_llm,
            before_tool_call=merged_before_tool,
            after_tool_call=merged_after_tool,
            after_turn_complete=existing.after_turn_complete,
        )
    else:
        # Build before_llm chain: spawn → rtac → pause
        _blm_fns = [spawn_before_llm]
        if rtac_before_llm:
            _blm_fns.append(rtac_before_llm)
        if pause_before_llm:
            _blm_fns.append(pause_before_llm)

        async def _chained_before_llm(r: Any, m: Any) -> None:
            for fn in _blm_fns:
                await fn(r, m)

        child_agent.tool_runner_hooks = ToolRunnerHooks(
            before_llm_call=_chained_before_llm,
            after_llm_call=spawn_after_llm,
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
        )


async def _chain_before_llm(spawn_fn: Any, rtac_fn: Any, runner: Any, messages: Any) -> None:
    """Chain spawn_before_llm and RTAC before_llm hooks."""
    await spawn_fn(runner, messages)
    await rtac_fn(runner, messages)


async def main() -> None:
    """CLI entrypoint for running as a subprocess."""
    # Install pause signal handlers before anything else
    try:
        from fast_agent.spawn.pause_signal_handler import install_pause_signal_handlers
        install_pause_signal_handlers(asyncio.get_event_loop())
    except Exception:
        pass  # Pause is optional

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
