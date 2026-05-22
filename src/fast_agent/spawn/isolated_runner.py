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
import atexit
import json
import logging
import os
import shutil
import signal as _signal
import sys
import tempfile
import time
import yaml
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


# Keep-alive listen timeout. Each tick: wake-signal channel listens for this
# many seconds, then falls through to poll the inbox for messages whose
# signal was dropped (e.g. producer sent ``wake`` before this consumer's
# AgentChannel socket was bound). Lower = faster recovery from signal loss
# but more idle CPU wake-ups; higher = lighter idle load but slower
# fallback. 30s balances both — verified against the 1h2m Sasha hang
# regression (signal lost during initial ``agent.send``).
KEEP_ALIVE_TIMEOUT_S = 30.0


def _install_termination_cleanup(
    run_id: str,
    agent_name: str,
    channel_sock_path: Path | None,
) -> None:
    """Install atexit + SIGTERM hooks so abnormal exits still:

    * Emit a final ``killed`` event so the bridge / DB status flips
      away from ``running``.
    * Unlink the agent's own channel socket file so the next instance
      doesn't find a stale orphan (which would otherwise fool
      ``send_signal`` into a wasted connect attempt).

    Limitations:
      - SIGKILL bypasses everything Python can install. There is no
        defense; the parent backend's escalation from SIGTERM → SIGKILL
        (after a 5s grace) will leak a sock file. The DB-level
        observability fix (Obs-1, compute effective status from
        liveness probe) compensates for that worst case.
    """
    cleaned = {"done": False}  # idempotency latch (atexit + signal both fire)

    def _cleanup() -> None:
        if cleaned["done"]:
            return
        cleaned["done"] = True

        # Best-effort: emit final lifecycle event. If the bridge socket
        # has died (R1/R2 race) this will silently no-op — that's fine,
        # Obs-1 reconciliation picks it up via the snapshot-trigger
        # / liveness-probe path.
        try:
            emit_event(
                "killed",
                run_id,
                agent_name,
                message="Subprocess exiting (SIGTERM or shutdown)",
            )
        except Exception:
            pass

        # Best-effort: unlink our own channel sock so the next spawn
        # of this agent name doesn't trip over a stale file.
        if channel_sock_path is not None:
            try:
                if channel_sock_path.exists():
                    channel_sock_path.unlink()
            except OSError:
                pass

    atexit.register(_cleanup)

    def _sigterm_handler(signum, frame):  # noqa: ARG001 — signature fixed
        # Convert SIGTERM → SystemExit so atexit fires. Default Python
        # behavior is to terminate without running atexit, which leaves
        # the channel sock file orphan and the DB status stuck at
        # "running" forever — exactly the bug this hook prevents.
        sys.exit(0)

    try:
        _signal.signal(_signal.SIGTERM, _sigterm_handler)
    except (ValueError, OSError):
        # Not on main thread, or platform doesn't support signal.signal —
        # atexit alone still handles normal exit / SystemExit paths.
        pass


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
                    "The filesystem server is scoped to this workspace and allowed project directories (e.g. skills).",
                    "",
                    "### Rules",
                    "- Write ALL output files inside your workspace (use relative paths like `src/`, `docs/`, `tests/`)",
                    "- You can read skill files from the shared skills directory",
                    "- You CANNOT write files outside your workspace",
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

    Strategy: load parent config as dict → resolve all relative paths to
    absolute → filter to requested servers → apply overrides → yaml.dump().

    Returns the path to the temp directory containing the config.
    """
    paths = get_runtime_paths(project_dir)
    project = str(Path(project_dir).resolve())

    # Use project-local .runtime/cache/tmp/ dir instead of system /tmp
    project_tmp = paths["tmp"] / "child_configs"
    project_tmp.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="fastagent_child_", dir=str(project_tmp))

    # Centralized log dir in .runtime/cache/logs/
    logs_dir = str(paths["logs"])
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = f"child_d{depth}_{run_id or 'unknown'}.jsonl"

    # ── Step 1: Load parent config ──
    parent_config = _load_config(project_dir)
    parent_servers = parent_config.get("mcp", {}).get("servers", {})

    # ── Step 2: Build child config dict ──
    child_config: dict[str, Any] = {
        "default_model": model if model else get_default_model(project_dir),
        "logger": {
            "type": "file",
            "path": f"{logs_dir}/{log_filename}",
            "level": "info",
            "truncate_tools": True,
        },
    }

    if servers:
        child_servers: dict[str, Any] = {}
        is_team_workspace = workspace_dir and "/workspaces/" in workspace_dir

        for srv in servers:
            parent_srv = parent_servers.get(srv, {})
            if not isinstance(parent_srv, dict):
                continue

            # Deep copy to avoid mutating parent config
            srv_cfg = dict(parent_srv)

            # Apply server_overrides on top of parent config
            srv_override = (server_overrides or {}).get(srv, {})
            if isinstance(srv_override, dict):
                for key, val in srv_override.items():
                    srv_cfg[key] = val

            cmd = srv_cfg.get("command", "")
            if not cmd and not srv_cfg.get("url"):
                continue

            # ── Resolve args: placeholders + relative paths → absolute ──
            raw_args = srv_cfg.get("args", [])
            if raw_args:
                resolved_args: list[str] = []
                for a in raw_args:
                    a_str = str(a)
                    # Substitute placeholders
                    a_str = a_str.replace("{workspace_dir}", workspace_dir or "")
                    a_str = a_str.replace("{project_dir}", project)
                    # "." → workspace dir
                    if a_str == ".":
                        a_str = workspace_dir
                    elif a_str.startswith("./"):
                        a_str = f"{workspace_dir}{a_str[1:]}"
                    # Resolve relative paths that exist under project_dir
                    if (
                        not a_str.startswith("/")
                        and not a_str.startswith("-")
                        and not a_str.startswith("$")
                        and not a_str.startswith("{")
                        and not a_str.startswith("@")
                        and ("/" in a_str or a_str.endswith(".py"))
                    ):
                        candidate = os.path.join(project, a_str)
                        if os.path.exists(candidate):
                            a_str = candidate
                    resolved_args.append(a_str)

                # For uv run: ensure --directory points to project
                if cmd == "uv" and "run" in resolved_args:
                    if "--directory" in resolved_args:
                        dir_idx = resolved_args.index("--directory")
                        if dir_idx + 1 < len(resolved_args):
                            dir_val = resolved_args[dir_idx + 1]
                            if not dir_val.startswith("/"):
                                abs_dir = os.path.join(project, dir_val)
                                if os.path.isdir(abs_dir):
                                    resolved_args[dir_idx + 1] = abs_dir
                    else:
                        run_idx = resolved_args.index("run")
                        resolved_args.insert(run_idx + 1, "--directory")
                        resolved_args.insert(run_idx + 2, project)

                srv_cfg["args"] = resolved_args

            # ── Resolve env values: ${VAR} + relative paths → absolute ──
            parent_env = parent_srv.get("env", {}) or {}
            override_env = (srv_override if isinstance(srv_override, dict) else {}).get("env", {}) or {}
            team_env = get_server_env(srv, workspace_dir, agent_name=agent_name) or {}
            merged_env = {**parent_env, **override_env, **team_env}

            if merged_env:
                import re
                resolved_env: dict[str, str] = {}
                for k, v in merged_env.items():
                    v_str = str(v)
                    # Resolve ${VAR} from current process env
                    if "${" in v_str:
                        v_str = re.sub(
                            r'\$\{(\w+)\}',
                            lambda m: os.environ.get(m.group(1), m.group(0)),
                            v_str,
                        )
                    # Resolve relative file/dir paths to absolute
                    if (
                        not v_str.startswith("/")
                        and not v_str.startswith("$")
                        and not v_str.startswith("http")
                        and ("/" in v_str or v_str.endswith(".db") or v_str.endswith(".py"))
                    ):
                        candidate = os.path.join(project, v_str)
                        if os.path.exists(candidate):
                            v_str = candidate
                    resolved_env[k] = v_str
                    # Detect unresolved ${VAR} — indicates missing env or secrets
                    if "${" in v_str:
                        logger.warning(
                            "[SPAWN-CONFIG] Unresolved env var %s=%s for "
                            "server '%s'. Ensure the value is provided in "
                            "fastagent.secrets.yaml or os.environ.",
                            k, v_str, srv,
                        )
                srv_cfg["env"] = resolved_env

            child_servers[srv] = srv_cfg


        if child_servers:
            child_config["mcp"] = {"servers": child_servers}

    # ── Step 3: Write config YAML ──
    config_path = os.path.join(temp_dir, "fastagent.config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(child_config, f, default_flow_style=False, allow_unicode=True)

    # Copy secrets as-is — FastAgent's deep_merge will overlay secret env values
    project_root = str(Path(project_dir).resolve())
    secrets_src = os.path.join(workspace_dir, "fastagent.secrets.yaml")
    if not os.path.exists(secrets_src):
        secrets_src = os.path.join(project_root, "fastagent.secrets.yaml")
    secrets_dst = os.path.join(temp_dir, "fastagent.secrets.yaml")
    if os.path.exists(secrets_src):
        shutil.copy2(secrets_src, secrets_dst)

        # ── Validation: detect secrets overriding resolved paths ──
        # When deep_merge(child_config, secrets) runs at FastAgent startup,
        # secrets that re-declare 'command' or 'args' will REPLACE the
        # absolute paths we resolved above (e.g. --directory), causing
        # MCP server startup failures in temp dir context.
        try:
            with open(secrets_src) as _sf:
                secrets_data = yaml.safe_load(_sf) or {}
            secrets_servers = secrets_data.get("mcp", {}).get("servers", {})
            for srv_name, srv_cfg in (secrets_servers or {}).items():
                if not isinstance(srv_cfg, dict):
                    continue
                overrides = []
                if "command" in srv_cfg:
                    overrides.append("command")
                if "args" in srv_cfg:
                    overrides.append("args")
                if overrides and srv_name in (child_servers if servers else {}):
                    logger.warning(
                        "[SPAWN-CONFIG] fastagent.secrets.yaml re-declares %s "
                        "for server '%s'. This will OVERRIDE resolved absolute "
                        "paths in child config via deep_merge, potentially "
                        "breaking the MCP server in spawn context. "
                        "Fix: only declare 'env' in secrets, not command/args.",
                        overrides,
                        srv_name,
                    )
        except Exception:
            pass  # best-effort validation, don't block spawn

    return temp_dir



async def _save_agent_context_snapshot(
    agent: Any,
    run_id: str,
    agent_name: str,
    trigger: str,
) -> None:
    """Save agent context window to SQLite — never raises.

    Uses lazy imports so isolated_runner works standalone without
    the Jarvis backend's core.database module.
    Uses SPAWN_REGISTRY_DB env var (absolute path) for DB access —
    safe even after os.chdir() to workspace.
    """
    try:
        import os as _os
        import sys as _sys
        from services.context_persistence import save_agent_context

        # Extract the actual child agent from AgentApp container.
        # AgentApp supports __getitem__ but is NOT a dict — so the old
        # isinstance(agent, dict) check always returned False, passing
        # the AgentApp itself which has no message_history.
        child_agent = agent
        try:
            child_agent = agent["child"]
        except (KeyError, TypeError):
            pass  # Already a raw agent or doesn't support __getitem__

        _sys.stderr.write(
            f"@@CONTEXT@@ Saving context for {agent_name} "
            f"trigger={trigger} has_history={hasattr(child_agent, 'message_history')} "
            f"msg_count={len(getattr(child_agent, 'message_history', []) or [])}\n"
        )
        _sys.stderr.flush()

        await save_agent_context(
            child_agent,
            run_id,
            trigger=trigger,
            agent_name=agent_name,
            session_id=_os.environ.get("TEAM_SESSION_ID"),
            team_name=_os.environ.get("TEAM_MY_ROLE", ""),
        )
    except Exception as _ctx_exc:
        import sys as _sys
        _sys.stderr.write(f"@@CONTEXT@@ Save FAILED ({trigger}): {_ctx_exc}\n")
        _sys.stderr.flush()
        logger.warning("[CONTEXT] Save failed (%s): %s", trigger, _ctx_exc)


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
                await _emit_mcp_status(agent, event_run_id, agent_name)
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

                # ── Unified inbox-driven lifetime loop ──
                # The agent's lifetime is a single loop over a persistent
                # inbox queue. Wake signal is an *optimization* to skip
                # polling — never a source of truth. State of "is there
                # work?" lives in the inbox file; we drain it on every
                # iteration so a race-lost signal cannot strand a message.
                #
                # See ROOT_CAUSE.md (or git blame) for the 1h2m hang that
                # motivated this refactor: producer fired wake while the
                # consumer was still in its first agent.send, sock not yet
                # bound — signal dropped silently, inbox sat unread until
                # an external poke arrived an hour later.
                _team_name_env = os.environ.get("TEAM_MY_NAME", "")
                _team_ws_env = os.environ.get("TEAM_WORKSPACE", "")
                is_team_agent = bool(_team_name_env and _team_ws_env)

                channel = None
                bus = None
                if is_team_agent:
                    from fast_agent.spawn.agent_channel import AgentChannel
                    from fast_agent.spawn.message_bus import MessageBus

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

                    channel = AgentChannel(agent_name)
                    await channel.start_server()
                    if _msgs_dir:
                        bus = MessageBus(messages_dir=_msgs_dir)

                    # Install SIGTERM / atexit hooks so an abnormal
                    # shutdown (backend restart, timeout cancel) still
                    # emits a terminal lifecycle event and unlinks our
                    # own channel sock. Without this, the path-aware
                    # signal handler bug surfaces as: DB status stuck
                    # on "running" + orphan sock file fooling future
                    # send_signal() calls into spurious connect attempts.
                    _install_termination_cleanup(
                        run_id=event_run_id,
                        agent_name=agent_name,
                        channel_sock_path=channel.socket_path,
                    )

                    logger.info(
                        "📡 %s entering keep-alive mode (timeout=%.1fs)",
                        agent_name, KEEP_ALIVE_TIMEOUT_S,
                    )

                # Initial task is the first "pending" item. After it runs,
                # subsequent iterations drain the inbox.
                pending: str | None = task
                pending_msg_count = 0
                is_first_iter = True
                last_was_timeout = False
                response = None

                try:
                    while True:
                        # 1) Process pending input (initial task OR inbox batch)
                        if pending is not None:
                            if not is_first_iter:
                                emit_event(
                                    "resumed",
                                    event_run_id,
                                    agent_name,
                                    message_count=pending_msg_count,
                                )

                            try:
                                response = await agent.send(pending)
                            except Exception as _send_exc:
                                import sys, traceback
                                logger.error(
                                    "[AGENT.SEND CRASH] agent=%s error=%s",
                                    agent_name, _send_exc,
                                )
                                traceback.print_exc(file=sys.stderr)
                                sys.stderr.flush()
                                raise

                            await _save_agent_context_snapshot(
                                agent,
                                event_run_id,
                                agent_name,
                                "task_complete" if is_first_iter else "idle",
                            )

                            pending = None
                            pending_msg_count = 0

                            if is_team_agent:
                                emit_event("idle", event_run_id, agent_name)

                            is_first_iter = False

                        # 2) Non-team agent: one-shot, exit after initial task
                        if not is_team_agent:
                            break

                        # 3) Drain inbox — the race-safe entry point. Catches
                        # messages that arrived during agent.send (when sock
                        # was not yet listening), and post-wake messages.
                        unread = bus.read_unread(agent_name) if bus else []
                        if unread:
                            # If we got here via timeout (not signal), the
                            # producer's wake signal was lost — surface so
                            # ops can investigate. Inbox-poll fallback
                            # recovered the message regardless.
                            if last_was_timeout:
                                logger.error(
                                    "wake_signal_missed agent=%s pending=%d — "
                                    "inbox-poll fallback recovered after 30s. "
                                    "Producer likely crashed mid-send, skipped "
                                    "auto_wake_if_idle, or signal delivery failed.",
                                    agent_name, len(unread),
                                )
                            last_was_timeout = False

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
                            pending = "\n".join(inbox_lines)
                            pending_msg_count = len(unread)
                            continue

                        # 4) Empty inbox — sleep waiting for wake signal.
                        # Timeout is a safety net for the rare cases where
                        # the signal is lost (producer SIGKILL between
                        # bus.send and auto_wake_if_idle, send_signal
                        # connect timeout, future sender forgetting to
                        # call auto_wake). Loop returns to step 3 on
                        # either signal or timeout.
                        signal = await channel.listen(timeout=KEEP_ALIVE_TIMEOUT_S)
                        last_was_timeout = (signal is None)
                        if signal:
                            logger.info(
                                "📬 %s woke up on signal '%s'",
                                agent_name, signal,
                            )
                finally:
                    if channel is not None:
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

        # Skill manifests (name + description + body content)
        skills_data = []
        for m in getattr(child_agent, "_skill_manifests", []):
            skills_data.append({
                "name": m.name,
                "description": getattr(m, "description", "") or "",
                "content": getattr(m, "body", "") or "",
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


async def _emit_mcp_status(agent_app: Any, run_id: str, role: str) -> None:
    """Emit MCP health status using real runtime data from MCPAggregator.

    Calls collect_server_status() on the live aggregator to get actual
    connection state, error messages, and implementation info — not
    set subtraction.
    """
    try:
        child_agent = agent_app["child"]
    except (KeyError, TypeError):
        return

    try:
        aggregator = getattr(child_agent, "_aggregator", None)
        if not aggregator:
            return

        # Real runtime status from live server connections
        status_map = await aggregator.collect_server_status()

        mcp_health: dict[str, Any] = {}
        total_connected = 0
        total_failed = 0

        for server_name, server_status in status_map.items():
            is_connected = getattr(server_status, "is_connected", None)
            error_msg = getattr(server_status, "error_message", None)
            impl_name = getattr(server_status, "implementation_name", None)
            transport = getattr(server_status, "transport", None)

            # Tool count from _server_to_tool_map
            server_tool_map = getattr(aggregator, "_server_to_tool_map", {})
            tool_count = len(server_tool_map.get(server_name, []))

            if is_connected:
                total_connected += 1
                status_str = "connected"
            else:
                total_failed += 1
                status_str = "failed"

            mcp_health[server_name] = {
                "status": status_str,
                "is_connected": is_connected,
                "tools": tool_count,
                "error": error_msg,
                "implementation": impl_name,
                "transport": transport,
            }

        total_configured = len(status_map)

        emit_event(
            "mcp_status",
            run_id,
            role,
            total_configured=total_configured,
            total_connected=total_connected,
            total_failed=total_failed,
            servers=mcp_health,
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

    # ── message_turn: stream child_agent.message_history deltas ──
    # Cursor lives in this closure; it survives across turns of the same
    # subprocess. The socket payload is FULL (untruncated). The parent-
    # process bridge applies size capping before SSE broadcast.
    _msg_cursor = {"v": 0}

    def _emit_message_turn_delta() -> None:
        try:
            history = list(getattr(child_agent, "message_history", None) or [])
        except Exception:
            return
        cur = _msg_cursor["v"]
        if cur > len(history):
            cur = 0  # history was cleared
        if cur >= len(history):
            return
        for idx in range(cur, len(history)):
            msg = history[idx]
            try:
                payload = msg.model_dump(mode="json", exclude_none=True)
            except Exception:
                continue
            emit_event(
                "message_turn",
                run_id,
                role,
                turn_idx=idx,
                msg_role=payload.get("role"),
                message=payload,
            )
        _msg_cursor["v"] = len(history)

    async def before_tool_call(runner: Any, request: Any) -> None:
        # File-based debug — does this hook fire?
        try:
            import pathlib as _pl2
            _hp = _pl2.Path(os.environ.get("PROJECT_DIR", ".")) / ".runtime" / "cache" / "logs" / "hooks_debug.log"
            _hp.parent.mkdir(parents=True, exist_ok=True)
            with open(_hp, "a") as _hf:
                _hf.write(f"{time.strftime('%H:%M:%S')} HOOK:before_tool_call FIRED\n")
        except Exception:
            pass
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

        # Stream the new tool_result turn appended to message_history.
        _emit_message_turn_delta()

    # ── Centralized activity hooks: thinking + response ──────────────

    async def spawn_before_llm(runner: Any, messages: Any) -> None:
        """Emit 'thinking' event before each LLM call."""
        # File-based debug — does this hook fire?
        try:
            import pathlib as _pl3
            _hp = _pl3.Path(os.environ.get("PROJECT_DIR", ".")) / ".runtime" / "cache" / "logs" / "hooks_debug.log"
            _hp.parent.mkdir(parents=True, exist_ok=True)
            with open(_hp, "a") as _hf:
                _hf.write(f"{time.strftime('%H:%M:%S')} HOOK:spawn_before_llm FIRED\n")
        except Exception:
            pass
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

        # Stream the new assistant turn appended to message_history.
        _emit_message_turn_delta()

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

    # Pause/Resume: signal-based pause checkpoint hooks.
    #   - ``pause_checkpoint``: used at both before_llm_call AND
    #     before_tool_call so the subprocess pauses at the same
    #     granularity as the in-process Jarvis agent.
    #   - ``pause_after_llm``: clears the in-flight LLM task ref so a
    #     subsequent SIGUSR1 doesn't cancel something past the LLM phase
    #     (tool calls are left running per strategy B).
    #   - ``pause_turn_complete``: after_turn_complete marker that flips
    #     the subprocess ``_active`` flag back to idle, letting the
    #     signal handler emit terminal ``agent_paused`` itself when the
    #     pause request lands on an idle agent.
    #   - ``pause_cancel_filter``: on_pause_cancel hook — when tool_runner
    #     catches CancelledError from a SIGUSR1-cancelled LLM call, this
    #     hook awaits resume and signals retry. Without it the cancel
    #     would propagate up and terminate the subprocess turn entirely.
    pause_before_llm: Any = None
    pause_before_tool: Any = None
    pause_after_llm_hook: Any = None
    pause_turn_done: Any = None
    pause_cancel: Any = None
    try:
        from fast_agent.spawn.pause_signal_handler import (
            pause_checkpoint,
            pause_after_llm,
            pause_turn_complete,
            pause_cancel_filter,
        )
        pause_before_llm = pause_checkpoint
        pause_before_tool = pause_checkpoint
        pause_after_llm_hook = pause_after_llm
        pause_turn_done = pause_turn_complete
        pause_cancel = pause_cancel_filter
    except Exception:
        pass  # Pause is optional — don't break spawn if it fails

    # ── Merge all hooks: spawn events + RTAC + card-defined hooks ──

    existing = child_agent.tool_runner_hooks
    if existing is not None:
        orig_before_tool = existing.before_tool_call
        orig_after_tool = existing.after_tool_call
        orig_before_llm = existing.before_llm_call
        orig_after_llm = existing.after_llm_call

        orig_after_turn = existing.after_turn_complete

        async def merged_before_tool(runner: Any, request: Any) -> None:
            if orig_before_tool:
                await orig_before_tool(runner, request)
            await before_tool_call(runner, request)
            if pause_before_tool:
                await pause_before_tool(runner, request)

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
            if pause_after_llm_hook:
                await pause_after_llm_hook(runner, message)

        async def merged_after_turn(runner: Any, message: Any) -> None:
            if orig_after_turn:
                await orig_after_turn(runner, message)
            if pause_turn_done:
                await pause_turn_done(runner, message)

        child_agent.tool_runner_hooks = ToolRunnerHooks(
            before_llm_call=merged_before_llm,
            after_llm_call=merged_after_llm,
            before_tool_call=merged_before_tool,
            after_tool_call=merged_after_tool,
            after_turn_complete=merged_after_turn,
            on_pause_cancel=pause_cancel,
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

        async def _chained_before_tool(r: Any, req: Any) -> None:
            await before_tool_call(r, req)
            if pause_before_tool:
                await pause_before_tool(r, req)

        async def _chained_after_llm(r: Any, m: Any) -> None:
            await spawn_after_llm(r, m)
            if pause_after_llm_hook:
                await pause_after_llm_hook(r, m)

        child_agent.tool_runner_hooks = ToolRunnerHooks(
            before_llm_call=_chained_before_llm,
            after_llm_call=_chained_after_llm,
            before_tool_call=_chained_before_tool,
            after_tool_call=after_tool_call,
            after_turn_complete=pause_turn_done,
            on_pause_cancel=pause_cancel,
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
