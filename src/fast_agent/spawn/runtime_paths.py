"""Central path constants for all spawn runtime artifacts.

All runtime-generated files live under .runtime/ with semantic subdirectories:
  - state/  — persistent across sessions (signals, messages, runs, registry)
  - cache/  — ephemeral, safe to delete (tmp, logs)
  - data/   — user-facing output (agent_cards, workspaces)

The ``project_dir`` must be provided by the caller (the host application)
since this module lives inside the library, not the project itself.
"""

from __future__ import annotations

from pathlib import Path


def get_runtime_paths(project_dir: str | Path) -> dict[str, Path]:
    """Build all runtime path constants relative to a project directory.

    Args:
        project_dir: Root directory of the host application.

    Returns:
        Dictionary mapping path names to their resolved ``Path`` objects.
    """
    root = Path(project_dir)
    runtime = root / ".runtime"

    # state/ — persistent across sessions
    state = runtime / "state"
    signals = state / "signals"
    messages = state / "messages"
    runs = state / "runs"
    registry_file = state / "spawn_registry.json"

    # cache/ — ephemeral, safe to delete
    cache = runtime / "cache"
    tmp = cache / "tmp"
    logs = cache / "logs"

    # data/ — user-facing output
    data = runtime / "data"
    agent_cards = data / "agent_cards"
    workspaces = data / "workspaces"

    return {
        "runtime": runtime,
        "state": state,
        "signals": signals,
        "messages": messages,
        "runs": runs,
        "registry_file": registry_file,
        "cache": cache,
        "tmp": tmp,
        "logs": logs,
        "data": data,
        "agent_cards": agent_cards,
        "workspaces": workspaces,
        "reload_signal": root / ".reload_needed",
    }


def ensure_runtime_dirs(project_dir: str | Path) -> None:
    """Create all runtime directories if they don't exist."""
    paths = get_runtime_paths(project_dir)
    for key in (
        "signals",
        "messages",
        "runs",
        "tmp",
        "logs",
        "agent_cards",
        "workspaces",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
