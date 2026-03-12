"""Workspace Manager — shared project workspace for team agents.

Creates a structured directory for team agents to read/write artifacts.
Each workspace has: specs/, src/, tests/, docs/, reviews/, changelog.md
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

WORKSPACE_DIRS = ["specs", "src", "tests", "docs", "reviews"]


def create_workspace(
    project_name: str,
    workspaces_dir: str | Path,
    root: Path | None = None,
) -> Path:
    """Create a structured workspace directory for a team project.

    Args:
        project_name: Human-readable project name (sanitized for filesystem).
        workspaces_dir: Default directory for workspaces.
        root: Optional override root directory.

    Returns:
        Path to the workspace directory.
    """
    base = root or Path(workspaces_dir)
    safe_name = project_name.lower().replace(" ", "_").replace("/", "_")[:50]
    workspace = base / safe_name
    workspace.mkdir(parents=True, exist_ok=True)

    for d in WORKSPACE_DIRS:
        (workspace / d).mkdir(exist_ok=True)

    # Create changelog
    changelog = workspace / "changelog.md"
    if not changelog.exists():
        changelog.write_text(
            f"# Changelog — {project_name}\n\n"
            f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "---\n\n",
            encoding="utf-8",
        )

    logger.info("Workspace created: %s", workspace)
    return workspace


def append_changelog(workspace: Path, agent_role: str, step: str, message: str) -> None:
    """Append an entry to the workspace changelog."""
    changelog = workspace / "changelog.md"
    entry = f"### [{time.strftime('%H:%M:%S')}] {agent_role} — {step}\n{message}\n\n"
    with open(changelog, "a", encoding="utf-8") as f:
        f.write(entry)


def get_workspace_summary(workspace: Path) -> dict[str, object]:
    """Get a summary of workspace contents."""
    summary: dict[str, object] = {
        "path": str(workspace),
        "directories": {},
    }
    dirs: dict[str, list[str]] = {}
    for d in WORKSPACE_DIRS:
        dir_path = workspace / d
        if dir_path.exists():
            dirs[d] = [f.name for f in dir_path.iterdir() if f.is_file()]
    summary["directories"] = dirs
    return summary
