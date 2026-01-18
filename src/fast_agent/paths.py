from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.constants import DEFAULT_ENVIRONMENT_DIR

DEFAULT_CLAUDE_SKILLS_DIR = Path(".claude/skills")

if TYPE_CHECKING:
    from fast_agent.config import Settings


@dataclass(frozen=True)
class EnvironmentPaths:
    root: Path
    agent_cards: Path
    tool_cards: Path
    skills: Path
    sessions: Path
    ui: Path
    permissions_file: Path


def _resolve_relative_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


def resolve_environment_dir(
    settings: "Settings | None" = None,
    *,
    cwd: Path | None = None,
    override: str | Path | None = None,
) -> Path:
    base = cwd or Path.cwd()
    environment_dir = override
    if environment_dir is None:
        if settings is None:
            from fast_agent.config import get_settings

            settings = get_settings()
        environment_dir = getattr(settings, "environment_dir", None)

    if environment_dir is None:
        env_path = Path(DEFAULT_ENVIRONMENT_DIR)
    else:
        env_path = Path(environment_dir).expanduser()
    return _resolve_relative_path(env_path, base)


def resolve_environment_paths(
    settings: "Settings | None" = None,
    *,
    cwd: Path | None = None,
    override: str | Path | None = None,
) -> EnvironmentPaths:
    root = resolve_environment_dir(settings=settings, cwd=cwd, override=override)
    return EnvironmentPaths(
        root=root,
        agent_cards=root / "agent-cards",
        tool_cards=root / "tool-cards",
        skills=root / "skills",
        sessions=root / "sessions",
        ui=root / "ui",
        permissions_file=root / "auths.md",
    )


def default_skill_paths(
    settings: "Settings | None" = None,
    *,
    cwd: Path | None = None,
    override: str | Path | None = None,
) -> list[Path]:
    base = cwd or Path.cwd()
    env_paths = resolve_environment_paths(settings=settings, cwd=base, override=override)
    return [env_paths.skills, _resolve_relative_path(DEFAULT_CLAUDE_SKILLS_DIR, base)]


def resolve_mcp_ui_output_dir(
    settings: "Settings | None" = None,
    *,
    cwd: Path | None = None,
    override: str | Path | None = None,
) -> Path:
    base = cwd or Path.cwd()
    if settings is None:
        from fast_agent.config import get_settings

        settings = get_settings()

    dir_setting = getattr(settings, "mcp_ui_output_dir", None)
    env_paths = resolve_environment_paths(settings=settings, cwd=base, override=override)
    if dir_setting in (None, str(Path(DEFAULT_ENVIRONMENT_DIR) / "ui")):
        return env_paths.ui

    return _resolve_relative_path(Path(dir_setting).expanduser(), base)
