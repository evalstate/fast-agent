from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.constants import DEFAULT_ENVIRONMENT_DIR, DEFAULT_SKILLS_PATHS
from fast_agent.home import resolve_fast_agent_home

if TYPE_CHECKING:
    from fast_agent.config import Settings


@dataclass(frozen=True)
class EnvironmentPaths:
    root: Path
    card_packs: Path
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


def _is_ambient_legacy_environment_dir(value: str | Path | None) -> bool:
    if value is None:
        return False
    legacy_value = os.getenv("ENVIRONMENT_DIR")
    if not legacy_value:
        return False
    return Path(value).expanduser() == Path(legacy_value).expanduser()


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
        if settings._fast_agent_noenv:
            raise ValueError("fast-agent home is disabled for these settings")
        configured_environment_dir = settings.environment_dir
        if (
            configured_environment_dir is not None
            and not _is_ambient_legacy_environment_dir(configured_environment_dir)
        ):
            environment_dir = configured_environment_dir
            env_path = Path(environment_dir).expanduser()
            return _resolve_relative_path(env_path, base)
        if settings._fast_agent_home is not None:
            return Path(settings._fast_agent_home).expanduser().resolve()

    if environment_dir is not None:
        env_path = Path(environment_dir).expanduser()
        return _resolve_relative_path(env_path, base)

    home = resolve_fast_agent_home(cwd=base)
    if home is not None:
        return home.path

    return _resolve_relative_path(Path(DEFAULT_ENVIRONMENT_DIR), base)


def resolve_environment_paths(
    settings: "Settings | None" = None,
    *,
    cwd: Path | None = None,
    override: str | Path | None = None,
) -> EnvironmentPaths:
    root = resolve_environment_dir(settings=settings, cwd=cwd, override=override)
    return EnvironmentPaths(
        root=root,
        card_packs=root / "card-packs",
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
    if settings is None:
        from fast_agent.config import Settings

        settings = Settings()
    env_paths = (
        None
        if override is None and settings._fast_agent_noenv
        else resolve_environment_paths(settings=settings, cwd=base, override=override)
    )
    resolved: list[Path] = []
    env_skills_entry = Path(DEFAULT_ENVIRONMENT_DIR) / "skills"
    for entry in DEFAULT_SKILLS_PATHS:
        raw_path = Path(entry).expanduser()
        if raw_path == env_skills_entry:
            if env_paths is None:
                continue
            path = env_paths.skills
        else:
            path = _resolve_relative_path(raw_path, base)
        if path not in resolved:
            resolved.append(path)
    return resolved


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
