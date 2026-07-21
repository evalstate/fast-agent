from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.constants import DEFAULT_HOME_DIR, DEFAULT_SKILLS_PATHS, FAST_AGENT_RUNTIME_HOME
from fast_agent.home import resolve_fast_agent_home

if TYPE_CHECKING:
    from fast_agent.config import Settings


@dataclass(frozen=True)
class HomePaths:
    root: Path
    card_packs: Path
    plugins: Path
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


def resolve_settings_start_path(
    settings: "Settings | None" = None,
    *,
    fallback_path: Path | None = None,
) -> Path:
    """Resolve the project base path implied by settings config/env metadata."""
    if settings is not None:
        config_file = settings._config_file
        if config_file:
            config_parent = Path(config_file).expanduser().resolve().parent
            if config_parent.name == DEFAULT_HOME_DIR:
                return config_parent.parent
            return config_parent

        home = settings.home
        if home:
            home_root = Path(home).expanduser()
            if home_root.is_absolute():
                return home_root.resolve().parent

    if fallback_path is not None:
        return fallback_path.resolve()

    return Path.cwd().resolve()


def resolve_home_dir(
    settings: "Settings | None" = None,
    *,
    cwd: Path | None = None,
    override: str | Path | None = None,
) -> Path:
    base = cwd or Path.cwd()
    home = override
    if home is None:
        if settings is None:
            from fast_agent.config import get_settings

            settings = get_settings()
        if settings._fast_agent_no_home:
            raise ValueError("fast-agent home is disabled for these settings")
        configured_home = settings.home
        if configured_home is not None:
            home = configured_home
            home_path = Path(home).expanduser()
            return _resolve_relative_path(home_path, base)
        if settings._fast_agent_home is not None:
            return Path(settings._fast_agent_home).expanduser().resolve()

    if home is not None:
        home_path = Path(home).expanduser()
        return _resolve_relative_path(home_path, base)

    runtime_home = os.getenv(FAST_AGENT_RUNTIME_HOME)
    if runtime_home:
        return _resolve_relative_path(Path(runtime_home).expanduser(), base)

    home = resolve_fast_agent_home(cwd=base)
    if home is not None:
        return home.path

    return _resolve_relative_path(Path(DEFAULT_HOME_DIR), base)


def resolve_home_paths(
    settings: "Settings | None" = None,
    *,
    cwd: Path | None = None,
    override: str | Path | None = None,
) -> HomePaths:
    root = resolve_home_dir(settings=settings, cwd=cwd, override=override)
    return HomePaths(
        root=root,
        card_packs=root / "card-packs",
        plugins=root / "plugins",
        agent_cards=root / "agent-cards",
        tool_cards=root / "tool-cards",
        skills=root / "skills",
        sessions=root / "sessions",
        ui=root / "ui",
        permissions_file=root / "auths.md",
    )


def resolve_log_file_path(settings: "Settings") -> Path:
    """Resolve the configured log path, defaulting to the active fast-agent home."""
    path = Path(settings.logger.path)
    if "path" in settings.logger.model_fields_set or settings._fast_agent_no_home:
        return path
    return resolve_home_dir(settings) / path


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
    home_paths = (
        None
        if override is None and settings._fast_agent_no_home
        else resolve_home_paths(settings=settings, cwd=base, override=override)
    )
    resolved: list[Path] = []
    home_skills_entry = Path(DEFAULT_HOME_DIR) / "skills"
    for entry in DEFAULT_SKILLS_PATHS:
        raw_path = Path(entry).expanduser()
        if raw_path == home_skills_entry:
            if home_paths is None:
                continue
            path = home_paths.skills
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

    dir_setting = settings.mcp_ui_output_dir
    home_paths = resolve_home_paths(settings=settings, cwd=base, override=override)
    if dir_setting in (None, str(Path(DEFAULT_HOME_DIR) / "ui")):
        return home_paths.ui

    return _resolve_relative_path(Path(dir_setting).expanduser(), base)
