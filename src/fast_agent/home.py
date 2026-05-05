"""fast-agent home and configuration discovery helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from fast_agent.constants import DEFAULT_ENVIRONMENT_DIR
from fast_agent.core.exceptions import ConfigFileError

HomeSource = Literal["cli", "FAST_AGENT_HOME", "ENVIRONMENT_DIR", "default"]
ConfigSource = Literal["explicit", "home", "cwd", "none"]
SecretsSource = Literal["same_dir", "home", "cwd", "none"]

PREFERRED_CONFIG_FILENAME = "fast-agent.yaml"
TRANSITIONAL_CONFIG_FILENAMES = ("fast-agent.config.yaml",)
LEGACY_CONFIG_FILENAMES = ("fastagent.config.yaml",)
CONFIG_FILENAMES = (
    PREFERRED_CONFIG_FILENAME,
    *TRANSITIONAL_CONFIG_FILENAMES,
    *LEGACY_CONFIG_FILENAMES,
)

PREFERRED_SECRETS_FILENAME = "fast-agent.secrets.yaml"
LEGACY_SECRETS_FILENAMES = ("fastagent.secrets.yaml",)
SECRETS_FILENAMES = (PREFERRED_SECRETS_FILENAME, *LEGACY_SECRETS_FILENAMES)


@dataclass(frozen=True, slots=True)
class FastAgentHome:
    path: Path
    source: HomeSource


@dataclass(frozen=True, slots=True)
class ConfigDiscoveryResult:
    home: FastAgentHome | None
    config_path: Path | None
    secrets_path: Path | None
    config_source: ConfigSource
    secrets_source: SecretsSource


class ConfigDiscoveryError(ConfigFileError):
    """Base class for fast-agent config discovery failures."""


class AmbiguousConfigFilesError(ConfigDiscoveryError):
    """Raised when multiple supported config aliases exist in one directory."""

    def __init__(self, directory: Path, candidates: tuple[Path, ...]) -> None:
        self.directory = directory
        self.candidates = candidates
        super().__init__(_format_ambiguity("config", directory, candidates))


class AmbiguousSecretsFilesError(ConfigDiscoveryError):
    """Raised when multiple supported secrets aliases exist in one directory."""

    def __init__(self, directory: Path, candidates: tuple[Path, ...]) -> None:
        self.directory = directory
        self.candidates = candidates
        super().__init__(_format_ambiguity("secrets", directory, candidates))


def resolve_fast_agent_home(
    *,
    cwd: Path | None = None,
    cli_override: str | Path | None = None,
    noenv: bool = False,
) -> FastAgentHome | None:
    """Resolve the active fast-agent home.

    Precedence: ``--env``/``cli_override`` > ``FAST_AGENT_HOME`` >
    ``ENVIRONMENT_DIR`` > ``./.fast-agent``. ``noenv`` disables home selection.
    """
    if noenv:
        return None

    base = _resolve_cwd(cwd)
    if cli_override is not None:
        return FastAgentHome(_resolve_path(cli_override, base), "cli")

    fast_agent_home = os.getenv("FAST_AGENT_HOME")
    if fast_agent_home:
        return FastAgentHome(_resolve_path(fast_agent_home, base), "FAST_AGENT_HOME")

    legacy_environment_dir = os.getenv("ENVIRONMENT_DIR")
    if legacy_environment_dir:
        return FastAgentHome(_resolve_path(legacy_environment_dir, base), "ENVIRONMENT_DIR")

    return FastAgentHome((base / DEFAULT_ENVIRONMENT_DIR).resolve(), "default")


def discover_config_files(
    *,
    cwd: Path | None = None,
    home: FastAgentHome | None = None,
    explicit_config_path: str | Path | None = None,
) -> ConfigDiscoveryResult:
    """Discover config and secrets files without parent-directory walking."""
    base = _resolve_cwd(cwd)

    if explicit_config_path is not None:
        config_path = _resolve_path(explicit_config_path, base)
        secrets_path = find_secrets_in_directory(config_path.parent)
        return ConfigDiscoveryResult(
            home=home,
            config_path=config_path,
            secrets_path=secrets_path,
            config_source="explicit",
            secrets_source="same_dir" if secrets_path else "none",
        )

    searched: set[Path] = set()
    if home is not None:
        home_dir = home.path.resolve()
        searched.add(home_dir)
        config_path = find_config_in_directory(home_dir)
        if config_path is not None:
            secrets_path = find_secrets_in_directory(config_path.parent)
            return ConfigDiscoveryResult(
                home=home,
                config_path=config_path,
                secrets_path=secrets_path,
                config_source="home",
                secrets_source="same_dir" if secrets_path else "none",
            )

    cwd_dir = base.resolve()
    if cwd_dir not in searched:
        config_path = find_config_in_directory(cwd_dir)
        if config_path is not None:
            secrets_path = find_secrets_in_directory(config_path.parent)
            return ConfigDiscoveryResult(
                home=home,
                config_path=config_path,
                secrets_path=secrets_path,
                config_source="cwd",
                secrets_source="same_dir" if secrets_path else "none",
            )

    if home is not None:
        secrets_path = find_secrets_in_directory(home.path)
        if secrets_path is not None:
            return ConfigDiscoveryResult(
                home=home,
                config_path=None,
                secrets_path=secrets_path,
                config_source="none",
                secrets_source="home",
            )

    if cwd_dir not in searched:
        secrets_path = find_secrets_in_directory(cwd_dir)
        if secrets_path is not None:
            return ConfigDiscoveryResult(
                home=home,
                config_path=None,
                secrets_path=secrets_path,
                config_source="none",
                secrets_source="cwd",
            )

    return ConfigDiscoveryResult(
        home=home,
        config_path=None,
        secrets_path=None,
        config_source="none",
        secrets_source="none",
    )


def find_config_in_directory(directory: Path) -> Path | None:
    """Return the single supported config file in ``directory``, or raise on ambiguity."""
    candidates = _existing_files(directory, CONFIG_FILENAMES)
    if len(candidates) > 1:
        raise AmbiguousConfigFilesError(directory.resolve(), candidates)
    return candidates[0] if candidates else None


def find_secrets_in_directory(directory: Path) -> Path | None:
    """Return the single supported secrets file in ``directory``, or raise on ambiguity."""
    candidates = _existing_files(directory, SECRETS_FILENAMES)
    if len(candidates) > 1:
        raise AmbiguousSecretsFilesError(directory.resolve(), candidates)
    return candidates[0] if candidates else None


def build_child_environment(
    *,
    active_home: str | Path | None,
    noenv: bool = False,
    base: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build an environment for shell/MCP child processes.

    ``FAST_AGENT_RUNTIME_ENVIRONMENT`` is the documented runtime export.
    ``ENVIRONMENT_DIR`` is exported alongside it as a legacy compatibility alias.
    In ``--noenv`` mode both are removed, including from explicit overrides.
    """
    from fast_agent.constants import FAST_AGENT_RUNTIME_ENVIRONMENT

    env = dict(os.environ if base is None else base)
    if not noenv and active_home is not None:
        home = str(Path(active_home).expanduser().resolve())
        env[FAST_AGENT_RUNTIME_ENVIRONMENT] = home
        env["ENVIRONMENT_DIR"] = home

    if overrides:
        env.update(overrides)

    if noenv:
        env.pop(FAST_AGENT_RUNTIME_ENVIRONMENT, None)
        env.pop("ENVIRONMENT_DIR", None)

    return env


def _existing_files(directory: Path, filenames: tuple[str, ...]) -> tuple[Path, ...]:
    resolved_dir = directory.expanduser().resolve()
    return tuple(path for filename in filenames if (path := resolved_dir / filename).is_file())


def _resolve_cwd(cwd: Path | None) -> Path:
    return (cwd or Path.cwd()).expanduser().resolve()


def _resolve_path(path: str | Path, cwd: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = cwd / resolved
    return resolved.resolve()


def _format_ambiguity(kind: str, directory: Path, candidates: tuple[Path, ...]) -> str:
    names = "\n".join(f"- {candidate.name}" for candidate in candidates)
    return (
        f"Multiple fast-agent {kind} files found in {directory}:\n"
        f"{names}\n\n"
        f"Please keep only one {kind} file in this directory."
    )
