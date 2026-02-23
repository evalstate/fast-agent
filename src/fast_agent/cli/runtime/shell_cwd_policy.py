"""Utilities for enforcing shell cwd policy before agent runtime starts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping, Sequence, cast

if TYPE_CHECKING:
    from fast_agent.core.agent_card_types import AgentCardData


MissingShellCwdPolicy = Literal["ask", "create", "warn", "error"]
ShellCwdIssueKind = Literal["missing", "not_directory"]

_VALID_POLICIES: set[str] = {"ask", "create", "warn", "error"}


@dataclass(frozen=True, slots=True)
class ShellCwdIssue:
    """Represents an invalid shell working directory for an agent."""

    agent_name: str
    configured_path: Path
    resolved_path: Path
    kind: ShellCwdIssueKind


@dataclass(frozen=True, slots=True)
class ShellCwdCreationError:
    """Represents a failed auto-create attempt for a missing shell cwd."""

    path: Path
    message: str


def resolve_missing_shell_cwd_policy(
    *,
    cli_override: MissingShellCwdPolicy | None,
    configured_policy: str | None,
) -> MissingShellCwdPolicy:
    """Resolve shell cwd policy with precedence: CLI override > config > default."""
    if cli_override is not None:
        return cli_override

    if configured_policy in _VALID_POLICIES:
        return cast("MissingShellCwdPolicy", configured_policy)

    return "warn"


def can_prompt_for_missing_cwd(
    *,
    mode: Literal["interactive", "serve"],
    message: str | None,
    prompt_file: str | None,
    stdin_is_tty: bool,
    tty_device_available: bool,
) -> bool:
    """Return True when we can ask interactively about creating missing directories."""
    return (
        mode == "interactive"
        and message is None
        and prompt_file is None
        and (stdin_is_tty or tty_device_available)
    )


def effective_missing_shell_cwd_policy(
    policy: MissingShellCwdPolicy,
    *,
    can_prompt: bool,
) -> MissingShellCwdPolicy:
    """Apply runtime constraints to policy (`ask` degrades to `warn` when needed)."""
    if policy == "ask" and not can_prompt:
        return "warn"
    return policy


def collect_shell_cwd_issues(
    agents: Mapping[str, AgentCardData],
    *,
    shell_runtime_requested: bool,
    cwd: Path | None = None,
) -> list[ShellCwdIssue]:
    """Collect invalid shell cwd entries from currently loaded agent configs."""
    base_dir = cwd or Path.cwd()
    issues: list[ShellCwdIssue] = []

    for agent_name in sorted(agents):
        agent_data = agents[agent_name]
        config = agent_data.get("config")
        if config is None:
            continue

        configured_cwd = getattr(config, "cwd", None)
        if not isinstance(configured_cwd, Path):
            continue

        if not _shell_runtime_active_for_agent(
            shell_runtime_requested=shell_runtime_requested,
            shell_enabled=bool(getattr(config, "shell", False)),
            skills_configured=bool(getattr(config, "skill_manifests", []) or []),
        ):
            continue

        resolved_cwd = _resolve_configured_cwd(configured_cwd, base_dir)
        if not resolved_cwd.exists():
            issues.append(
                ShellCwdIssue(
                    agent_name=agent_name,
                    configured_path=configured_cwd,
                    resolved_path=resolved_cwd,
                    kind="missing",
                )
            )
            continue

        if not resolved_cwd.is_dir():
            issues.append(
                ShellCwdIssue(
                    agent_name=agent_name,
                    configured_path=configured_cwd,
                    resolved_path=resolved_cwd,
                    kind="not_directory",
                )
            )

    return issues


def collect_shell_cwd_issues_from_runtime_agents(
    agents: Mapping[str, object],
    *,
    cwd: Path | None = None,
) -> list[ShellCwdIssue]:
    """Collect invalid shell cwd entries from instantiated runtime agents."""
    base_dir = cwd or Path.cwd()
    issues: list[ShellCwdIssue] = []

    for agent_name in sorted(agents):
        agent = agents[agent_name]
        if not bool(getattr(agent, "shell_runtime_enabled", False)):
            continue

        config = getattr(agent, "config", None)
        configured_cwd = getattr(config, "cwd", None)
        if not isinstance(configured_cwd, Path):
            continue

        resolved_cwd = _resolve_configured_cwd(configured_cwd, base_dir)
        if not resolved_cwd.exists():
            issues.append(
                ShellCwdIssue(
                    agent_name=agent_name,
                    configured_path=configured_cwd,
                    resolved_path=resolved_cwd,
                    kind="missing",
                )
            )
            continue

        if not resolved_cwd.is_dir():
            issues.append(
                ShellCwdIssue(
                    agent_name=agent_name,
                    configured_path=configured_cwd,
                    resolved_path=resolved_cwd,
                    kind="not_directory",
                )
            )

    return issues


def create_missing_shell_cwd_directories(
    issues: Sequence[ShellCwdIssue],
) -> tuple[list[Path], list[ShellCwdCreationError]]:
    """Create missing cwd directories and return created paths + creation errors."""
    created_paths: list[Path] = []
    errors: list[ShellCwdCreationError] = []
    seen_paths: set[Path] = set()

    for issue in issues:
        if issue.kind != "missing":
            continue
        if issue.resolved_path in seen_paths:
            continue

        seen_paths.add(issue.resolved_path)

        try:
            issue.resolved_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(ShellCwdCreationError(path=issue.resolved_path, message=str(exc)))
            continue

        if issue.resolved_path.is_dir():
            created_paths.append(issue.resolved_path)
            continue

        errors.append(
            ShellCwdCreationError(
                path=issue.resolved_path,
                message="Path exists but is not a directory.",
            )
        )

    return created_paths, errors


def format_shell_cwd_issues(issues: Sequence[ShellCwdIssue]) -> list[str]:
    """Format invalid cwd entries for grouped diagnostics."""
    if not issues:
        return []

    lines = ["Invalid shell working directories detected:"]
    for issue in issues:
        reason = (
            "does not exist"
            if issue.kind == "missing"
            else "is not a directory"
        )
        lines.append(
            " - "
            f"{issue.agent_name}: cwd={issue.configured_path} "
            f"(resolved: {issue.resolved_path}) [{reason}]"
        )

    return lines


def _resolve_configured_cwd(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _shell_runtime_active_for_agent(
    *,
    shell_runtime_requested: bool,
    shell_enabled: bool,
    skills_configured: bool,
) -> bool:
    return shell_runtime_requested or shell_enabled or skills_configured
