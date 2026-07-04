"""
Helpers for applying template variables to system prompts after initial bootstrap.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fast_agent.core.internal_resources import (
    format_internal_resources_for_prompt,
    list_internal_resources,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

    from fast_agent.skills import SkillManifest
    from fast_agent.tools.execution_environment import ShellEnvironment

logger = get_logger(__name__)


@runtime_checkable
class _EnvironmentCwdProvider(Protocol):
    @property
    def cwd(self) -> str: ...


def _display_name_with_version(
    info: Mapping[str, str],
    *,
    title_key: str = "title",
    name_key: str = "name",
    version_key: str = "version",
) -> str | None:
    display_name = strip_to_none(info.get(title_key)) or strip_to_none(info.get(name_key))
    if display_name is None:
        return None

    version = strip_to_none(info.get(version_key))
    if version and version != "unknown":
        return f"{display_name} {version}"
    return display_name


def _format_client_info(client_info: Mapping[str, str]) -> str | None:
    display = _display_name_with_version(client_info)
    if not display:
        return None

    via = _display_name_with_version(
        client_info,
        title_key="viaTitle",
        name_key="viaName",
        version_key="viaVersion",
    )
    if via:
        return f"{display} via {via}"
    return display


def _format_execution_environment(
    *,
    name: str | None,
    kind: str | None,
    provider: str | None,
    shell_name: str | None,
    cwd: str | None,
) -> str:
    parts: list[str] = []
    if name:
        parts.append(name)
    if kind:
        parts.append(kind)
    if provider and provider != kind:
        parts.append(provider)

    text = " ".join(parts) if parts else "unknown"
    details: list[str] = []
    if shell_name:
        details.append(f"shell: {shell_name}")
    if cwd:
        details.append(f"cwd: {cwd}")
    if details:
        text = f"{text} ({', '.join(details)})"
    return text


def refresh_execution_environment_context(
    context: MutableMapping[str, str],
    shell_environment: "ShellEnvironment | None",
) -> None:
    """Populate active execution-environment prompt placeholders."""
    if shell_environment is None:
        return

    from fast_agent.tools.environment_registry import environment_name

    info = shell_environment.runtime_info()
    resolved_name = info.environment_name or environment_name(shell_environment)
    environment_cwd = (
        shell_environment.cwd if isinstance(shell_environment, _EnvironmentCwdProvider) else None
    )
    if info.kind != "local" and environment_cwd:
        host_workspace_root = context.get("workspaceRoot")
        if host_workspace_root:
            context["hostWorkspaceRoot"] = host_workspace_root
        context["workspaceRoot"] = environment_cwd

    if resolved_name:
        context["executionEnvironmentName"] = resolved_name
    context["executionEnvironmentKind"] = info.kind
    if info.provider:
        context["executionEnvironmentProvider"] = info.provider
    if info.name:
        context["executionEnvironmentShell"] = info.name
    if environment_cwd:
        context["executionEnvironmentCwd"] = environment_cwd
    context["executionEnvironment"] = _format_execution_environment(
        name=resolved_name,
        kind=info.kind,
        provider=info.provider,
        shell_name=info.name,
        cwd=environment_cwd,
    )
    _refresh_env_summary(context)


def _refresh_env_summary(context: MutableMapping[str, str]) -> None:
    env_lines: list[str] = []
    workspace_root = context.get("workspaceRoot")
    if workspace_root:
        env_lines.append(f"Workspace root: {workspace_root}")
    execution_environment = context.get("executionEnvironment")
    if execution_environment:
        env_lines.append(f"Execution environment: {execution_environment}")
    client = context.get("clientDisplay")
    if client:
        env_lines.append(f"Client: {client}")
    host_platform = context.get("hostPlatform")
    environment_kind = context.get("executionEnvironmentKind")
    if host_platform and (environment_kind is None or environment_kind == "local"):
        env_lines.append(f"Host platform: {host_platform}")

    if env_lines:
        context["env"] = "Environment:\n- " + "\n- ".join(env_lines)


def load_skills_for_context(
    workspace_root: str | None,
    skills_directory_override: str | Path | Sequence[str | Path] | None = None,
    *,
    no_home: bool = False,
) -> list["SkillManifest"]:
    """
    Load skill manifests from the workspace root or override directory.

    Args:
        workspace_root: The workspace root directory
        skills_directory_override: Optional override for skills directories (relative to workspace_root)

    Returns:
        List of SkillManifest objects
    """
    from fast_agent.skills.registry import SkillRegistry

    if not workspace_root:
        return []

    base_dir = Path(workspace_root)

    # If override is provided, treat it as relative to workspace_root
    override_dirs = None
    if skills_directory_override is not None:
        entries = (
            [skills_directory_override]
            if isinstance(skills_directory_override, (str, Path))
            else list(skills_directory_override)
        )
        override_dirs = []
        for entry in entries:
            override_path = Path(entry)
            if override_path.is_absolute():
                override_dirs.append(override_path)
            else:
                override_dirs.append(base_dir / override_path)
    else:
        from fast_agent.config import get_settings
        from fast_agent.paths import default_skill_paths

        settings = get_settings()
        settings_for_skills = (
            settings
            if no_home
            or settings.home is not None
            or settings._fast_agent_home_source != "default"
            else None
        )
        override_dirs = default_skill_paths(
            settings_for_skills,
            cwd=base_dir,
        )

    registry = SkillRegistry(base_dir=base_dir, directories=override_dirs)
    try:
        return registry.load_manifests()
    except Exception as exc:
        logger.warning("Failed to load skills; continuing without them", data={"error": str(exc)})
        return []


def enrich_with_environment_context(
    context: MutableMapping[str, str],
    cwd: str | None,
    client_info: Mapping[str, str] | None,
    skills_directory_override: str | Path | Sequence[str | Path] | None = None,
    *,
    no_home: bool = False,
    shell_environment: "ShellEnvironment | None" = None,
) -> None:
    """
    Populate the provided context mapping with environment details used for template replacement.

    Args:
        context: The context mapping to populate
        cwd: The current working directory (workspace root)
        client_info: Client information mapping
        skills_directory_override: Optional override for skills directories
    """
    if cwd:
        context["workspaceRoot"] = cwd
        if not no_home:
            from fast_agent.paths import resolve_home_paths

            home_paths = resolve_home_paths(cwd=Path(cwd))
            context["homeDir"] = str(home_paths.root)
            context["homeAgentCardsDir"] = str(home_paths.agent_cards)
            context["homeToolCardsDir"] = str(home_paths.tool_cards)

    server_platform = platform.platform()
    python_version = platform.python_version()

    # Provide individual placeholders for automation
    if server_platform:
        context["hostPlatform"] = server_platform
    context["pythonVer"] = python_version

    # Load and format agent skills
    # In ACP context, use read_text_file as the tool for reading skills
    if cwd:
        from fast_agent.skills.registry import format_skills_for_prompt

        skill_manifests = load_skills_for_context(cwd, skills_directory_override, no_home=no_home)
        skills_text = format_skills_for_prompt(skill_manifests, read_tool_name="read_text_file")
        context["agentSkills"] = skills_text

    internal_resources = list_internal_resources()
    context["agentInternalResources"] = format_internal_resources_for_prompt(internal_resources)

    if client_info:
        formatted_client = _format_client_info(client_info)
        if formatted_client:
            context["clientDisplay"] = formatted_client
    refresh_execution_environment_context(context, shell_environment)
    _refresh_env_summary(context)
