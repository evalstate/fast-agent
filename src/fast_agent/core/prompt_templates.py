"""
Helpers for applying template variables to system prompts after initial bootstrap.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, MutableMapping, Sequence

from fast_agent.core.internal_resources import (
    format_internal_resources_for_prompt,
    list_internal_resources,
)
from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from fast_agent.skills import SkillManifest

logger = get_logger(__name__)


def _display_name_with_version(
    info: Mapping[str, str],
    *,
    title_key: str = "title",
    name_key: str = "name",
    version_key: str = "version",
) -> str | None:
    display_name = info.get(title_key) or info.get(name_key)
    if not display_name:
        return None

    version = info.get(version_key)
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


def load_skills_for_context(
    workspace_root: str | None,
    skills_directory_override: str | Path | Sequence[str | Path] | None = None,
    *,
    noenv: bool = False,
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
            if noenv
            or settings.environment_dir is not None
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load skills; continuing without them", data={"error": str(exc)})
        return []


def enrich_with_environment_context(
    context: MutableMapping[str, str],
    cwd: str | None,
    client_info: Mapping[str, str] | None,
    skills_directory_override: str | Path | Sequence[str | Path] | None = None,
    *,
    noenv: bool = False,
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
        if not noenv:
            from fast_agent.paths import resolve_environment_paths

            env_paths = resolve_environment_paths(cwd=Path(cwd))
            context["environmentDir"] = str(env_paths.root)
            context["environmentAgentCardsDir"] = str(env_paths.agent_cards)
            context["environmentToolCardsDir"] = str(env_paths.tool_cards)

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

        skill_manifests = load_skills_for_context(cwd, skills_directory_override, noenv=noenv)
        skills_text = format_skills_for_prompt(skill_manifests, read_tool_name="read_text_file")
        context["agentSkills"] = skills_text

    internal_resources = list_internal_resources()
    context["agentInternalResources"] = format_internal_resources_for_prompt(internal_resources)

    env_lines: list[str] = []
    if cwd:
        env_lines.append(f"Workspace root: {cwd}")
    if client_info:
        formatted_client = _format_client_info(client_info)
        if formatted_client:
            env_lines.append(f"Client: {formatted_client}")
    if server_platform:
        env_lines.append(f"Host platform: {server_platform}")

    if env_lines:
        formatted = "Environment:\n- " + "\n- ".join(env_lines)
        context["env"] = formatted
