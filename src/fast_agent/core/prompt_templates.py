"""
Helpers for applying template variables to system prompts after initial bootstrap.
"""

from __future__ import annotations

import platform
import re
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, MutableMapping, Sequence

if TYPE_CHECKING:
    from fast_agent.skills import SkillManifest


def apply_template_variables(
    template: str | None, variables: Mapping[str, str | None] | None
) -> str | None:
    """
    Apply a mapping of template variables to the provided template string.

    This helper intentionally performs no work when either the template or variables
    are empty so callers can safely execute it during both the initial and late
    initialization passes without accidentally stripping placeholders too early.

    Supports both simple variable substitution and file template patterns:
    - {{variable}} - Simple variable replacement
    - {{file:relative/path}} - Reads file contents (relative to workspaceRoot, errors if missing)
    - {{file_silent:relative/path}} - Reads file contents (relative to workspaceRoot, empty if missing)
    """
    if not template or not variables:
        return template

    resolved = template

    # Get workspaceRoot for file resolution
    workspace_root = variables.get("workspaceRoot")

    # Apply {{file:...}} templates (relative paths required, resolved from workspaceRoot)
    file_pattern = re.compile(r"\{\{file:([^}]+)\}\}")

    def replace_file(match):
        file_path_str = match.group(1).strip()
        file_path = Path(file_path_str).expanduser()

        # Enforce relative paths
        if file_path.is_absolute():
            raise ValueError(
                f"File template paths must be relative, got absolute path: {file_path_str}"
            )

        # Resolve against workspaceRoot if available
        if workspace_root:
            resolved_path = (Path(workspace_root) / file_path).resolve()
        else:
            resolved_path = file_path.resolve()

        return resolved_path.read_text(encoding="utf-8")

    resolved = file_pattern.sub(replace_file, resolved)

    # Apply {{file_silent:...}} templates (missing files become empty strings)
    file_silent_pattern = re.compile(r"\{\{file_silent:([^}]+)\}\}")

    def replace_file_silent(match):
        file_path_str = match.group(1).strip()
        file_path = Path(file_path_str).expanduser()

        # Enforce relative paths
        if file_path.is_absolute():
            raise ValueError(
                f"File template paths must be relative, got absolute path: {file_path_str}"
            )

        # Resolve against workspaceRoot if available
        if workspace_root:
            resolved_path = (Path(workspace_root) / file_path).resolve()
        else:
            resolved_path = file_path.resolve()

        try:
            return resolved_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    resolved = file_silent_pattern.sub(replace_file_silent, resolved)

    # Apply simple variable substitutions
    for key, value in variables.items():
        if value is None:
            continue
        placeholder = f"{{{{{key}}}}}"
        if placeholder in resolved:
            resolved = resolved.replace(placeholder, value)

    return resolved


def load_skills_for_context(
    workspace_root: str | None,
    skills_directory_override: str | Path | Sequence[str | Path] | None = None,
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

    registry = SkillRegistry(base_dir=base_dir, directories=override_dirs)
    return registry.load_manifests()


def enrich_with_environment_context(
    context: MutableMapping[str, str],
    cwd: str | None,
    client_info: Mapping[str, str] | None,
    skills_directory_override: str | Path | Sequence[str | Path] | None = None,
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

        skill_manifests = load_skills_for_context(cwd, skills_directory_override)
        skills_text = format_skills_for_prompt(skill_manifests, read_tool_name="read_text_file")
        context["agentSkills"] = skills_text

    env_lines: list[str] = []
    if cwd:
        env_lines.append(f"Workspace root: {cwd}")
    if client_info:
        display_name = client_info.get("title") or client_info.get("name")
        version = client_info.get("version")
        if display_name:
            if version and version != "unknown":
                env_lines.append(f"Client: {display_name} {version}")
            else:
                env_lines.append(f"Client: {display_name}")
    if server_platform:
        env_lines.append(f"Host platform: {server_platform}")

    if env_lines:
        formatted = "Environment:\n- " + "\n- ".join(env_lines)
        context["env"] = formatted
