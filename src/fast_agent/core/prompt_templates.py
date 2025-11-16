"""
Helpers for applying template variables to system prompts after initial bootstrap.
"""

from __future__ import annotations

import platform
from typing import Mapping, MutableMapping


def apply_template_variables(
    template: str | None, variables: Mapping[str, str | None] | None
) -> str | None:
    """
    Apply a mapping of template variables to the provided template string.

    This helper intentionally performs no work when either the template or variables
    are empty so callers can safely execute it during both the initial and late
    initialization passes without accidentally stripping placeholders too early.
    """
    if not template or not variables:
        return template

    resolved = template
    for key, value in variables.items():
        if value is None:
            continue
        placeholder = f"{{{{{key}}}}}"
        if placeholder in resolved:
            resolved = resolved.replace(placeholder, value)
    return resolved


def enrich_with_environment_context(
    context: MutableMapping[str, str], cwd: str | None, client_info: Mapping[str, str] | None
) -> None:
    """
    Populate the provided context mapping with environment details used for template replacement.
    """
    if cwd:
        context.setdefault("sessionCwd", cwd)

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

    server_platform = platform.platform()
    if server_platform:
        env_lines.append(f"Server platform: {server_platform}")

    python_version = platform.python_version()
    env_lines.append(f"Python: {python_version}")

    if env_lines:
        formatted = "Environment:\n- " + "\n- ".join(env_lines)
        context["env"] = formatted
