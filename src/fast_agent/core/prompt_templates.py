"""Prompt template utilities for fast-agent.

This module provides functions for building and rendering prompt templates,
managing instruction context, and enriching prompts with environment information.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fast_agent.core.logging.logger import get_logger
from fast_agent.resources import get_resource_path

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


_SMART_PROMPT_KEY = "internal:smart_prompt"


def smart_prompt_template() -> str:
    """Return the built-in smart prompt template."""
    path = get_resource_path("smart_prompt.md")
    return path.read_text()


# ---------------------------------------------------------------------------
# Instruction building helpers
# ---------------------------------------------------------------------------


class InstructionBuilder:
    """Utility for resolving ``{{placeholder}}`` values in instruction templates.

    Supports both static string values and async *resolvers* – callables
    that are invoked lazily at build time.  Static values take precedence
    when the same key has both a static value and a resolver.

    Usage::

        builder = InstructionBuilder(template)
        builder.set("key", "value")
        builder.set_resolver("other", some_async_fn)
        result = await builder.build()
    """

    def __init__(self, template: str, *, source: str | None = None) -> None:
        self._template = template
        self._static: dict[str, str] = {}
        self._resolvers: dict[str, Any] = {}
        self._source = source  # diagnostic label (agent name, etc.)

    # -- public setters -----------------------------------------------------

    def set(self, key: str, value: str) -> None:
        """Register a static replacement value for ``{{key}}``."""
        self._static[key] = value

    def set_many(self, values: dict[str, str]) -> None:
        """Register multiple static replacement values."""
        self._static.update(values)

    def set_resolver(self, key: str, resolver: Any) -> None:
        """Register an async callable that produces the replacement for ``{{key}}``."""
        self._resolvers[key] = resolver

    # -- build --------------------------------------------------------------

    async def build(self) -> str:
        """Resolve all placeholders and return the final instruction."""
        result = self._template

        # 1. Static values first (fast path)
        for key, value in self._static.items():
            placeholder = "{{" + key + "}}"
            if placeholder in result:
                result = result.replace(placeholder, value)

        # 2. Resolvers for anything still unresolved
        for key, resolver in self._resolvers.items():
            placeholder = "{{" + key + "}}"
            if placeholder in result:
                try:
                    value = await resolver()
                    result = result.replace(placeholder, value)
                except Exception as exc:
                    logger.warning(
                        "Resolver for '%s' failed: %s",
                        key,
                        exc,
                        extra={"source": self._source},
                    )

        return result


# ---------------------------------------------------------------------------
# Internal-resource prompt formatting
# ---------------------------------------------------------------------------


def list_internal_resources() -> list[dict[str, str]]:
    """Return manifests for built-in ``internal:`` resources.

    Currently only the smart-prompt resource is advertised.
    """
    return [
        {
            "key": _SMART_PROMPT_KEY,
            "name": "Smart Prompt",
            "description": (
                "A comprehensive, best-practice system prompt that combines "
                "well-known prompting guidelines."
            ),
        }
    ]


def format_internal_resources_for_prompt(resources: list[dict[str, str]]) -> str:
    """Render the internal-resources list into prompt-ready text."""
    if not resources:
        return ""

    lines = [
        "The following internal resources are available as instruction templates:",
        "",
    ]
    for resource in resources:
        lines.append(f"  - `{resource['key']}`: {resource['description']}")
    lines.append("")
    lines.append(
        "Reference them in agent card instructions with "
        "{{internal:resource_name}} to include their content."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Skills loading helper
# ---------------------------------------------------------------------------


def load_skills_for_context(
    cwd: str | Path,
    skills_directory_override: str | None = None,
) -> list[Any]:
    """Discover skill manifests under *cwd* (or the override directory).

    Returns a list of ``SkillManifest`` objects found on disk.
    """
    from fast_agent.skills.registry import SkillRegistry

    registry = SkillRegistry(
        base_dir=Path(cwd),
        skills_directory_override=skills_directory_override,
    )
    return list(registry.discover_all().values())


# ---------------------------------------------------------------------------
# Environment context enrichment
# ---------------------------------------------------------------------------


def enrich_with_environment_context(
    context: dict[str, str],
    *,
    cwd: str | None = None,
    skills_directory_override: str | None = None,
) -> dict[str, str]:
    """Populate *context* with environment-derived values.

    This is used when building instructions for agent cards so that
    placeholders like ``{{hostPlatform}}``, ``{{pythonVer}}``, and
    ``{{agentSkills}}`` have concrete values.
    """
    server_platform = platform.system()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

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
        # NOTE: Do NOT set context["agentSkills"] here.
        # The per-agent dynamic resolver in instruction_refresh.py handles
        # agentSkills correctly — it filters to only the skills configured
        # for each individual agent. Setting it here as a static value would
        # override the dynamic resolver and cause ALL skills to appear in
        # every agent's prompt, regardless of their skill configuration.

    internal_resources = list_internal_resources()
    context["agentInternalResources"] = format_internal_resources_for_prompt(internal_resources)

    env_lines: list[str] = []
    if cwd:
        env_lines.append(f"Workspace root: {cwd}")

    if server_platform:
        env_lines.append(f"Platform: {server_platform}")

    if python_version:
        env_lines.append(f"Python: {python_version}")

    if env_lines:
        context["environment"] = "\n".join(env_lines)

    return context
