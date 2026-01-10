"""
InstructionBuilder - Builds instruction strings from templates and sources.

This module provides a clean abstraction for constructing agent instructions
from templates with placeholder substitution. Sources can be static values
or dynamic resolvers that are called at build time.

Built-in placeholders (automatically resolved):
    {{currentDate}} - Current date in "17 December 2025" format
    {{hostPlatform}} - Platform info (e.g., "Linux-6.6.0-x86_64")
    {{pythonVer}} - Python version (e.g., "3.12.0")
    {{url:https://...}} - Fetches content from URL
    {{file:path}} - Reads file content (requires workspaceRoot)
    {{file_silent:path}} - Reads file, empty string if missing

Context placeholders (set by caller):
    {{workspaceRoot}} - Working directory
    {{env}} - Environment description
    {{serverInstructions}} - MCP server instructions
    {{agentSkills}} - Agent skill descriptions

Usage:
    builder = InstructionBuilder(template="You are helpful. {{currentDate}}")
    builder.set("workspaceRoot", "/path/to/workspace")
    builder.set_resolver("serverInstructions", fetch_server_instructions)

    instruction = await builder.build()
"""

from __future__ import annotations

import platform
import re
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)

# Type aliases
Resolver = Callable[[], Awaitable[str]]  # Type alias for async resolvers
Set = set  # Preserve built-in set type before method shadowing


def _get_current_date() -> str:
    """Return current date in human-readable format."""
    return datetime.now().strftime("%d %B %Y")


def _get_host_platform() -> str:
    """Return platform information."""
    return platform.platform()


def _get_python_version() -> str:
    """Return Python version."""
    return platform.python_version()


def _fetch_url_content(url: str) -> str:
    """
    Fetch content from a URL.

    Args:
        url: The URL to fetch content from

    Returns:
        The text content from the URL

    Raises:
        requests.RequestException: If the URL cannot be fetched
    """
    import requests

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


class InstructionBuilder:
    """
    Builds instruction strings from templates with placeholder substitution.

    The builder supports two types of sources:
    - Static: String values set once (e.g., workspaceRoot)
    - Dynamic: Async resolvers called each time build() is invoked (e.g., serverInstructions)

    Built-in placeholders are automatically resolved unless overridden:
    - {{currentDate}} - Current date
    - {{hostPlatform}} - Platform info
    - {{pythonVer}} - Python version

    Special patterns:
    - {{url:https://...}} - Fetches content from URL (resolved at build time)
    - {{file:path}} - Reads file content relative to workspace (requires workspaceRoot)
    - {{file_silent:path}} - Like file: but returns empty string if missing
    """

    # Built-in values that are automatically available
    _BUILTINS: dict[str, Callable[[], str]] = {
        "currentDate": _get_current_date,
        "hostPlatform": _get_host_platform,
        "pythonVer": _get_python_version,
    }

    def __init__(self, template: str):
        """
        Initialize the builder with a template string.

        Args:
            template: The instruction template with {{placeholder}} patterns
        """
        self._template = template
        self._static: dict[str, str] = {}
        self._resolvers: dict[str, Resolver] = {}

    @property
    def template(self) -> str:
        """The original template string."""
        return self._template

    # ─────────────────────────────────────────────────────────────────────────
    # Source Registration (Fluent API)
    # ─────────────────────────────────────────────────────────────────────────

    def set(self, placeholder: str, value: str) -> "InstructionBuilder":
        """
        Set a static value for a placeholder.

        Args:
            placeholder: The placeholder name (without braces)
            value: The string value to substitute

        Returns:
            Self for method chaining
        """
        self._static[placeholder] = value
        return self

    def set_resolver(self, placeholder: str, resolver: Resolver) -> "InstructionBuilder":
        """
        Set a dynamic resolver for a placeholder.

        The resolver is called each time build() is invoked, allowing
        for dynamic content that may change between builds.

        Args:
            placeholder: The placeholder name (without braces)
            resolver: Async callable that returns the string value

        Returns:
            Self for method chaining
        """
        self._resolvers[placeholder] = resolver
        return self

    def set_many(self, values: dict[str, str]) -> "InstructionBuilder":
        """
        Set multiple static values at once.

        Args:
            values: Dict mapping placeholder names to values

        Returns:
            Self for method chaining
        """
        self._static.update(values)
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Building
    # ─────────────────────────────────────────────────────────────────────────

    async def build(self) -> str:
        """
        Build the instruction string by resolving all placeholders.

        Resolution order:
        1. {{url:...}} patterns (fetch from URL)
        2. {{file:...}} patterns (read from file, requires workspaceRoot set)
        3. {{file_silent:...}} patterns (read from file, empty if missing)
        4. Built-in values (currentDate, hostPlatform, pythonVer)
        5. Static values (override built-ins if set)
        6. Dynamic resolvers

        Returns:
            The fully resolved instruction string
        """
        result = self._template

        # 1. Resolve {{url:...}} patterns
        result = self._resolve_url_patterns(result)

        # 2. Resolve {{file:...}} patterns (strict - errors if missing)
        result = self._resolve_file_patterns(result, silent=False)

        # 3. Resolve {{file_silent:...}} patterns (returns empty if missing)
        result = self._resolve_file_patterns(result, silent=True)

        # 4. Apply built-in values (can be overridden by static values)
        for placeholder, value_fn in self._BUILTINS.items():
            if placeholder not in self._static:  # Allow override
                pattern = f"{{{{{placeholder}}}}}"
                if pattern in result:
                    result = result.replace(pattern, value_fn())

        # 5. Apply static values
        for placeholder, value in self._static.items():
            pattern = f"{{{{{placeholder}}}}}"
            if pattern in result:
                result = result.replace(pattern, value)

        # 6. Resolve dynamic values
        for placeholder, resolver in self._resolvers.items():
            pattern = f"{{{{{placeholder}}}}}"
            if pattern in result:
                try:
                    value = await resolver()
                    result = result.replace(pattern, value)
                except Exception as e:
                    logger.warning(f"Failed to resolve {{{{placeholder}}}}: {e}")
                    # Leave placeholder in place or replace with empty?
                    # For now, replace with empty to avoid confusing the LLM
                    result = result.replace(pattern, "")

        return result

    def _resolve_url_patterns(self, text: str) -> str:
        """Resolve {{url:https://...}} patterns by fetching content."""
        url_pattern = re.compile(r"\{\{url:(https?://[^}]+)\}\}")

        def replace_url(match: re.Match) -> str:
            url = match.group(1)
            try:
                return _fetch_url_content(url)
            except Exception as e:
                logger.warning(f"Failed to fetch URL {url}: {e}")
                return ""

        return url_pattern.sub(replace_url, text)

    def _resolve_file_patterns(self, text: str, *, silent: bool) -> str:
        """
        Resolve {{file:path}} or {{file_silent:path}} patterns.

        Args:
            text: The text to process
            silent: If True, use file_silent pattern and return empty on missing

        Returns:
            Text with file patterns resolved
        """
        pattern_name = "file_silent" if silent else "file"
        file_pattern = re.compile(rf"\{{\{{{pattern_name}:([^}}]+)\}}\}}")

        workspace_root = self._static.get("workspaceRoot")

        def replace_file(match: re.Match) -> str:
            file_path_str = match.group(1).strip()
            file_path = Path(file_path_str).expanduser()

            # Enforce relative paths
            if file_path.is_absolute():
                if silent:
                    return ""
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
            except FileNotFoundError as exc:
                if silent:
                    return ""
                raise AgentConfigError(
                    "Instruction file not found for template placeholder",
                    f"Placeholder: {{{{file:{file_path_str}}}}}\nMissing: {resolved_path}",
                ) from exc

        return file_pattern.sub(replace_file, text)

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def get_placeholders(self) -> Set[str]:
        """
        Extract all placeholder names from the template.

        Returns:
            Set of placeholder names (without braces)
        """
        # Match {{name}} but not {{url:...}}, {{file:...}}, {{file_silent:...}}
        pattern = re.compile(r"\{\{(?!url:|file:|file_silent:)([^}]+)\}\}")
        return set(pattern.findall(self._template))

    def get_unresolved_placeholders(self) -> Set[str]:
        """
        Get placeholders that don't have a source registered.

        Returns:
            Set of placeholder names without sources
        """
        all_placeholders = self.get_placeholders()
        registered = set(self._static.keys()) | set(self._resolvers.keys()) | set(self._BUILTINS.keys())
        return all_placeholders - registered

    def copy(self) -> "InstructionBuilder":
        """
        Create a copy of this builder with the same template and sources.

        Returns:
            A new InstructionBuilder with copied state
        """
        new_builder = InstructionBuilder(self._template)
        new_builder._static = self._static.copy()
        new_builder._resolvers = self._resolvers.copy()
        return new_builder

    def __repr__(self) -> str:
        static_count = len(self._static)
        resolver_count = len(self._resolvers)
        template_preview = (
            self._template[:50] + "..." if len(self._template) > 50 else self._template
        )
        return (
            f"InstructionBuilder(template={template_preview!r}, "
            f"static={static_count}, resolvers={resolver_count})"
        )
