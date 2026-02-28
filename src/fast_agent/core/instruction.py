"""
InstructionBuilder - Builds instruction strings from templates and sources.

This module provides a clean abstraction for constructing agent instructions
from templates with placeholder substitution. Sources can be static values
or dynamic resolvers that are called at build time.

Built-in placeholders (automatically resolved):
    {{internal:resource_id}} - Loads packaged internal resource content
    {{currentDate}} - Current date in "17 December 2025" format
    {{hostPlatform}} - Platform info (e.g., "Linux-6.6.0-x86_64")
    {{pythonVer}} - Python version (e.g., "3.12.0")
    {{url:https://...}} - Fetches content from URL
    {{file:path}} - Reads file content (requires workspaceRoot)
    {{file_silent:path}} - Reads file, empty string if missing

Context placeholders (set by caller):
    {{workspaceRoot}} - Working directory
    {{env}} - Environment description
    {{agentName}} - Current agent name
    {{agentType}} - Current agent type
    {{agentCardPath}} - Source AgentCard path (if available)
    {{agentCardDir}} - Source AgentCard directory (if available)
    {{serverInstructions}} - MCP server instructions
    {{agentSkills}} - Agent skill descriptions
    {{agentInternalResources}} - Internal resource index

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
from importlib.resources import files
from pathlib import Path
from typing import Awaitable, Callable

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.template_escape import protect_escaped_braces, restore_escaped_braces

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


def _load_internal_resource(resource_id: str) -> str:
    """Load a packaged internal resource by ID."""
    normalized_id = resource_id.strip()
    if not normalized_id:
        raise AgentConfigError(
            "Invalid internal resource placeholder",
            "Resource ID must not be empty",
        )

    # Source checkout fallback for local development/testing.
    source_resource_path = (
        Path(__file__).resolve().parents[3]
        / "resources"
        / "shared"
        / f"{normalized_id}.md"
    )
    if source_resource_path.is_file():
        return source_resource_path.read_text(encoding="utf-8")

    resource_path = (
        files("fast_agent")
        .joinpath("resources")
        .joinpath("shared")
        .joinpath(f"{normalized_id}.md")
    )
    if resource_path.is_file():
        return resource_path.read_text(encoding="utf-8")

    raise AgentConfigError(
        "Unknown internal resource for template placeholder",
        f"Placeholder: {{{{internal:{normalized_id}}}}}",
    )


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
    - {{internal:resource_id}} - Loads packaged internal resource content
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

    def __init__(self, template: str, *, source: str | None = None):
        """
        Initialize the builder with a template string.

        Args:
            template: The instruction template with {{placeholder}} patterns
            source: Optional label for diagnostics (agent name, card, etc.)
        """
        self._template = template
        self._source = source
        self._static: dict[str, str] = {}
        self._resolvers: dict[str, Resolver] = {}

    @property
    def template(self) -> str:
        """The original template string."""
        return self._template

    @property
    def source(self) -> str | None:
        """Optional source label for diagnostics (agent name, card, etc.)."""
        return self._source

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
        1. {{internal:...}} patterns (load packaged internal resources)
        2. {{url:...}} patterns (fetch from URL)
        3. {{file:...}} patterns (read from file, requires workspaceRoot set)
        4. {{file_silent:...}} patterns (read from file, empty if missing)
        5. Built-in values (currentDate, hostPlatform, pythonVer)
        6. Static values (override built-ins if set)
        7. Dynamic resolvers

        Returns:
            The fully resolved instruction string
        """
        result = protect_escaped_braces(self._template)

        # 1. Resolve {{internal:...}} patterns
        result = self._resolve_internal_patterns(result)

        # 2. Resolve {{url:...}} patterns
        result = self._resolve_url_patterns(result)

        # 3. Resolve {{file:...}} patterns (strict - errors if missing)
        result = self._resolve_file_patterns(result, silent=False)

        # 4. Resolve {{file_silent:...}} patterns (returns empty if missing)
        result = self._resolve_file_patterns(result, silent=True)

        # 5. Apply built-in values (can be overridden by static values)
        for placeholder, value_fn in self._BUILTINS.items():
            if placeholder not in self._static:  # Allow override
                pattern = f"{{{{{placeholder}}}}}"
                if pattern in result:
                    result = result.replace(pattern, value_fn())

        # 6. Apply static values
        for placeholder, value in self._static.items():
            pattern = f"{{{{{placeholder}}}}}"
            if pattern in result:
                result = result.replace(pattern, value)

        # 7. Resolve dynamic values
        for placeholder, resolver in self._resolvers.items():
            pattern = f"{{{{{placeholder}}}}}"
            if pattern in result:
                try:
                    value = await resolver()
                    result = result.replace(pattern, value)
                except Exception as e:
                    logger.warning(
                        f"Failed to resolve {{{{{placeholder}}}}}: {e}",
                        placeholder=placeholder,
                        source=self._source,
                    )
                    # Leave placeholder in place or replace with empty?
                    # For now, replace with empty to avoid confusing the LLM
                    result = result.replace(pattern, "")

        return restore_escaped_braces(result)

    def _resolve_internal_patterns(self, text: str) -> str:
        """Resolve {{internal:resource_id}} patterns from packaged resources."""
        internal_pattern = re.compile(r"\{\{internal:([^}]+)\}\}")

        # Allow internal resources to include other internal resources while guarding
        # against accidental recursion loops.
        max_internal_include_depth = 10
        result = text

        for _ in range(max_internal_include_depth):
            if not internal_pattern.search(result):
                return result

            def replace_internal(match: re.Match) -> str:
                resource_id = match.group(1)
                internal_text = _load_internal_resource(resource_id)
                return protect_escaped_braces(internal_text)

            result = internal_pattern.sub(replace_internal, result)

        raise AgentConfigError(
            "Internal resource include depth exceeded",
            "Detected recursive or excessively deep {{internal:...}} include chain",
        )

    def _resolve_url_patterns(self, text: str) -> str:
        """Resolve {{url:https://...}} patterns by fetching content."""
        url_pattern = re.compile(r"\{\{url:(https?://[^}]+)\}\}")

        def replace_url(match: re.Match) -> str:
            url = match.group(1)
            try:
                return _fetch_url_content(url)
            except Exception as e:
                logger.warning(
                    f"Failed to fetch URL {url}: {e}",
                    url=url,
                    source=self._source,
                )
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

        def _should_fallback(path: Path) -> bool:
            if not path.parts:
                return False
            return path.parts[0] in {".fast-agent", ".dev"}

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
                if _should_fallback(file_path) and not resolved_path.exists():
                    fallback_path = (Path.cwd() / file_path).resolve()
                    if fallback_path.exists():
                        resolved_path = fallback_path
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
        # Match {{name}} but not special patterns
        pattern = re.compile(
            r"(?<!\\)\{\{(?!url:|file:|file_silent:|internal:)([^}]+)\}\}"
        )
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
        new_builder = InstructionBuilder(self._template, source=self._source)
        new_builder._static = self._static.copy()
        new_builder._resolvers = self._resolvers.copy()
        return new_builder

    def __repr__(self) -> str:
        static_count = len(self._static)
        resolver_count = len(self._resolvers)
        template_preview = (
            self._template[:50] + "..." if len(self._template) > 50 else self._template
        )
        source_label = f", source={self._source!r}" if self._source else ""
        return (
            f"InstructionBuilder(template={template_preview!r}, "
            f"static={static_count}, resolvers={resolver_count}{source_label})"
        )
