"""
Prompt Template Module

Handles prompt templating, variable extraction, and substitution for the prompt server.
Provides clean, testable classes for managing template substitution.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from fast_agent.core.template_render import (
    extract_template_variables,
    render_template_text,
)
from fast_agent.mcp.message_roles import MESSAGE_ROLE_NAMES, MessageRole, is_message_role
from fast_agent.mcp.prompts.prompt_constants import (
    DEFAULT_DELIMITER_MAP,
    RESOURCE_DELIMITER,
)


class PromptMetadata(BaseModel):
    """Metadata about a prompt file"""

    name: str
    description: str
    template_variables: set[str] = set()
    resource_paths: list[str] = []
    file_path: Path


@dataclass(slots=True)
class _TemplateParseState:
    sections: list["PromptContent"] = field(default_factory=list)
    current_role: MessageRole | None = None
    current_content_lines: list[str] = field(default_factory=list)
    current_resources: list[str] = field(default_factory=list)
    preamble_lines: list[str] = field(default_factory=list)

    def append_preamble_section(self) -> None:
        preamble_text = "\n".join(self.preamble_lines).strip()
        if preamble_text:
            self.sections.append(PromptContent(text=preamble_text, role="user", resources=[]))
        self.preamble_lines = []

    def append_current_section(self) -> None:
        if self.current_role is None or not self.current_content_lines:
            return
        self.sections.append(
            PromptContent(
                text="\n".join(self.current_content_lines).strip(),
                role=self.current_role,
                resources=self.current_resources,
            )
        )

    def start_section(self, role: MessageRole) -> None:
        if self.current_role is None and self.preamble_lines:
            self.append_preamble_section()
        self.append_current_section()
        self.current_role = role
        self.current_content_lines = []
        self.current_resources = []


class PromptContent(BaseModel):
    """Content of a prompt, which may include template variables"""

    text: str
    role: MessageRole = "user"
    resources: list[str] = Field(default_factory=list)

    @field_validator("role")
    @classmethod
    def validate_role(cls, role: str) -> MessageRole:
        """Validate that the role is a known value"""
        if not is_message_role(role):
            raise ValueError(f"Invalid role: {role}. Must be one of: {MESSAGE_ROLE_NAMES}")
        return role

    def apply_substitutions(self, context: dict[str, Any]) -> "PromptContent":
        """Apply variable substitutions to the text and resources"""

        # Apply substitutions to resource paths
        substituted_resources = [
            render_template_text(resource, context).text for resource in self.resources
        ]

        return PromptContent(
            text=render_template_text(self.text, context).text,
            role=self.role,
            resources=substituted_resources,
        )


class PromptTemplate:
    """
    A template for a prompt that can have variables substituted.
    """

    def __init__(
        self,
        template_text: str,
        delimiter_map: dict[str, str] | None = None,
        template_file_path: Path | None = None,
    ) -> None:
        """
        Initialize a prompt template.

        Args:
            template_text: The text of the template
            delimiter_map: Optional map of delimiters to roles (e.g. {"---USER": "user"})
            template_file_path: Optional path to the template file (for resource resolution)
        """
        self.template_text = template_text
        self.template_file_path = template_file_path
        self.delimiter_map = delimiter_map or DEFAULT_DELIMITER_MAP
        self._template_variables = self._extract_template_variables(template_text)
        self._parsed_content = self._parse_template()

    @property
    def template_variables(self) -> set[str]:
        """Get the template variables in this template"""
        return self._template_variables

    @property
    def content_sections(self) -> list[PromptContent]:
        """Get the parsed content sections"""
        return self._parsed_content

    def apply_substitutions(self, context: dict[str, Any]) -> list[PromptContent]:
        """
        Apply variable substitutions to the template.

        Args:
            context: Dictionary of variable names to values

        Returns:
            List of PromptContent with substitutions applied
        """
        # Create a new list with substitutions applied to each section
        return [section.apply_substitutions(context) for section in self._parsed_content]

    def _extract_template_variables(self, text: str) -> set[str]:
        """Extract template variables from text using regex"""
        return extract_template_variables(text)

    def _parse_template(self) -> list[PromptContent]:
        """
        Parse the template into sections based on delimiters.
        If no delimiters are found, treat the entire template as a single user message.

        Resources are now collected within their parent sections, keeping the same role.
        """
        lines = self.template_text.split("\n")
        delimiter_values = set(self.delimiter_map.keys())
        if not _has_delimiter(lines, delimiter_values):
            return [PromptContent(text=self.template_text, role="user", resources=[])]

        state = _TemplateParseState()
        i = 0
        while i < len(lines):
            i = self._parse_template_line(lines, i, state)

        state.append_current_section()
        return state.sections

    def _parse_template_line(
        self,
        lines: list[str],
        index: int,
        state: _TemplateParseState,
    ) -> int:
        line = lines[index]
        role_type = self.delimiter_map.get(line.strip())
        if role_type == "resource":
            return _parse_resource_delimiter(lines, index, state)
        if is_message_role(role_type):
            state.start_section(role_type)
        elif state.current_role is not None:
            state.current_content_lines.append(line)
        else:
            state.preamble_lines.append(line)
        return index + 1


def _has_delimiter(lines: list[str], delimiter_values: set[str]) -> bool:
    return any(line.strip() in delimiter_values for line in lines)


def _parse_resource_delimiter(
    lines: list[str],
    index: int,
    state: _TemplateParseState,
) -> int:
    next_index = index + 1
    if next_index < len(lines):
        state.current_resources.append(lines[next_index].strip())
        return next_index + 1
    return next_index


class PromptTemplateLoader:
    """
    Loads and processes prompt templates from files.
    """

    def __init__(self, delimiter_map: dict[str, str] | None = None) -> None:
        """
        Initialize the loader with optional custom delimiters.

        Args:
            delimiter_map: Optional map of delimiters to roles
        """
        self.delimiter_map = delimiter_map or DEFAULT_DELIMITER_MAP

    def load_from_file(self, file_path: Path) -> PromptTemplate:
        """
        Load a prompt template from a file.

        Args:
            file_path: Path to the template file

        Returns:
            A PromptTemplate object
        """
        with Path(file_path).open("r", encoding="utf-8") as f:
            content = f.read()

        return PromptTemplate(content, self.delimiter_map, template_file_path=file_path)

    def get_metadata(self, file_path: Path) -> PromptMetadata:
        """
        Analyze a prompt file to extract metadata and template variables.

        Args:
            file_path: Path to the template file

        Returns:
            PromptMetadata with information about the template
        """
        template = self.load_from_file(file_path)
        lines = template.template_text.split("\n")
        delimiter_values = set(self.delimiter_map.keys())
        description = _template_metadata_description(
            lines=lines,
            delimiter_map=self.delimiter_map,
            delimiter_values=delimiter_values,
            prompt_name=file_path.stem,
        )
        resource_paths = _template_resource_paths(lines, delimiter_map=self.delimiter_map)

        return PromptMetadata(
            name=file_path.stem,
            description=description,
            template_variables=template.template_variables,
            resource_paths=resource_paths,
            file_path=file_path,
        )


def _template_metadata_description(
    *,
    lines: list[str],
    delimiter_map: dict[str, str],
    delimiter_values: set[str],
    prompt_name: str,
) -> str:
    if not _has_delimiter(lines, delimiter_values):
        return _simple_template_description(lines, prompt_name=prompt_name)

    first_role, first_content_index = _first_delimited_content(delimiter_map, lines)
    if first_role is None or first_content_index is None:
        return prompt_name

    preview = _template_preview_after_delimiter(
        lines,
        start_index=first_content_index,
        delimiter_values=delimiter_values,
    )
    if preview is None:
        return prompt_name
    return f"[{first_role.upper()}] {preview}"


def _simple_template_description(lines: list[str], *, prompt_name: str) -> str:
    first_line = lines[0].strip() if lines else ""
    if len(first_line) < 60 and "{{" not in first_line and "}}" not in first_line:
        return first_line
    return f"Simple prompt: {prompt_name}"


def _first_delimited_content(
    delimiter_map: dict[str, str],
    lines: list[str],
) -> tuple[str | None, int | None]:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped in delimiter_map:
            return delimiter_map[stripped], index + 1
    return None, None


def _template_preview_after_delimiter(
    lines: list[str],
    *,
    start_index: int,
    delimiter_values: set[str],
) -> str | None:
    if start_index >= len(lines):
        return None

    preview_lines: list[str] = []
    for line in lines[start_index : start_index + 10]:
        stripped = line.strip()
        if stripped and stripped not in delimiter_values:
            preview_lines.append(stripped)
            if len(preview_lines) >= 3:
                break
    if not preview_lines:
        return None

    preview = " ".join(preview_lines)
    if len(preview) > 50:
        return preview[:47] + "..."
    return preview


def _template_resource_paths(
    lines: list[str],
    *,
    delimiter_map: dict[str, str],
) -> list[str]:
    resource_delimiter = next(
        (k for k, v in delimiter_map.items() if v == "resource"), RESOURCE_DELIMITER
    )
    return [
        lines[index + 1].strip()
        for index, line in enumerate(lines)
        if line.strip() == resource_delimiter
        and index + 1 < len(lines)
        and lines[index + 1].strip()
    ]
