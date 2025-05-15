"""
Prompt Template Module

Handles prompt templating, variable extraction, and substitution for the prompt server.
Provides clean, testable classes for managing template substitution.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

from mcp.types import (
    EmbeddedResource,
    TextContent,
    TextResourceContents,
)
from pydantic import BaseModel, field_validator

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.prompt_serialization import (
    multipart_messages_to_delimited_format,
)
from mcp_agent.mcp.prompts.prompt_constants import (
    ASSISTANT_DELIMITER,
    DEFAULT_DELIMITER_MAP,
    RESOURCE_DELIMITER,
    USER_DELIMITER,
)


class PromptMetadata(BaseModel):
    """Metadata about a prompt file"""

    name: str
    description: str
    template_variables: Set[str] = set()
    resource_paths: List[str] = []
    file_path: Path


# Define valid message roles for better type safety
MessageRole = Literal["user", "assistant"]


class PromptContent(BaseModel):
    """Content of a prompt, which may include template variables"""

    text: str
    role: str = "user"
    resources: List[str] = []

    @field_validator("role")
    @classmethod
    def validate_role(cls, role: str) -> str:
        """Validate that the role is a known value"""
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role}. Must be one of: user, assistant")
        return role

    def apply_substitutions(self, context: Dict[str, Any]) -> "PromptContent":
        """Apply variable substitutions to the text and resources"""

        # Define placeholder pattern once to avoid repetition
        def make_placeholder(key: str) -> str:
            return f"{{{{{key}}}}}"

        # Apply substitutions to text
        result = self.text
        for key, value in context.items():
            result = result.replace(make_placeholder(key), str(value))

        # Apply substitutions to resource paths
        substituted_resources = []
        for resource in self.resources:
            substituted = resource
            for key, value in context.items():
                substituted = substituted.replace(make_placeholder(key), str(value))
            substituted_resources.append(substituted)

        return PromptContent(text=result, role=self.role, resources=substituted_resources)


class PromptTemplate:
    """
    A template for a prompt that can have variables substituted.
    """

    def __init__(
        self,
        template_text: str,
        delimiter_map: Optional[Dict[str, str]] = None,
        template_file_path: Optional[Path] = None,
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

    @classmethod
    def from_multipart_messages(
        cls,
        messages: List[PromptMessageMultipart],
        delimiter_map: Optional[Dict[str, str]] = None,
    ) -> "PromptTemplate":
        """
        Create a PromptTemplate from a list of PromptMessageMultipart objects.

        Args:
            messages: List of PromptMessageMultipart objects
            delimiter_map: Optional map of delimiters to roles

        Returns:
            A new PromptTemplate object
        """
        # Use default delimiter map if none provided
        delimiter_map = delimiter_map or DEFAULT_DELIMITER_MAP

        # Convert to delimited format
        delimited_content = multipart_messages_to_delimited_format(
            messages,
            user_delimiter=next(
                (k for k, v in delimiter_map.items() if v == "user"), USER_DELIMITER
            ),
            assistant_delimiter=next(
                (k for k, v in delimiter_map.items() if v == "assistant"),
                ASSISTANT_DELIMITER,
            ),
        )

        # Join into a single string
        content = "\n".join(delimited_content)

        # Create and return the template
        return cls(content, delimiter_map)

    @property
    def template_variables(self) -> Set[str]:
        """Get the template variables in this template"""
        return self._template_variables

    @property
    def content_sections(self) -> List[PromptContent]:
        """Get the parsed content sections"""
        return self._parsed_content

    def apply_substitutions(self, context: Dict[str, Any]) -> List[PromptContent]:
        """
        Apply variable substitutions to the template.

        Args:
            context: Dictionary of variable names to values

        Returns:
            List of PromptContent with substitutions applied
        """
        # Create a new list with substitutions applied to each section
        return [section.apply_substitutions(context) for section in self._parsed_content]

    def apply_substitutions_to_multipart(
        self, context: Dict[str, Any]
    ) -> List[PromptMessageMultipart]:
        """
        Apply variable substitutions to the template and return PromptMessageMultipart objects.

        Args:
            context: Dictionary of variable names to values

        Returns:
            List of PromptMessageMultipart objects with substitutions applied
        """
        # First create a substituted template
        content_sections = self.apply_substitutions(context)

        # Convert content sections to multipart messages
        multiparts = []
        for section in content_sections:
            # Handle text content
            content_items = [TextContent(type="text", text=section.text)]

            # Handle resources (if any)
            for resource_path in section.resources:
                # In a real implementation, you would load the resource here
                # For now, we'll just create a placeholder
                content_items.append(
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri=f"resource://fast-agent/{resource_path}",
                            mimeType="text/plain",
                            text=f"Content of {resource_path}",
                        ),
                    )
                )

            multiparts.append(PromptMessageMultipart(role=section.role, content=content_items))

        return multiparts

    def _extract_template_variables(self, text: str) -> Set[str]:
        """Extract template variables from text using regex"""
        variable_pattern = r"{{([^}]+)}}"
        matches = re.findall(variable_pattern, text)
        return set(matches)

    def to_multipart_messages(self) -> List[PromptMessageMultipart]:
        """
        Convert this template to a list of PromptMessageMultipart objects.

        Returns:
            List of PromptMessageMultipart objects
        """
        multiparts = []

        for section in self._parsed_content:
            # Convert each section to a multipart message
            content_items = [TextContent(type="text", text=section.text)]

            # Add any resources as embedded resources
            for resource_path in section.resources:
                # In a real implementation, you would determine the MIME type
                # and load the resource appropriately. Here we'll just use a placeholder.
                content_items.append(
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri=f"resource://{resource_path}",
                            mimeType="text/plain",
                            text=f"Content of {resource_path}",
                        ),
                    )
                )

            multiparts.append(PromptMessageMultipart(role=section.role, content=content_items))

        return multiparts

    def _parse_template(self) -> List[PromptContent]:
        """
        Parse the template into sections based on delimiters.
        If no delimiters are found, treat the entire template as a single user message.

        Resources are now collected within their parent sections, keeping the same role.
        """
        lines = self.template_text.split("\n")

        # Check if we're in simple mode (no delimiters)
        first_non_empty_line = next((line for line in lines if line.strip()), "")
        delimiter_values = set(self.delimiter_map.keys())

        is_simple_mode = first_non_empty_line and first_non_empty_line not in delimiter_values

        if is_simple_mode:
            # Simple mode: treat the entire content as a single user message
            return [PromptContent(text=self.template_text, role="user", resources=[])]

        # Standard mode with delimiters
        sections = []
        current_role = None
        current_content = ""
        current_resources = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check if we hit a delimiter
            if line.strip() in self.delimiter_map:
                role_type = self.delimiter_map[line.strip()]

                # If we're moving to a new user/assistant section (not resource)
                if role_type != "resource":
                    # Save the previous section if it exists
                    if current_role is not None and current_content:
                        sections.append(
                            PromptContent(
                                text=current_content.strip(),
                                role=current_role,
                                resources=current_resources,
                            )
                        )

                    # Start a new section
                    current_role = role_type
                    current_content = ""
                    current_resources = []

                # Handle resource delimiters within sections
                elif role_type == "resource" and i + 1 < len(lines):
                    resource_path = lines[i + 1].strip()
                    current_resources.append(resource_path)
                    # Skip the resource path line
                    i += 1

            # If we're in a section, add to the current content
            elif current_role is not None:
                current_content += line + "\n"

            i += 1

        # Add the last section if there is one
        if current_role is not None and current_content:
            sections.append(
                PromptContent(
                    text=current_content.strip(),
                    role=current_role,
                    resources=current_resources,
                )
            )

        return sections


class PromptTemplateLoader:
    """
    Loads and processes prompt templates from files.
    """

    def __init__(self, delimiter_map: Optional[Dict[str, str]] = None) -> None:
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
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return PromptTemplate(content, self.delimiter_map, template_file_path=file_path)

    def load_from_multipart(self, messages: List[PromptMessageMultipart]) -> PromptTemplate:
        """
        Create a PromptTemplate from a list of PromptMessageMultipart objects.

        Args:
            messages: List of PromptMessageMultipart objects

        Returns:
            A PromptTemplate object
        """
        # Use the class method directly to avoid code duplication
        return PromptTemplate.from_multipart_messages(messages, self.delimiter_map)

    def get_metadata(self, file_path: Path) -> PromptMetadata:
        """
        Analyze a prompt file to extract metadata and template variables.

        Args:
            file_path: Path to the template file

        Returns:
            PromptMetadata with information about the template
        """
        template = self.load_from_file(file_path)

        # Generate a description based on content
        lines = template.template_text.split("\n")
        first_non_empty_line = next((line for line in lines if line.strip()), "")

        # Check if we're in simple mode
        is_simple_mode = first_non_empty_line and first_non_empty_line not in self.delimiter_map

        if is_simple_mode:
            # In simple mode, use first line as description if it seems like one
            first_line = lines[0].strip() if lines else ""
            if len(first_line) < 60 and "{{" not in first_line and "}}" not in first_line:
                description = first_line
            else:
                description = f"Simple prompt: {file_path.stem}"
        else:
            # Regular mode - find text after first delimiter for the description
            description = file_path.stem

            # Look for first delimiter and role
            first_role = None
            first_content_index = None

            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped in self.delimiter_map:
                    first_role = self.delimiter_map[stripped]
                    first_content_index = i + 1
                    break

            if first_role and first_content_index and first_content_index < len(lines):
                # Get up to 3 non-empty lines after the delimiter for a preview
                preview_lines = []
                for j in range(first_content_index, min(first_content_index + 10, len(lines))):
                    stripped = lines[j].strip()
                    if stripped and stripped not in self.delimiter_map:
                        preview_lines.append(stripped)
                        if len(preview_lines) >= 3:
                            break

                if preview_lines:
                    preview = " ".join(preview_lines)
                    if len(preview) > 50:
                        preview = preview[:47] + "..."
                    # Include role in the description but not the filename
                    description = f"[{first_role.upper()}] {preview}"

        # Extract resource paths from all sections that come after RESOURCE delimiters
        resource_paths = []
        resource_delimiter = next(
            (k for k, v in self.delimiter_map.items() if v == "resource"), RESOURCE_DELIMITER
        )
        for i, line in enumerate(lines):
            if line.strip() == resource_delimiter:
                if i + 1 < len(lines) and lines[i + 1].strip():
                    resource_paths.append(lines[i + 1].strip())

        return PromptMetadata(
            name=file_path.stem,
            description=description,
            template_variables=template.template_variables,
            resource_paths=resource_paths,
            file_path=file_path,
        )
