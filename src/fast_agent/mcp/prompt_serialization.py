"""
Utilities for converting between different prompt message formats.

This module provides utilities for converting between different serialization formats
and PromptMessageExtended objects. It includes functionality for:

1. JSON Serialization:
   - Parsing GetPromptResult JSON into PromptMessageExtended objects
   - This is ideal for programmatic use and ensures full MCP compatibility

2. Delimited Text Format:
   - Converting PromptMessageExtended objects to delimited text (---USER, ---ASSISTANT)
   - Converting resources to JSON after resource delimiter (---RESOURCE)
   - Parsing delimited text back into PromptMessageExtended objects
   - This maintains human readability for text content while preserving structure for resources
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from mcp.types import (
    AudioContent,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.mcp.message_roles import MessageRole
from fast_agent.mcp.prompts.prompt_constants import (
    ASSISTANT_DELIMITER,
    RESOURCE_DELIMITER,
    USER_DELIMITER,
)
from fast_agent.mcp.resource_utils import to_any_url
from fast_agent.types import PromptMessageExtended
from fast_agent.utils.text import strip_casefold

# -------------------------------------------------------------------------
# Serialization Helpers
# -------------------------------------------------------------------------


def serialize_to_dict(obj, exclude_none: bool = True):
    """Standardized Pydantic serialization to dictionary.

    Args:
        obj: Pydantic model object to serialize
        exclude_none: Whether to exclude None values (default: True)

    Returns:
        Dictionary representation suitable for JSON serialization
    """
    return obj.model_dump(by_alias=True, mode="json", exclude_none=exclude_none)


# -------------------------------------------------------------------------
# JSON Serialization Functions
# -------------------------------------------------------------------------


def to_json(messages: list[PromptMessageExtended]) -> str:
    """
    Convert PromptMessageExtended objects directly to JSON, preserving all extended fields.

    This preserves tool_calls, tool_results, channels, and stop_reason.

    Args:
        messages: List of PromptMessageExtended objects

    Returns:
        JSON string representation preserving all PromptMessageExtended data
    """
    # Convert each message to dict using standardized serialization
    messages_dicts = [serialize_to_dict(msg) for msg in messages]

    # Wrap in a container similar to GetPromptResult for consistency
    result_dict = {"messages": messages_dicts}

    # Convert to JSON string
    return json.dumps(result_dict, indent=2)


def from_json(json_str: str) -> list[PromptMessageExtended]:
    """
    Parse a JSON string into PromptMessageExtended objects.

    Handles both:
    - Enhanced format with full PromptMessageExtended data
    - Legacy GetPromptResult format (missing extended fields default to None)

    Args:
        json_str: JSON string representation

    Returns:
        List of PromptMessageExtended objects
    """
    # Parse JSON to dictionary
    result_dict = json.loads(json_str)

    # Extract messages array
    messages_data = result_dict.get("messages", [])

    extended_messages: list[PromptMessageExtended] = []
    basic_buffer: list[PromptMessage] = []

    def flush_basic_buffer() -> None:
        nonlocal basic_buffer
        if not basic_buffer:
            return
        extended_messages.extend(PromptMessageExtended.to_extended(basic_buffer))
        basic_buffer = []

    for msg_data in messages_data:
        content = msg_data.get("content")
        is_enhanced = isinstance(content, list)
        if is_enhanced:
            try:
                msg = PromptMessageExtended.model_validate(msg_data)
            except Exception:
                is_enhanced = False
            else:
                flush_basic_buffer()
                extended_messages.append(msg)
                continue

        try:
            basic_msg = PromptMessage.model_validate(msg_data)
        except Exception:
            continue
        basic_buffer.append(basic_msg)

    flush_basic_buffer()

    return extended_messages


def save_json(messages: list[PromptMessageExtended], file_path: str) -> None:
    """
    Save PromptMessageExtended objects to a JSON file using enhanced format.

    Uses the enhanced format that preserves tool_calls, tool_results, channels,
    and stop_reason data.

    Args:
        messages: List of PromptMessageExtended objects
        file_path: Path to save the JSON file
    """
    json_str = to_json(messages)

    with Path(file_path).open("w", encoding="utf-8") as f:
        f.write(json_str)


def load_json(file_path: str) -> list[PromptMessageExtended]:
    """
    Load PromptMessageExtended objects from a JSON file.

    Handles both enhanced format and legacy GetPromptResult format.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of PromptMessageExtended objects
    """
    with Path(file_path).open("r", encoding="utf-8") as f:
        json_str = f.read()

    try:
        return from_json(json_str)
    except json.JSONDecodeError as exc:
        raise AgentConfigError(
            f"Failed to parse JSON prompt file: {file_path}",
            str(exc),
        ) from exc


def save_messages(messages: list[PromptMessageExtended], file_path: str) -> None:
    """
    Save PromptMessageExtended objects to a file, with format determined by file extension.

    Uses enhanced JSON format for .json files (preserves all fields) and
    delimited text format for other extensions.

    Args:
        messages: List of PromptMessageExtended objects
        file_path: Path to save the file
    """
    if strip_casefold(Path(file_path).suffix) == ".json":
        save_json(messages, file_path)
        return
    save_delimited(messages, file_path)


def load_messages(file_path: str) -> list[PromptMessageExtended]:
    """
    Load PromptMessageExtended objects from a file, with format determined by file extension.

    Uses JSON format for .json files and delimited text format for other extensions.

    Args:
        file_path: Path to the file

    Returns:
        List of PromptMessageExtended objects
    """
    if strip_casefold(Path(file_path).suffix) == ".json":
        return load_json(file_path)
    return load_delimited(file_path)


# -------------------------------------------------------------------------
# Delimited Text Format Functions
# -------------------------------------------------------------------------


DelimitedContent = EmbeddedResource | ImageContent
DelimitedRole = MessageRole


@dataclass
class _DelimitedParseState:
    current_role: DelimitedRole | None = None
    text_contents: list[TextContent] = field(default_factory=list)
    resource_contents: list[DelimitedContent] = field(default_factory=list)
    collecting_json: bool = False
    json_lines: list[str] = field(default_factory=list)
    collecting_text: bool = False
    text_lines: list[str] = field(default_factory=list)

    def reset_for_role(self, role: DelimitedRole) -> None:
        self.current_role = role
        self.text_contents.clear()
        self.resource_contents.clear()
        self.collecting_json = False
        self.json_lines.clear()
        self.collecting_text = False
        self.text_lines.clear()

    def flush_text(self) -> None:
        if self.collecting_text and self.text_lines:
            self.text_contents.append(TextContent(type="text", text="\n".join(self.text_lines)))
        self.collecting_text = False
        self.text_lines.clear()

    def start_json(self) -> None:
        self.flush_text()
        self.collecting_json = True
        self.json_lines.clear()

    def append_text_line(self, line: str) -> None:
        if not self.collecting_text:
            self.collecting_text = True
            self.text_lines.clear()
        self.text_lines.append(line)

    def append_message(self, messages: list[PromptMessageExtended]) -> None:
        if self.current_role is None:
            return
        self.flush_text()
        filtered_text = [tc for tc in self.text_contents if tc.text.strip() != ""]
        combined_content = [*filtered_text, *self.resource_contents]
        if not combined_content:
            return
        messages.append(
            PromptMessageExtended(
                role=self.current_role,
                content=cast(
                    "list[TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource]",
                    combined_content,
                ),
            )
        )


def multipart_messages_to_delimited_format(
    messages: list[PromptMessageExtended],
    user_delimiter: str = USER_DELIMITER,
    assistant_delimiter: str = ASSISTANT_DELIMITER,
    resource_delimiter: str = RESOURCE_DELIMITER,
    combine_text: bool = True,  # Set to False to maintain backward compatibility
) -> list[str]:
    """
    Convert PromptMessageExtended objects to a hybrid delimited format:
    - Plain text for user/assistant text content with delimiters
    - JSON for resources after resource delimiter

    This approach maintains human readability for text content while
    preserving structure for resources.

    Args:
        messages: List of PromptMessageExtended objects
        user_delimiter: Delimiter for user messages
        assistant_delimiter: Delimiter for assistant messages
        resource_delimiter: Delimiter for resources
        combine_text: Whether to combine multiple text contents into one (default: True)

    Returns:
        List of strings representing the delimited content
    """
    delimited_content = []

    for message in messages:
        # Add role delimiter
        if message.role == "user":
            delimited_content.append(user_delimiter)
        else:
            delimited_content.append(assistant_delimiter)

        # Process content parts based on combine_text preference
        if combine_text:
            # Collect text content parts
            text_contents = [
                content.text for content in message.content if isinstance(content, TextContent)
            ]

            # Add combined text content if any exists
            if text_contents:
                delimited_content.append("\n\n".join(text_contents))

            # Then add resources and images
            for content in message.content:
                if not isinstance(content, TextContent):
                    # Resource or image - add delimiter and JSON
                    delimited_content.append(resource_delimiter)

                    # Convert to dictionary using proper JSON mode
                    content_dict = serialize_to_dict(content)

                    # Add to delimited content as JSON
                    delimited_content.append(json.dumps(content_dict, indent=2))
        else:
            # Don't combine text contents - preserve each content part in sequence
            for content in message.content:
                if isinstance(content, TextContent):
                    # Add each text content separately
                    delimited_content.append(content.text)
                else:
                    # Resource or image - add delimiter and JSON
                    delimited_content.append(resource_delimiter)

                    # Convert to dictionary using proper JSON mode
                    content_dict = serialize_to_dict(content)

                    # Add to delimited content as JSON
                    delimited_content.append(json.dumps(content_dict, indent=2))

    return delimited_content


def delimited_format_to_extended_messages(
    content: str,
    user_delimiter: str = USER_DELIMITER,
    assistant_delimiter: str = ASSISTANT_DELIMITER,
    resource_delimiter: str = RESOURCE_DELIMITER,
) -> list[PromptMessageExtended]:
    """
    Parse hybrid delimited format into PromptMessageExtended objects:
    - Plain text for user/assistant text content with delimiters
    - JSON for resources after resource delimiter

    Args:
        content: String containing the delimited content
        user_delimiter: Delimiter for user messages
        assistant_delimiter: Delimiter for assistant messages
        resource_delimiter: Delimiter for resources

    Returns:
        List of PromptMessageExtended objects
    """
    if user_delimiter not in content and assistant_delimiter not in content:
        stripped = content.strip()
        if not stripped:
            return []
        return [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text=stripped)],
            )
        ]

    lines = content.split("\n")
    messages: list[PromptMessageExtended] = []
    state = _DelimitedParseState()
    legacy_format = resource_delimiter in content and '"type":' not in content

    for line in lines:
        line_stripped = line.strip()

        if line_stripped in (user_delimiter, assistant_delimiter):
            state.append_message(messages)
            role = "user" if line_stripped == user_delimiter else "assistant"
            state.reset_for_role(role)
        elif line_stripped == resource_delimiter:
            state.start_json()
        elif state.current_role is not None:
            _consume_delimited_content_line(state, line, legacy_format=legacy_format)

    state.append_message(messages)
    return messages


def _consume_delimited_content_line(
    state: _DelimitedParseState,
    line: str,
    *,
    legacy_format: bool,
) -> None:
    if not state.collecting_json:
        state.append_text_line(line)
        return

    line_stripped = line.strip()
    state.json_lines.append(line)
    legacy_resource = _legacy_resource_from_line(line_stripped) if legacy_format else None
    if legacy_resource is not None:
        state.resource_contents.append(legacy_resource)
        state.collecting_json = False
        state.json_lines.clear()
        return

    parsed_content = _content_from_json_lines(state.json_lines)
    if parsed_content is None:
        return
    state.resource_contents.append(parsed_content)
    state.collecting_json = False
    state.json_lines.clear()


def _legacy_resource_from_line(line_stripped: str) -> EmbeddedResource | None:
    if not line_stripped or line_stripped.startswith("{"):
        return None
    resource_uri = line_stripped
    if not resource_uri.startswith("resource://"):
        resource_uri = f"resource://fast-agent/{resource_uri}"
    return EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=to_any_url(resource_uri),
            mimeType="text/plain",
            text="",
        ),
    )


def _content_from_json_lines(json_lines: list[str]) -> DelimitedContent | None:
    try:
        json_data = json.loads("\n".join(json_lines))
    except json.JSONDecodeError:
        return None

    content_type = json_data.get("type")
    if content_type == "resource":
        return EmbeddedResource.model_validate(json_data)
    if content_type == "image":
        return ImageContent.model_validate(json_data)
    return None


def save_delimited(
    messages: list[PromptMessageExtended],
    file_path: str,
    user_delimiter: str = USER_DELIMITER,
    assistant_delimiter: str = ASSISTANT_DELIMITER,
    resource_delimiter: str = RESOURCE_DELIMITER,
    combine_text: bool = True,
) -> None:
    """
    Save PromptMessageExtended objects to a file in hybrid delimited format.

    Args:
        messages: List of PromptMessageExtended objects
        file_path: Path to save the file
        user_delimiter: Delimiter for user messages
        assistant_delimiter: Delimiter for assistant messages
        resource_delimiter: Delimiter for resources
        combine_text: Whether to combine multiple text contents into one (default: True)
    """
    delimited_content = multipart_messages_to_delimited_format(
        messages,
        user_delimiter,
        assistant_delimiter,
        resource_delimiter,
        combine_text=combine_text,
    )

    with Path(file_path).open("w", encoding="utf-8") as f:
        f.write("\n".join(delimited_content))


def load_delimited(
    file_path: str,
    user_delimiter: str = USER_DELIMITER,
    assistant_delimiter: str = ASSISTANT_DELIMITER,
    resource_delimiter: str = RESOURCE_DELIMITER,
) -> list[PromptMessageExtended]:
    """
    Load PromptMessageExtended objects from a file in hybrid delimited format.

    Args:
        file_path: Path to the file
        user_delimiter: Delimiter for user messages
        assistant_delimiter: Delimiter for assistant messages
        resource_delimiter: Delimiter for resources

    Returns:
        List of PromptMessageExtended objects
    """
    with Path(file_path).open("r", encoding="utf-8") as f:
        content = f.read()

    return delimited_format_to_extended_messages(
        content,
        user_delimiter,
        assistant_delimiter,
        resource_delimiter,
    )
