"""
Utilities for converting between different prompt message formats.

This module provides utilities for converting between different serialization formats
and PromptMessageExtended objects. It includes functionality for:

1. JSON Serialization:
   - Converting PromptMessageExtended objects to MCP-compatible GetPromptResult JSON format
   - Parsing GetPromptResult JSON into PromptMessageExtended objects
   - This is ideal for programmatic use and ensures full MCP compatibility

2. Delimited Text Format:
   - Converting PromptMessageExtended objects to delimited text (---USER, ---ASSISTANT)
   - Converting resources to JSON after resource delimiter (---RESOURCE)
   - Parsing delimited text back into PromptMessageExtended objects
   - This maintains human readability for text content while preserving structure for resources
"""

import json
from typing import List

from mcp.types import (
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    PromptMessage,
    TextContent,
    TextResourceContents,
)

from fast_agent.mcp.prompts.prompt_constants import (
    ASSISTANT_DELIMITER,
    RESOURCE_DELIMITER,
    USER_DELIMITER,
)
from fast_agent.types import PromptMessageExtended

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


def to_get_prompt_result(
    messages: List[PromptMessageExtended],
) -> GetPromptResult:
    """
    Convert PromptMessageExtended objects to a GetPromptResult container.

    Args:
        messages: List of PromptMessageExtended objects

    Returns:
        GetPromptResult object containing flattened messages
    """
    # Convert multipart messages to regular PromptMessage objects
    flat_messages = []
    for message in messages:
        flat_messages.extend(message.from_multipart())

    # Create a GetPromptResult with the flattened messages
    return GetPromptResult(messages=flat_messages)



def to_get_prompt_result_json(messages: List[PromptMessageExtended]) -> str:
    """
    Convert PromptMessageExtended objects to MCP-compatible GetPromptResult JSON.

    This is a lossy conversion that flattens multipart messages and loses extended fields
    like tool_calls, channels, and stop_reason. Use for MCP server compatibility.

    Args:
        messages: List of PromptMessageExtended objects

    Returns:
        JSON string in GetPromptResult format
    """
    result = to_get_prompt_result(messages)
    result_dict = serialize_to_dict(result)
    return json.dumps(result_dict, indent=2)


def to_json(messages: List[PromptMessageExtended]) -> str:
    """
    Convert PromptMessageExtended objects directly to JSON, preserving all extended fields.

    This preserves tool_calls, tool_results, channels, and stop_reason that would be lost
    in the standard GetPromptResult conversion.

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


def from_json(json_str: str) -> List[PromptMessageExtended]:
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

    extended_messages: List[PromptMessageExtended] = []
    basic_buffer: List[PromptMessage] = []

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


def save_json(messages: List[PromptMessageExtended], file_path: str) -> None:
    """
    Save PromptMessageExtended objects to a JSON file using enhanced format.

    Uses the enhanced format that preserves tool_calls, tool_results, channels,
    and stop_reason data.

    Args:
        messages: List of PromptMessageExtended objects
        file_path: Path to save the JSON file
    """
    json_str = to_json(messages)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json_str)


def load_json(file_path: str) -> List[PromptMessageExtended]:
    """
    Load PromptMessageExtended objects from a JSON file.

    Handles both enhanced format and legacy GetPromptResult format.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of PromptMessageExtended objects
    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_str = f.read()

    return from_json(json_str)


def save_messages(messages: List[PromptMessageExtended], file_path: str) -> None:
    """
    Save PromptMessageExtended objects to a file, with format determined by file extension.

    Uses enhanced JSON format for .json files (preserves all fields) and
    delimited text format for other extensions.

    Args:
        messages: List of PromptMessageExtended objects
        file_path: Path to save the file
    """
    path_str = str(file_path).lower()

    if path_str.endswith(".json"):
        save_json(messages, file_path)
    else:
        save_delimited(messages, file_path)


def load_messages(file_path: str) -> List[PromptMessageExtended]:
    """
    Load PromptMessageExtended objects from a file, with format determined by file extension.

    Uses JSON format for .json files and delimited text format for other extensions.

    Args:
        file_path: Path to the file

    Returns:
        List of PromptMessageExtended objects
    """
    path_str = str(file_path).lower()

    if path_str.endswith(".json"):
        return load_json(file_path)
    else:
        return load_delimited(file_path)


# -------------------------------------------------------------------------
# Delimited Text Format Functions
# -------------------------------------------------------------------------


def multipart_messages_to_delimited_format(
    messages: List[PromptMessageExtended],
    user_delimiter: str = USER_DELIMITER,
    assistant_delimiter: str = ASSISTANT_DELIMITER,
    resource_delimiter: str = RESOURCE_DELIMITER,
    combine_text: bool = True,  # Set to False to maintain backward compatibility
) -> List[str]:
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
            text_contents = []

            # First, add all text content
            for content in message.content:
                if content.type == "text":
                    # Collect text content to combine
                    text_contents.append(content.text)

            # Add combined text content if any exists
            if text_contents:
                delimited_content.append("\n\n".join(text_contents))

            # Then add resources and images
            for content in message.content:
                if content.type != "text":
                    # Resource or image - add delimiter and JSON
                    delimited_content.append(resource_delimiter)

                    # Convert to dictionary using proper JSON mode
                    content_dict = serialize_to_dict(content)

                    # Add to delimited content as JSON
                    delimited_content.append(json.dumps(content_dict, indent=2))
        else:
            # Don't combine text contents - preserve each content part in sequence
            for content in message.content:
                if content.type == "text":
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
) -> List[PromptMessageExtended]:
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
    messages = []

    current_role = None
    text_contents = []  # List of TextContent
    resource_contents = []  # List of EmbeddedResource or ImageContent
    collecting_json = False
    json_lines = []
    collecting_text = False
    text_lines = []

    # Check if this is a legacy format (pre-JSON serialization)
    legacy_format = resource_delimiter in content and '"type":' not in content

    # Add a condition to ensure we process the first user message properly
    # This is the key fix: We need to process the first line correctly
    if lines and lines[0].strip() == user_delimiter:
        current_role = "user"
        collecting_text = True

    # Process each line
    for line in lines[1:] if lines else []:  # Skip the first line if already processed above
        line_stripped = line.strip()

        # Handle role delimiters
        if line_stripped == user_delimiter or line_stripped == assistant_delimiter:
            # Save previous message if it exists
            if current_role is not None and (text_contents or resource_contents or text_lines):
                # If we were collecting text, add it to the text contents
                if collecting_text and text_lines:
                    text_contents.append(TextContent(type="text", text="\n".join(text_lines)))
                    text_lines = []

                # Create content list with text parts first, then resource parts
                combined_content = []

                # Filter out any empty text content items
                filtered_text_contents = [tc for tc in text_contents if tc.text.strip() != ""]

                combined_content.extend(filtered_text_contents)
                combined_content.extend(resource_contents)

                messages.append(
                    PromptMessageExtended(
                        role=current_role,
                        content=combined_content,
                    )
                )

            # Start a new message
            current_role = "user" if line_stripped == user_delimiter else "assistant"
            text_contents = []
            resource_contents = []
            collecting_json = False
            json_lines = []
            collecting_text = False
            text_lines = []

        # Handle resource delimiter
        elif line_stripped == resource_delimiter:
            # If we were collecting text, add it to text contents
            if collecting_text and text_lines:
                text_contents.append(TextContent(type="text", text="\n".join(text_lines)))
                text_lines = []

            # Switch to collecting JSON or legacy format
            collecting_text = False
            collecting_json = True
            json_lines = []

        # Process content based on context
        elif current_role is not None:
            if collecting_json:
                # Collect JSON data
                json_lines.append(line)

                # For legacy format or files where resources are just plain text
                if legacy_format and line_stripped and not line_stripped.startswith("{"):
                    # This is probably a legacy resource reference like a filename
                    resource_uri = line_stripped
                    if not resource_uri.startswith("resource://"):
                        resource_uri = f"resource://fast-agent/{resource_uri}"

                    # Create a simple resource with just the URI
                    # For legacy format, we don't have the actual content, just the reference
                    resource = EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri=resource_uri,
                            mimeType="text/plain",
                            text="",  # Legacy format doesn't include content
                        ),
                    )
                    resource_contents.append(resource)
                    collecting_json = False
                    json_lines = []
                    continue

                # Try to parse the JSON to see if we have a complete object
                try:
                    json_text = "\n".join(json_lines)
                    json_data = json.loads(json_text)

                    # Successfully parsed JSON
                    content_type = json_data.get("type")

                    if content_type == "resource":
                        # Create resource object using model_validate
                        resource = EmbeddedResource.model_validate(json_data)
                        resource_contents.append(resource)  # Add to resource contents
                    elif content_type == "image":
                        # Create image object using model_validate
                        image = ImageContent.model_validate(json_data)
                        resource_contents.append(image)  # Add to resource contents

                    # Reset JSON collection
                    collecting_json = False
                    json_lines = []

                except json.JSONDecodeError:
                    # Not a complete JSON object yet, keep collecting
                    pass
            else:
                # Regular text content
                if not collecting_text:
                    collecting_text = True
                    text_lines = []

                text_lines.append(line)

    # Handle any remaining content
    if current_role is not None:
        # Add any remaining text
        if collecting_text and text_lines:
            text_contents.append(TextContent(type="text", text="\n".join(text_lines)))

        # Add the final message if it has content
        if text_contents or resource_contents:
            # Create content list with text parts first, then resource parts
            combined_content = []

            # Filter out any empty text content items
            filtered_text_contents = [tc for tc in text_contents if tc.text.strip() != ""]

            combined_content.extend(filtered_text_contents)
            combined_content.extend(resource_contents)

            messages.append(
                PromptMessageExtended(
                    role=current_role,
                    content=combined_content,
                )
            )

    return messages


def save_delimited(
    messages: List[PromptMessageExtended],
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

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(delimited_content))


def load_delimited(
    file_path: str,
    user_delimiter: str = USER_DELIMITER,
    assistant_delimiter: str = ASSISTANT_DELIMITER,
    resource_delimiter: str = RESOURCE_DELIMITER,
) -> List[PromptMessageExtended]:
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
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return delimited_format_to_extended_messages(
        content,
        user_delimiter,
        assistant_delimiter,
        resource_delimiter,
    )
