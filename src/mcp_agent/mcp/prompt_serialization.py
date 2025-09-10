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

from fast_agent.types import PromptMessageExtended
from mcp.types import (
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from mcp_agent.mcp.prompts.prompt_constants import (
    ASSISTANT_DELIMITER,
    RESOURCE_DELIMITER,
    USER_DELIMITER,
)

# -------------------------------------------------------------------------
# JSON Serialization Functions
# -------------------------------------------------------------------------


def multipart_messages_to_get_prompt_result(
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


def multipart_messages_to_json(messages: List[PromptMessageExtended]) -> str:
    """
    Convert PromptMessageExtended objects to a pure JSON string in GetPromptResult format.

    This approach preserves all data and structure exactly as is, compatible with
    the MCP GetPromptResult type.

    Args:
        messages: List of PromptMessageExtended objects

    Returns:
        JSON string representation with GetPromptResult container
    """
    # First convert to GetPromptResult
    result = multipart_messages_to_get_prompt_result(messages)

    # Convert to dictionary using model_dump with proper JSON mode
    result_dict = result.model_dump(by_alias=True, mode="json", exclude_none=True)

    # Convert to JSON string
    return json.dumps(result_dict, indent=2)


def json_to_extended_messages(json_str: str) -> List[PromptMessageExtended]:
    """
    Parse a JSON string in GetPromptResult format into PromptMessageExtended objects.

    Args:
        json_str: JSON string representation of GetPromptResult

    Returns:
        List of PromptMessageExtended objects
    """
    # Parse JSON to dictionary
    result_dict = json.loads(json_str)

    # Parse as GetPromptResult
    result = GetPromptResult.model_validate(result_dict)

    # Convert to multipart messages
    return PromptMessageExtended.to_extended(result.messages)


def save_messages_to_json_file(messages: List[PromptMessageExtended], file_path: str) -> None:
    """
    Save PromptMessageExtended objects to a JSON file.

    Args:
        messages: List of PromptMessageExtended objects
        file_path: Path to save the JSON file
    """
    json_str = multipart_messages_to_json(messages)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json_str)


def load_messages_from_json_file(file_path: str) -> List[PromptMessageExtended]:
    """
    Load PromptMessageExtended objects from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of PromptMessageExtended objects
    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_str = f.read()

    return json_to_extended_messages(json_str)


def save_messages_to_file(messages: List[PromptMessageExtended], file_path: str) -> None:
    """
    Save PromptMessageExtended objects to a file, with format determined by file extension.

    Uses GetPromptResult JSON format for .json files (fully MCP compatible) and
    delimited text format for other extensions.

    Args:
        messages: List of PromptMessageExtended objects
        file_path: Path to save the file
    """
    path_str = str(file_path).lower()

    if path_str.endswith(".json"):
        # Use GetPromptResult JSON format for .json files (fully MCP compatible)
        save_messages_to_json_file(messages, file_path)
    else:
        # Use delimited text format for other extensions
        save_messages_to_delimited_file(messages, file_path)


def load_messages_from_file(file_path: str) -> List[PromptMessageExtended]:
    """
    Load PromptMessageExtended objects from a file, with format determined by file extension.

    Uses GetPromptResult JSON format for .json files (fully MCP compatible) and
    delimited text format for other extensions.

    Args:
        file_path: Path to the file

    Returns:
        List of PromptMessageExtended objects
    """
    path_str = str(file_path).lower()

    if path_str.endswith(".json"):
        # Use GetPromptResult JSON format for .json files (fully MCP compatible)
        return load_messages_from_json_file(file_path)
    else:
        # Use delimited text format for other extensions
        return load_messages_from_delimited_file(file_path)


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
                    content_dict = content.model_dump(by_alias=True, mode="json", exclude_none=True)

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
                    content_dict = content.model_dump(by_alias=True, mode="json", exclude_none=True)

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
                    resource = EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri=resource_uri,
                            mimeType="text/plain",
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


def save_messages_to_delimited_file(
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


def load_messages_from_delimited_file(
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
