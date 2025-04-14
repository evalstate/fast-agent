from pathlib import Path
from typing import List, Literal

from mcp.server.fastmcp.prompts.base import (
    AssistantMessage,
    Message,
    UserMessage,
)
from mcp.types import PromptMessage, TextContent

from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp import mime_utils, resource_utils
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.prompts.prompt_template import (
    PromptContent,
    PromptTemplate,
    PromptTemplateLoader,
)

# Define message role type
MessageRole = Literal["user", "assistant"]
logger = get_logger("prompt_load")


def cast_message_role(role: str) -> MessageRole:
    """Cast a string role to a MessageRole literal type"""
    if role == "user" or role == "assistant":
        return role  # type: ignore
    # Default to user if the role is invalid
    logger.warning(f"Invalid message role: {role}, defaulting to 'user'")
    return "user"


def create_messages_with_resources(
    content_sections: List[PromptContent], prompt_files: List[Path]
) -> List[PromptMessage]:
    """
    Create a list of messages from content sections, with resources properly handled.

    This implementation produces one message for each content section's text,
    followed by separate messages for each resource (with the same role type
    as the section they belong to).

    Args:
        content_sections: List of PromptContent objects
        prompt_files: List of prompt files (to help locate resource files)

    Returns:
        List of Message objects
    """

    messages = []

    for section in content_sections:
        # Convert to our literal type for role
        role = cast_message_role(section.role)

        # Add the text message
        messages.append(create_content_message(section.text, role))

        # Add resource messages with the same role type as the section
        for resource_path in section.resources:
            try:
                # Load resource with information about its type
                resource_content, mime_type, is_binary = resource_utils.load_resource_content(
                    resource_path, prompt_files
                )

                # Create and add the resource message
                resource_message = create_resource_message(
                    resource_path, resource_content, mime_type, is_binary, role
                )
                messages.append(resource_message)
            except Exception as e:
                logger.error(f"Error loading resource {resource_path}: {e}")

    return messages


def create_content_message(text: str, role: MessageRole) -> PromptMessage:
    """Create a text content message with the specified role"""
    return PromptMessage(role=role, content=TextContent(type="text", text=text))


def create_resource_message(
    resource_path: str, content: str, mime_type: str, is_binary: bool, role: MessageRole
) -> Message:
    """Create a resource message with the specified content and role"""
    message_class = UserMessage if role == "user" else AssistantMessage

    if mime_utils.is_image_mime_type(mime_type):
        # For images, create an ImageContent
        image_content = resource_utils.create_image_content(data=content, mime_type=mime_type)
        return message_class(content=image_content)
    else:
        # For other resources, create an EmbeddedResource
        embedded_resource = resource_utils.create_embedded_resource(
            resource_path, content, mime_type, is_binary
        )
        return message_class(content=embedded_resource)


def load_prompt(file: Path) -> List[PromptMessage]:
    """
    Load a prompt from a file and return as PromptMessage objects.

    The loader uses file extension to determine the format:
    - .json files are loaded as MCP SDK compatible GetPromptResult JSON format
    - All other files are loaded using the template-based delimited format

    Args:
        file: Path to the prompt file

    Returns:
        List of PromptMessage objects
    """
    file_str = str(file).lower()

    if file_str.endswith(".json"):
        # Handle JSON format as GetPromptResult
        import json

        from mcp.types import GetPromptResult

        # Load JSON directly into GetPromptResult
        with open(file, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Parse as GetPromptResult object
        result = GetPromptResult.model_validate(json_data)

        # Return the messages directly
        return result.messages
    else:
        # Template-based format (delimited text)
        template: PromptTemplate = PromptTemplateLoader().load_from_file(file)
        return create_messages_with_resources(template.content_sections, [file])


def load_prompt_multipart(file: Path) -> List[PromptMessageMultipart]:
    """
    Load a prompt from a file and return as PromptMessageMultipart objects.

    The loader uses file extension to determine the format:
    - .json files are loaded as MCP SDK compatible GetPromptResult JSON format
    - All other files are loaded using the template-based delimited format

    Args:
        file: Path to the prompt file

    Returns:
        List of PromptMessageMultipart objects
    """
    # First load as regular PromptMessage objects
    messages = load_prompt(file)
    # Then convert to multipart messages
    return PromptMessageMultipart.to_multipart(messages)
