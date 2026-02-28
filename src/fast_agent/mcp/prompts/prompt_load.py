import json
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp.prompts.base import (
    AssistantMessage,
    Message,
    UserMessage,
)
from mcp.types import PromptMessage, TextContent

from fast_agent.constants import FAST_AGENT_USAGE
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import AgentProtocol
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import FastAgentUsage, TurnUsage
from fast_agent.mcp import mime_utils, resource_utils
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompts.prompt_template import (
    PromptContent,
)
from fast_agent.types import PromptMessageExtended

# Define message role type
MessageRole = Literal["user", "assistant"]
logger = get_logger("prompt_load")
_RESPONSES_USAGE_PROVIDERS = frozenset(
    {Provider.RESPONSES, Provider.CODEX_RESPONSES, Provider.OPENRESPONSES}
)


def cast_message_role(role: str) -> MessageRole:
    """Cast a string role to a MessageRole literal type"""
    if role == "user" or role == "assistant":
        return role  # type: ignore
    # Default to user if the role is invalid
    logger.warning(f"Invalid message role: {role}, defaulting to 'user'")
    return "user"


def create_messages_with_resources(
    content_sections: list[PromptContent], prompt_files: list[Path]
) -> list[PromptMessage]:
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


def load_prompt(file: Path | str) -> list[PromptMessageExtended]:
    """
    Load a prompt from a file and return as PromptMessageExtended objects.

    The loader uses file extension to determine the format:
    - .json files are loaded using enhanced format that preserves tool_calls, channels, etc.
    - All other files are loaded using the template-based delimited format with resource loading

    Args:
        file: Path to the prompt file (Path object or string)

    Returns:
        List of PromptMessageExtended objects with full conversation state
    """
    if isinstance(file, str):
        file = Path(file)
    path_str = str(file).lower()

    if path_str.endswith(".json"):
        # JSON files use the serialization module directly
        from fast_agent.mcp.prompt_serialization import load_messages

        return load_messages(str(file))
    else:
        # Non-JSON files need template processing for resource loading
        from fast_agent.mcp.prompts.prompt_template import PromptTemplateLoader

        loader = PromptTemplateLoader()
        template = loader.load_from_file(file)

        # Render the template without arguments to get the messages
        messages = create_messages_with_resources(
            template.content_sections,
            [file],  # Pass the file path for resource resolution
        )

        # Convert to PromptMessageExtended
        return PromptMessageExtended.to_extended(messages)


def load_prompt_as_get_prompt_result(file: Path):
    """
    Load a prompt from a file and convert to GetPromptResult format for MCP compatibility.

    This loses extended fields (tool_calls, channels, etc.) but provides
    compatibility with MCP prompt servers.

    Args:
        file: Path to the prompt file

    Returns:
        GetPromptResult object for MCP compatibility
    """
    from fast_agent.mcp.prompt_serialization import to_get_prompt_result

    # Load with full data
    messages = load_prompt(file)

    # Convert to GetPromptResult (loses extended fields)
    return to_get_prompt_result(messages)


def _extract_usage_payloads(messages: list[PromptMessageExtended]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in messages:
        channels = message.channels
        if not isinstance(channels, dict):
            continue

        usage_blocks = channels.get(FAST_AGENT_USAGE)
        if not isinstance(usage_blocks, list):
            continue

        for block in usage_blocks:
            block_text = get_text(block)
            if not block_text:
                continue
            try:
                payload = json.loads(block_text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payloads.append(payload)

    return payloads


def _payload_model(payload: dict[str, Any]) -> str | None:
    summary = payload.get("summary")
    if isinstance(summary, dict):
        summary_model = summary.get("model")
        if isinstance(summary_model, str) and summary_model:
            return summary_model

    turn = payload.get("turn")
    if isinstance(turn, dict):
        turn_model = turn.get("model")
        if isinstance(turn_model, str) and turn_model:
            return turn_model

    return None


def _payload_provider(payload: dict[str, Any]) -> Provider | None:
    turn = payload.get("turn")
    if not isinstance(turn, dict):
        return None

    provider_value = turn.get("provider")
    if isinstance(provider_value, Provider):
        return provider_value
    if not isinstance(provider_value, str) or not provider_value:
        return None

    try:
        return Provider(provider_value)
    except ValueError:
        return None


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _turn_usage_from_payload(payload: dict[str, Any]) -> TurnUsage | None:
    turn_data = payload.get("turn")
    if not isinstance(turn_data, dict):
        return None

    turn_snapshot = dict(turn_data)
    input_tokens = _coerce_int(turn_snapshot.get("input_tokens"))
    output_tokens = _coerce_int(turn_snapshot.get("output_tokens"))
    turn_snapshot["raw_usage"] = FastAgentUsage(
        input_chars=input_tokens,
        output_chars=output_tokens,
        model_type="rehydrated",
    )

    try:
        return TurnUsage.model_validate(turn_snapshot)
    except Exception:
        return None


def _rehydrate_responses_usage(
    agent: AgentProtocol,
    messages: list[PromptMessageExtended],
) -> str | None:
    llm = agent.llm
    if llm is None or llm.provider not in _RESPONSES_USAGE_PROVIDERS:
        return None

    payloads = _extract_usage_payloads(messages)
    usage_accumulator = llm.usage_accumulator
    if usage_accumulator is None:
        return None

    usage_accumulator.turns = []
    usage_accumulator.model = None
    usage_accumulator.last_cache_activity_time = None

    if not payloads:
        return None

    history_provider = _payload_provider(payloads[-1])
    current_provider = llm.provider
    switched_responses_provider = (
        history_provider in _RESPONSES_USAGE_PROVIDERS
        and history_provider != current_provider
    )

    history_model = _payload_model(payloads[-1])
    current_model = llm.model_name
    if (
        not switched_responses_provider
        and isinstance(history_model, str)
        and history_model
        and isinstance(current_model, str)
        and current_model
        and history_model != current_model
    ):
        notice = (
            f"Model changed from {history_model} to {current_model} -- usage info not available"
        )
        logger.warning(notice)
        return notice

    for payload in payloads:
        turn_usage = _turn_usage_from_payload(payload)
        if turn_usage is None:
            continue
        usage_accumulator.add_turn(turn_usage)

    return None


def load_history_into_agent(agent: AgentProtocol, file_path: Path) -> str | None:
    """
    Load conversation history directly into agent without triggering LLM call.

    This function restores saved conversation state by directly setting the
    agent's _message_history. No LLM API calls are made.

    Args:
        agent: Agent instance to restore history into (FastAgentLLM or subclass)
        file_path: Path to saved history file (JSON or template format)

    Note:
        - The agent's history is cleared before loading
        - Provider diagnostic history will be updated on the next API call
        - Templates are NOT cleared by this function

    Returns:
        Optional resume notice string when usage state cannot be restored.
    """
    messages = load_prompt(file_path)

    # Direct restoration - no LLM call
    agent.clear(clear_prompts=True)
    agent.message_history.extend(messages)

    # Note: Provider diagnostic history will be updated on next API call
    return _rehydrate_responses_usage(agent, messages)
