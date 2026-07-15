import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mcp.types import (
    ContentBlock,
    EmbeddedResource,
    PromptMessage,
    TextContent,
    TextResourceContents,
)

from fast_agent.constants import FAST_AGENT_USAGE
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.template_render import extract_template_variables, render_template_text
from fast_agent.interfaces import AgentProtocol
from fast_agent.io.source_resolver import materialized_text_source
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import TurnUsage, UsageAccumulator, UsageReport, UsageSchema
from fast_agent.mcp import mime_utils, resource_utils
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.message_roles import (
    DEFAULT_MESSAGE_ROLE,
    MessageRole,
    coerce_message_role,
    is_message_role,
)
from fast_agent.mcp.prompts.prompt_template import (
    PromptContent,
)
from fast_agent.types import PromptMessageExtended
from fast_agent.utils.numeric import nonnegative_int_or_none
from fast_agent.utils.text import strip_casefold

logger = get_logger("prompt_load")
_RESPONSES_USAGE_PROVIDERS = frozenset(
    {Provider.RESPONSES, Provider.CODEX_RESPONSES, Provider.OPENRESPONSES}
)


def cast_message_role(role: str) -> MessageRole:
    """Cast a string role to a MessageRole literal type"""
    if is_message_role(role):
        return role
    # Default to user if the role is invalid
    logger.warning(f"Invalid message role: {role}, defaulting to '{DEFAULT_MESSAGE_ROLE}'")
    return coerce_message_role(role)


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
                resource_content = resource_utils.load_resource_content(resource_path, prompt_files)

                # Create and add the resource message
                resource_message = create_resource_message(
                    resource_path,
                    resource_content.content,
                    resource_content.mime_type,
                    resource_content.is_binary,
                    role,
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
) -> PromptMessage:
    """Create a resource message with the specified content and role"""
    if mime_utils.is_image_mime_type(mime_type):
        # For images, create an ImageContent
        image_content = resource_utils.create_image_content(data=content, mime_type=mime_type)
        return PromptMessage(role=role, content=image_content)

    # For other resources, create an EmbeddedResource
    embedded_resource = resource_utils.create_embedded_resource(
        resource_path, content, mime_type, is_binary
    )
    return PromptMessage(role=role, content=embedded_resource)


def _render_text_content(content: TextContent, arguments: Mapping[str, str]) -> TextContent:
    rendered = render_template_text(content.text, arguments).text
    if rendered == content.text:
        return content
    return content.model_copy(update={"text": rendered})


def _render_embedded_text_resource(
    content: EmbeddedResource, arguments: Mapping[str, str]
) -> EmbeddedResource:
    resource = content.resource
    if not isinstance(resource, TextResourceContents):
        return content
    rendered = render_template_text(resource.text, arguments).text
    if rendered == resource.text:
        return content
    return content.model_copy(update={"resource": resource.model_copy(update={"text": rendered})})


def _render_message_templates(
    messages: list[PromptMessageExtended], arguments: Mapping[str, str]
) -> list[PromptMessageExtended]:
    rendered_messages: list[PromptMessageExtended] = []
    for message in messages:
        content: list[ContentBlock] = []
        changed = False
        for item in message.content:
            rendered_item = item
            if isinstance(item, TextContent):
                rendered_item = _render_text_content(item, arguments)
            elif isinstance(item, EmbeddedResource):
                rendered_item = _render_embedded_text_resource(item, arguments)
            changed = changed or rendered_item is not item
            content.append(rendered_item)
        rendered_messages.append(
            message.model_copy(update={"content": content}) if changed else message
        )
    return rendered_messages


def _message_template_variables(messages: list[PromptMessageExtended]) -> set[str]:
    variables: set[str] = set()
    for message in messages:
        for item in message.content:
            if isinstance(item, TextContent):
                variables.update(extract_template_variables(item.text))
            elif isinstance(item, EmbeddedResource) and isinstance(
                item.resource, TextResourceContents
            ):
                variables.update(extract_template_variables(item.resource.text))
    return variables


def prompt_file_template_variables(file: Path | str) -> set[str]:
    """Return value-only ``{{placeholder}}`` names from a prompt file."""
    with materialized_text_source(file, label="prompt file") as source_file:
        if strip_casefold(source_file.suffix) == ".json":
            from fast_agent.mcp.prompt_serialization import load_messages

            return _message_template_variables(load_messages(str(source_file)))

        from fast_agent.mcp.prompts.prompt_template import PromptTemplateLoader

        return PromptTemplateLoader().load_from_file(source_file).template_variables


def load_prompt(
    file: Path | str,
    arguments: Mapping[str, str] | None = None,
) -> list[PromptMessageExtended]:
    """
    Load a prompt from a file and return as PromptMessageExtended objects.

    The loader uses file extension to determine the format:
    - .json files are loaded using enhanced format that preserves tool_calls, channels, etc.
    - All other files are loaded using the template-based delimited format with resource loading

    Args:
        file: Path to the prompt file (Path object or string)
        arguments: Optional values for ``{{placeholder}}`` substitutions

    Returns:
        List of PromptMessageExtended objects with full conversation state
    """
    with materialized_text_source(file, label="prompt file") as source_file:
        if strip_casefold(source_file.suffix) == ".json":
            # JSON files use the serialization module directly
            from fast_agent.mcp.prompt_serialization import load_messages

            messages = load_messages(str(source_file))
            return _render_message_templates(messages, arguments) if arguments else messages

        # Non-JSON files need template processing for resource loading
        from fast_agent.mcp.prompts.prompt_template import PromptTemplateLoader

        loader = PromptTemplateLoader()
        template = loader.load_from_file(source_file)
        content_sections = (
            template.apply_substitutions(dict(arguments))
            if arguments
            else template.content_sections
        )

        # Render the template to get the messages
        messages = create_messages_with_resources(
            content_sections,
            [source_file],  # Pass the file path for resource resolution
        )

        # Convert to PromptMessageExtended
        return PromptMessageExtended.to_extended(messages)


def _history_messages(
    history: Path | str | list[PromptMessageExtended],
) -> list[PromptMessageExtended]:
    if isinstance(history, list):
        return history
    return load_prompt(history)


def _copy_messages(messages: list[PromptMessageExtended]) -> list[PromptMessageExtended]:
    return [message.model_copy(deep=True) for message in messages]


def _snapshot_usage_state(agent: AgentProtocol) -> UsageAccumulator | None:
    usage_accumulator = agent.usage_accumulator
    if usage_accumulator is None:
        return None
    return usage_accumulator.model_copy(deep=True)


def _restore_usage_state(agent: AgentProtocol, snapshot: UsageAccumulator | None) -> None:
    if snapshot is None:
        return

    usage_accumulator = agent.usage_accumulator
    if usage_accumulator is None:
        return

    usage_accumulator.turns = [turn.model_copy(deep=True) for turn in snapshot.turns]
    usage_accumulator.model = snapshot.model
    usage_accumulator.last_cache_activity_time = snapshot.last_cache_activity_time
    usage_accumulator.set_context_window_size(snapshot.context_window_size)


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


def _usage_report_from_payload(payload: dict[str, Any]) -> UsageReport | None:
    try:
        return UsageReport.model_validate(payload)
    except ValueError:
        return _legacy_responses_usage_report(payload)


def _legacy_responses_usage_report(payload: dict[str, Any]) -> UsageReport | None:
    """Translate released unversioned Responses usage at the history-load boundary."""

    if "schema" in payload:
        return None
    turn = payload.get("turn")
    if not isinstance(turn, dict):
        return None

    provider_value = turn.get("provider")
    model = turn.get("model")
    if not isinstance(provider_value, str) or not isinstance(model, str):
        return None
    try:
        provider = Provider(provider_value)
    except ValueError:
        return None
    if provider not in _RESPONSES_USAGE_PROVIDERS:
        return None

    prompt_tokens = nonnegative_int_or_none(turn.get("input_tokens"))
    completion_tokens = nonnegative_int_or_none(turn.get("output_tokens"))
    if prompt_tokens is None or completion_tokens is None:
        return None

    cache_usage = turn.get("cache_usage")
    cache = cache_usage if isinstance(cache_usage, dict) else {}
    snapshot: dict[str, object] = {
        "provider": provider,
        "usage_schema": UsageSchema.OPENAI_RESPONSES,
        "model": model,
        "prompt": {
            "total": prompt_tokens,
            "cache_read": nonnegative_int_or_none(cache.get("cache_hit_tokens")) or 0,
            "cache_write": nonnegative_int_or_none(cache.get("cache_write_tokens")) or 0,
            "tool_use": turn.get("tool_use_tokens", 0),
        },
        "completion": {
            "total": completion_tokens,
            "reasoning": turn.get("reasoning_tokens", 0),
        },
        "tool_calls": turn.get("tool_calls", 0),
        "service_tier": turn.get("service_tier"),
        "raw_usage": payload.get("raw_usage"),
    }
    timestamp = turn.get("timestamp")
    if isinstance(timestamp, int | float) and not isinstance(timestamp, bool):
        snapshot["timestamp"] = float(timestamp)
    try:
        attempt = TurnUsage.model_validate(snapshot)
    except ValueError:
        return None
    return UsageReport(provider_attempts=[attempt])


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

    usage_accumulator.reset()

    if not payloads:
        return None

    reports = [
        report
        for payload in payloads
        if (report := _usage_report_from_payload(payload)) is not None
    ]
    if not reports:
        return None

    history_provider = reports[-1].final_attempt.provider
    current_provider = llm.provider
    switched_responses_provider = (
        history_provider in _RESPONSES_USAGE_PROVIDERS and history_provider != current_provider
    )

    history_model = reports[-1].final_attempt.model
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

    for report in reports:
        for attempt in report.provider_attempts:
            usage_accumulator.add_turn(attempt)

    return None


def load_transcript_into_agent(
    agent: AgentProtocol,
    history: Path | str | list[PromptMessageExtended],
) -> None:
    """Replace visible conversation transcript without rebuilding usage state."""
    messages = _history_messages(history)
    usage_snapshot = _snapshot_usage_state(agent)

    agent.clear(clear_prompts=True)
    agent.message_history.extend(_copy_messages(messages))

    _restore_usage_state(agent, usage_snapshot)


def rehydrate_usage_from_history(
    agent: AgentProtocol,
    history: Path | str | list[PromptMessageExtended],
) -> str | None:
    """Rebuild usage state from saved history without mutating transcript state."""
    messages = _history_messages(history)
    return _rehydrate_responses_usage(agent, messages)
