"""Persisted Responses cache state; no provider input-index cache.

Assistant channels survive history copies and session serialization. Each records
the effective effort and response format of that request (the API reports only the
request-level baseline).
"""

from collections.abc import Callable
from typing import Any

from mcp_types import TextContent
from pydantic import BaseModel

from fast_agent.constants import FAST_AGENT_TOOL_MEDIA_MESSAGE
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

CACHE_STATE_CHANNEL = "fast-agent-responses-cache-state"
# Copilot checkpoint replay remains limited to the previously verified Astra route.
ASTRA_MODEL = "gpt-6-astra"
# Verified Responses/Codex models supporting prefix-preserving configuration updates.
EFFORT_UPDATE_MODELS = (ASTRA_MODEL, "gpt-6.2-astra", "gpt-6.2-sol")
# Copilot forwards (and validates) configuration updates for GPT-6.2, verified 2026-11-18.
COPILOT_UPDATE_MODELS = ("gpt-6.2-astra", "gpt-6.2-sol")
TEXT_FORMAT: dict[str, Any] = {"type": "text"}
# Format updates accept only these; other formats stay request-level.
_UPDATABLE_FORMATS = ("json_schema", "text")


class ResponsesCacheState(BaseModel):
    model: str
    key: str
    effort: str | None = None
    text_format: dict[str, Any] | None = None
    """Response format set by configuration updates; ``None`` is plain text."""

    def attach(self, message: PromptMessageExtended) -> None:
        message.channels = {
            **(message.channels or {}),
            CACHE_STATE_CHANNEL: [TextContent(type="text", text=self.model_dump_json())],
        }


def read_cache_state(message: PromptMessageExtended) -> ResponsesCacheState | None:
    for block in (message.channels or {}).get(CACHE_STATE_CHANNEL, ()):
        if isinstance(block, TextContent):
            return ResponsesCacheState.model_validate_json(block.text)
    return None


def requested_model(params: RequestParams) -> object:
    """The request's model: a ``metadata["model"]`` override, else ``params.model``."""
    return (params.metadata or {}).get("model") or params.model


def requested_astra_effort(params: RequestParams, default_effort: str | None) -> str:
    """Desired Astra effort: ``metadata["reasoning"]["effort"]``, else the default."""
    reasoning = (params.metadata or {}).get("reasoning", {})
    if not isinstance(reasoning, dict):
        raise ModelConfigError("Astra reasoning must be an object")
    effort = reasoning.get("effort", default_effort)
    if effort not in ("low", "medium", "high", "xhigh", "max"):
        raise ModelConfigError("Astra reasoning effort must be low, medium, high, xhigh or max")
    return effort


def prepare_cached_request(
    messages: list[PromptMessageExtended],
    params: RequestParams,
    *,
    key: str,
    default_effort: str | None,
    convert: Callable[[PromptMessageExtended], list[dict[str, Any]]],
    text_format: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], ResponsesCacheState, RequestParams]:
    """Build cached input; ``text_format`` is the request's normalized response format.

    For configuration-update models a JSON schema is sent as an update and removed from
    the returned params, so the request-level ``text.format`` never changes.
    """
    metadata = dict(params.metadata or {})
    model = requested_model(params)
    if not isinstance(model, str):
        raise ModelConfigError("Responses cache state requires a model")
    extra = metadata.get("extra_body")
    if (
        isinstance(extra, dict)
        and "model" in extra
        and (model in EFFORT_UPDATE_MODELS or extra["model"] in EFFORT_UPDATE_MODELS)
    ):
        raise ModelConfigError("Set model directly in metadata, not extra_body")
    effort = None
    managed_format = None
    if model in EFFORT_UPDATE_MODELS:
        validate_cache_overrides(metadata)
        effort = requested_astra_effort(params, default_effort)
        if text_format is None or text_format.get("type") in _UPDATABLE_FORMATS:
            managed_format = text_format or TEXT_FORMAT
    items, state, baseline = prepare_cached_input(
        messages,
        model=model,
        key=key,
        effort=effort,
        convert=convert,
        text_format=managed_format,
    )
    metadata.setdefault("prompt_cache_key", state.key)
    if not isinstance(metadata["prompt_cache_key"], str):
        raise ModelConfigError("Responses prompt_cache_key must be a string")
    state.key = metadata["prompt_cache_key"]
    if baseline is not None:
        metadata["reasoning"] = {
            "summary": "auto",
            **metadata.get("reasoning", {}),
            "effort": baseline,
        }
    update: dict[str, Any] = {"metadata": metadata}
    if managed_format is not None:
        update["response_format"] = None
    return items, state, params.model_copy(update=update)


def prepare_cached_input(
    messages: list[PromptMessageExtended],
    *,
    model: str,
    key: str,
    effort: str | None,
    convert: Callable[[PromptMessageExtended], list[dict[str, Any]]],
    text_format: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], ResponsesCacheState, str | None]:
    """Replay configuration updates, using surviving history as the baseline.

    Only models supporting configuration updates supply effort. Tool-result continuations
    keep the current effective effort; a setting change takes effect at the next user
    message. A ``text_format`` (``TEXT_FORMAT`` for plain text) enables format updates:
    a format change is placed before the user message, or appended after tool results
    when it changes mid-turn (deferred structured output), so it never edits the prefix.
    """
    states = [read_cache_state(message) for message in messages]
    matching = [state for state in states if state and state.model == model]
    first = matching[0] if matching else None
    if matching:
        key = matching[-1].key
    baseline = first.effort if first and effort is not None else effort
    effective = baseline
    # Explicit compaction reverts to the request-level (plain text) format.
    effective_format = TEXT_FORMAT
    # Effort in force when the latest compaction ran, used if no user turn follows.
    carried: str | None = None
    items: list[dict[str, Any]] = []

    def response_format(response: int) -> dict[str, Any]:
        """Format of the request answered by ``messages[response]``, or this request."""
        if response == len(messages):
            assert text_format is not None
            return text_format
        state = states[response]
        if state and state.model == model:
            return state.text_format or TEXT_FORMAT
        return effective_format

    for index, message in enumerate(messages):
        converted = convert(message)
        if any(item.get("type") == "compaction" for item in converted):
            if effort is not None:
                # OpenAI: after compaction, send a fresh update before the next user message.
                carried, effective = effective, None
            effective_format = TEXT_FORMAT
        if (effort is not None or text_format is not None) and _is_user_turn(message):
            # The next assistant records this request's effective effort. Stop at
            # another user turn, so imported/unannotated history is not relabeled.
            # After a checkpoint (effective unknown), unannotated turns fall back to
            # the effort in force at compaction, so the update is always re-sent.
            desired = effective if effective is not None else (carried or baseline)
            response = _response_index(messages, index)
            if response == len(messages):
                desired = effort
            elif response is not None:
                state = states[response]
                if state and state.model == model and state.effort is not None:
                    desired = state.effort
            desired_format = (
                effective_format
                if response is None or text_format is None
                else response_format(response)
            )
            for item in converted:
                if item.get("role") == "user":
                    update: dict[str, Any] = {}
                    if effort is not None and desired != effective:
                        update["reasoning"] = {"effort": desired}
                        effective = desired
                    if text_format is not None and desired_format != effective_format:
                        update["text"] = {"format": desired_format}
                        effective_format = desired_format
                    if update:
                        items.append({"type": "configuration_update", **update})
                items.append(item)
            continue
        items.extend(converted)
        if (
            text_format is not None
            and message.role == "user"
            and (index + 1 == len(messages) or messages[index + 1].role == "assistant")
        ):
            desired_format = response_format(index + 1)
            if desired_format != effective_format:
                items.append({"type": "configuration_update", "text": {"format": desired_format}})
                effective_format = desired_format
    if effective is None:
        effective = carried
    state = ResponsesCacheState(
        model=model,
        key=key,
        effort=effective,
        text_format=None if effective_format == TEXT_FORMAT else effective_format,
    )
    return items, state, baseline


def _response_index(messages: list[PromptMessageExtended], index: int) -> int | None:
    """The assistant answering the user turn at ``index``; ``len(messages)`` for this
    request, or ``None`` when another user turn intervenes (unannotated history)."""
    for following in range(index + 1, len(messages)):
        if messages[following].role == "assistant":
            return following
        if _is_user_turn(messages[following]):
            return None
    return len(messages)


def is_tool_continuation(message: PromptMessageExtended) -> bool:
    """Tool results (or their staged media) continuing the current turn."""
    return bool(message.tool_results) or FAST_AGENT_TOOL_MEDIA_MESSAGE in (message.channels or {})


def _is_user_turn(message: PromptMessageExtended) -> bool:
    return message.role == "user" and bool(message.content) and not is_tool_continuation(message)


def validate_cache_overrides(metadata: dict[str, Any]) -> None:
    """Reject overrides that bypass the managed history or rewrite its prefix."""
    for source in (metadata, metadata.get("extra_body", {})):
        if not isinstance(source, dict):
            raise ModelConfigError("Responses extra_body must be an object")
        if any(name in source for name in ("input", "previous_response_id")):
            raise ModelConfigError(
                "Managed Astra effort updates require fast-agent message history"
            )
        if source.get("truncation") not in (None, "disabled") or source.get("context_management"):
            raise ModelConfigError(
                "Astra effort updates cannot use automatic truncation or compaction"
            )
        reasoning = source.get("reasoning", {})
        if isinstance(reasoning, dict) and reasoning.get("mode") not in (None, "standard"):
            raise ModelConfigError("Astra effort updates require standard single-agent mode")
        multi_agent = source.get("multi_agent", {})
        if not isinstance(multi_agent, dict) or multi_agent.get("enabled"):
            raise ModelConfigError("Astra effort updates require standard single-agent mode")
    extra = metadata.get("extra_body", {})
    if any(name in extra for name in ("reasoning", "prompt_cache_key", "model")):
        raise ModelConfigError(
            "Set reasoning, prompt_cache_key and model directly in metadata, not extra_body"
        )
