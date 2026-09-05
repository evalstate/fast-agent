"""Persisted Responses cache state; no provider input-index cache.

Assistant channels survive history copies and session serialization. Each records
the effective effort of that request (the API reports only the baseline).
"""

from collections.abc import Callable
from typing import Any

from mcp_types import TextContent
from pydantic import BaseModel

from fast_agent.constants import FAST_AGENT_TOOL_MEDIA_MESSAGE
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

CACHE_STATE_CHANNEL = "fast-agent-responses-cache-state"


class ResponsesCacheState(BaseModel):
    model: str
    key: str
    effort: str | None = None

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


def prepare_cached_request(
    messages: list[PromptMessageExtended],
    params: RequestParams,
    *,
    key: str,
    default_effort: str | None,
    convert: Callable[[PromptMessageExtended], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], ResponsesCacheState, RequestParams]:
    metadata = dict(params.metadata or {})
    model = metadata.get("model") or params.model
    if not isinstance(model, str):
        raise ValueError("Responses cache state requires a model")
    extra = metadata.get("extra_body")
    if (
        isinstance(extra, dict)
        and "model" in extra
        and (model == "gpt-6-astra" or extra["model"] == "gpt-6-astra")
    ):
        raise ValueError("Set model directly in metadata, not extra_body")
    effort = None
    if model == "gpt-6-astra":
        validate_cache_overrides(metadata)
        reasoning = metadata.get("reasoning", {})
        if not isinstance(reasoning, dict):
            raise ValueError("Astra reasoning must be an object")
        effort = reasoning.get("effort", default_effort)
        if effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("Astra reasoning effort must be low, medium, high, xhigh or max")
    items, state, baseline = prepare_cached_input(
        messages, model=model, key=key, effort=effort, convert=convert
    )
    metadata.setdefault("prompt_cache_key", state.key)
    if not isinstance(metadata["prompt_cache_key"], str):
        raise ValueError("Responses prompt_cache_key must be a string")
    state.key = metadata["prompt_cache_key"]
    if baseline is not None:
        metadata["reasoning"] = {
            "summary": "auto",
            **metadata.get("reasoning", {}),
            "effort": baseline,
        }
    return items, state, params.model_copy(update={"metadata": metadata})


def prepare_cached_input(
    messages: list[PromptMessageExtended],
    *,
    model: str,
    key: str,
    effort: str | None,
    convert: Callable[[PromptMessageExtended], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], ResponsesCacheState, str | None]:
    """Replay effort at user boundaries, using surviving history as the baseline.

    Only Astra callers supply effort. Tool-result continuations keep the current
    effective effort; a setting change takes effect at the next user message.
    """
    states = [read_cache_state(message) for message in messages]
    matching = [state for state in states if state and state.model == model]
    first = matching[0] if matching else None
    if matching:
        key = matching[-1].key
    baseline = first.effort if first and effort is not None else effort
    effective = baseline
    items: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        converted = convert(message)
        if effort is not None and _is_user_turn(message):
            # The next assistant records this request's effective effort. Stop at
            # another user turn, so imported/unannotated history is not relabeled.
            desired = effective
            for following_index in range(index + 1, len(messages)):
                following = messages[following_index]
                state = states[following_index]
                if following.role == "assistant":
                    if state and state.model == model and state.effort is not None:
                        desired = state.effort
                    break
                if _is_user_turn(following):
                    break
            else:
                desired = effort
            for item in converted:
                if item.get("role") == "user":
                    if desired != effective:
                        items.append(
                            {"type": "configuration_update", "reasoning": {"effort": desired}}
                        )
                        effective = desired
                items.append(item)
        else:
            items.extend(converted)
    return items, ResponsesCacheState(model=model, key=key, effort=effective), baseline


def _is_user_turn(message: PromptMessageExtended) -> bool:
    return (
        message.role == "user"
        and bool(message.content)
        and not message.tool_results
        and FAST_AGENT_TOOL_MEDIA_MESSAGE not in (message.channels or {})
    )


def validate_cache_overrides(metadata: dict[str, Any]) -> None:
    """Reject overrides that bypass the managed history or rewrite its prefix."""
    for source in (metadata, metadata.get("extra_body", {})):
        if not isinstance(source, dict):
            raise ValueError("Responses extra_body must be an object")
        if any(name in source for name in ("input", "previous_response_id")):
            raise ValueError("Managed Astra effort updates require fast-agent message history")
        if source.get("truncation") not in (None, "disabled") or source.get("context_management"):
            raise ValueError("Astra effort updates cannot use automatic truncation or compaction")
        reasoning = source.get("reasoning", {})
        if isinstance(reasoning, dict) and reasoning.get("mode") not in (None, "standard"):
            raise ValueError("Astra effort updates require standard single-agent mode")
        multi_agent = source.get("multi_agent", {})
        if not isinstance(multi_agent, dict) or multi_agent.get("enabled"):
            raise ValueError("Astra effort updates require standard single-agent mode")
    extra = metadata.get("extra_body", {})
    if any(name in extra for name in ("reasoning", "prompt_cache_key", "model")):
        raise ValueError(
            "Set reasoning, prompt_cache_key and model directly in metadata, not extra_body"
        )
