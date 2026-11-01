"""Plugin hook that reviews an assistant message's tool calls before any of them run.

Reviewers run in enabled-plugin order. Each sees the calls earlier reviewers left
running and returns the ids it denies, mapped to the error text the model receives
as that call's result. Review fails closed: a reviewer that cannot be loaded, raises,
or returns something else denies every call it was asked about.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, cast

from fast_agent.core.logging.logger import get_logger
from fast_agent.plugins.models import (
    PluginToolCall,
    PluginToolCallReviewContext,
    PluginToolCallReviewFunction,
)
from fast_agent.tools.python_file_loader import (
    PythonCallableLoadMessages,
    load_callable_from_file_spec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp_types import Tool

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.plugins.models import PluginToolCallReviewSpec

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedPluginToolCallReview:
    plugin_name: str
    handler: PluginToolCallReviewFunction | None
    load_error: str | None = None


@cache
def load_plugin_tool_call_reviewers(
    specs: tuple[PluginToolCallReviewSpec, ...],
) -> tuple[LoadedPluginToolCallReview, ...]:
    """Load reviewers once per spec set; a failed load is kept so its calls are denied."""
    loaded: list[LoadedPluginToolCallReview] = []
    for spec in specs:
        if spec.load_error is not None:
            loaded.append(LoadedPluginToolCallReview(spec.plugin_name, None, spec.load_error))
            continue
        try:
            handler = load_callable_from_file_spec(
                spec.handler,
                module_name_prefix="_plugin_tool_call_review",
                messages=PythonCallableLoadMessages(
                    invalid_spec="Invalid tool-call review handler '{spec}'. Expected 'module.py:function'",
                    module_not_found="Tool-call review module file not found for '{spec}'",
                    module_spec_failed="Failed to create module spec for tool-call review handler '{spec}'",
                    import_failed="Failed to import tool-call review module for '{spec}'",
                    callable_not_found="Tool-call review function '{func_name}' not found for '{spec}'",
                    not_callable="Tool-call review target '{func_name}' is not callable for '{spec}'",
                ),
            )
        except Exception as exc:
            logger.error(
                "Failed to load tool-call review plugin",
                plugin_name=spec.plugin_name,
                handler=spec.handler,
                error=str(exc),
            )
            loaded.append(LoadedPluginToolCallReview(spec.plugin_name, None, str(exc)))
            continue
        loaded.append(
            LoadedPluginToolCallReview(
                spec.plugin_name, cast("PluginToolCallReviewFunction", handler)
            )
        )
    return tuple(loaded)


def _deny_all(pending: Sequence[PluginToolCall], message: str) -> dict[str, str]:
    return {call.id: message for call in pending}


async def review_plugin_tool_calls(
    reviewers: Sequence[LoadedPluginToolCallReview],
    *,
    agent_name: str,
    model: str | None,
    instruction: str,
    tools: tuple[Tool, ...],
    history: tuple[PromptMessageExtended, ...],
    request: PromptMessageExtended,
    tool_calls: Sequence[PluginToolCall],
    config: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    """Run reviewers in order; return denied call ids mapped to their error text."""
    denials: dict[str, str] = {}
    for reviewer in reviewers:
        pending = tuple(call for call in tool_calls if call.id not in denials)
        if not pending:
            break
        name = reviewer.plugin_name
        if reviewer.handler is None:
            denials.update(
                _deny_all(
                    pending,
                    f"Tool call blocked: enabled plugin '{name}' failed to load, so it cannot "
                    f"review tool calls: {reviewer.load_error}",
                )
            )
            continue
        ctx = PluginToolCallReviewContext(
            plugin_name=name,
            agent_name=agent_name,
            model=model,
            instruction=instruction,
            tools=tools,
            history=history,
            request=request,
            tool_calls=pending,
            config=config.get(name, {}),
        )
        try:
            result = reviewer.handler(ctx)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            logger.error("Tool-call review plugin failed", plugin_name=name, error=str(exc))
            denials.update(
                _deny_all(pending, f"Tool call blocked: review plugin '{name}' failed: {exc}")
            )
            continue
        if result is None:
            continue
        if not isinstance(result, Mapping):
            denials.update(
                _deny_all(
                    pending,
                    f"Tool call blocked: review plugin '{name}' returned "
                    f"{type(result).__name__}; expected a mapping of call ids to messages, or None",
                )
            )
            continue
        pending_ids = {call.id for call in pending}
        if unknown_ids := [call_id for call_id in result if call_id not in pending_ids]:
            logger.warning(
                "Tool-call review plugin denied unknown call ids",
                plugin_name=name,
                call_ids=unknown_ids,
            )
        denials.update(
            {
                str(call_id): str(message)
                for call_id, message in result.items()
                if call_id in pending_ids
            }
        )
    return denials
