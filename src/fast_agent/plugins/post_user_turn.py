"""Post-user-turn display plugin contract and runtime."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.plugins.models import (
    PluginPostUserTurnContext,
    PluginPostUserTurnFunction,
    PluginPostUserTurnOutput,
)
from fast_agent.tools.python_file_loader import (
    PythonCallableLoadMessages,
    load_callable_from_file_spec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from fast_agent.llm.usage_tracking import TurnUsage
    from fast_agent.plugins.models import PluginPostUserTurnSpec

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedPluginPostUserTurn:
    plugin_name: str
    handler: PluginPostUserTurnFunction


def load_plugin_post_user_turn_function(spec: str) -> PluginPostUserTurnFunction:
    function = load_callable_from_file_spec(
        spec,
        module_name_prefix="_plugin_post_user_turn",
        messages=PythonCallableLoadMessages(
            invalid_spec="Invalid post-user-turn handler '{spec}'. Expected 'module.py:function'",
            module_not_found="Post-user-turn module file not found for '{spec}'",
            module_spec_failed="Failed to create module spec for post-user-turn handler '{spec}'",
            import_failed="Failed to import post-user-turn module for '{spec}'",
            callable_not_found="Post-user-turn function '{func_name}' not found for '{spec}'",
            not_callable="Post-user-turn target '{func_name}' is not callable for '{spec}'",
        ),
        register_module=True,
    )
    return cast("PluginPostUserTurnFunction", function)


def load_plugin_post_user_turn_handlers(
    specs: Sequence[PluginPostUserTurnSpec],
) -> list[LoadedPluginPostUserTurn]:
    loaded: list[LoadedPluginPostUserTurn] = []
    for spec in specs:
        try:
            handler = load_plugin_post_user_turn_function(spec.handler)
        except Exception as exc:
            logger.warning(
                "Failed to load post-user-turn plugin",
                plugin_name=spec.plugin_name,
                handler=spec.handler,
                error=str(exc),
            )
            continue
        loaded.append(LoadedPluginPostUserTurn(plugin_name=spec.plugin_name, handler=handler))
    return loaded


async def run_plugin_post_user_turn(
    handlers: Sequence[LoadedPluginPostUserTurn],
    *,
    agent_name: str,
    turn_usage: tuple[TurnUsage, ...],
    session_usage: tuple[TurnUsage, ...],
    config: Mapping[str, Mapping[str, object]],
    display: Callable[[str], None],
    report_session_usage: Callable[[str], None] | None = None,
) -> None:
    for loaded in handlers:
        ctx = PluginPostUserTurnContext(
            plugin_name=loaded.plugin_name,
            agent_name=agent_name,
            turn_usage=turn_usage,
            session_usage=session_usage,
            config=config.get(loaded.plugin_name, {}),
        )
        try:
            result = loaded.handler(ctx)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, PluginPostUserTurnOutput):
                if result.display:
                    display(result.display)
                if result.session_usage and report_session_usage is not None:
                    report_session_usage(result.session_usage)
            elif result is not None and not isinstance(result, str):
                raise AgentConfigError(
                    f"Post-user-turn plugin '{loaded.plugin_name}' returned "
                    f"{type(result).__name__}; expected str, "
                    "PluginPostUserTurnOutput, or None"
                )
            elif result:
                display(result)
        except Exception as exc:
            logger.warning(
                "Post-user-turn plugin failed",
                plugin_name=loaded.plugin_name,
                error=str(exc),
            )
