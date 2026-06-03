"""
Dynamic hook loader for tool runner hooks.

Loads Python hook functions from files for use with ToolRunnerHooks.
Supports string specs like "module.py:function_name" mapped to hook types.
"""

from __future__ import annotations

import inspect
from dataclasses import fields
from typing import TYPE_CHECKING, Protocol, cast

from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.hooks.hook_context import HookContext
from fast_agent.hooks.hook_messages import show_hook_failure
from fast_agent.tools.python_file_loader import (
    PythonCallableLoadMessages,
    load_callable_from_file_spec,
    parse_callable_file_spec,
)
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from fast_agent.agents.tool_runner import ToolRunner
    from fast_agent.hooks.hook_context import HookAgentProtocol

# Valid hook types that can be specified in tool_hooks config.
VALID_HOOK_TYPES = frozenset(field.name for field in fields(ToolRunnerHooks))

# Type alias for hook functions that accept HookContext
class HookFunction(Protocol):
    __name__: str

    def __call__(self, ctx: HookContext) -> Awaitable[None]: ...


def load_hook_function(spec: str, base_path: Path | None = None) -> HookFunction:
    """
    Load a Python hook function from a spec string.

    Args:
        spec: A string in the format "module.py:function_name" or "path/to/module.py:function_name"
        base_path: Optional base path for resolving relative module paths.
                   If None, uses current working directory.

    Returns:
        The loaded async hook function that accepts HookContext.

    Raises:
        AgentConfigError: If the spec format is invalid or the function cannot be loaded.
    """
    func = load_callable_from_file_spec(
        spec,
        base_path=base_path,
        module_name_prefix="_hook_module",
        messages=PythonCallableLoadMessages(
            invalid_spec="Invalid hook spec '{spec}'. Expected format: 'module.py:function_name'",
            module_not_found="Hook module file not found for '{spec}'",
            module_spec_failed="Failed to create module spec for hook '{spec}'",
            import_failed="Failed to import hook module for '{spec}'",
            callable_not_found="Hook function '{func_name}' not found for '{spec}'",
            not_callable="Hook '{func_name}' is not callable for '{spec}'",
        ),
    )
    if not inspect.iscoroutinefunction(func):
        parsed_spec = parse_callable_file_spec(
            spec,
            invalid_message="Invalid hook spec '{spec}'. Expected format: 'module.py:function_name'",
        )
        func_name = getattr(func, "__name__", parsed_spec.callable_name)
        raise AgentConfigError(f"Hook function '{func_name}' must be async")
    return cast("HookFunction", func)


def _create_hook_wrapper(
    hook_func: HookFunction,
    hook_type: str,
    *,
    hook_name: str | None = None,
    hook_spec: str | None = None,
) -> Callable[..., Awaitable[None]]:
    """
    Create a wrapper that converts ToolRunner hook signatures to HookContext.

    The ToolRunnerHooks expect different signatures:
    - before_llm_call: (runner, messages) -> None
    - after_llm_call, before_tool_call, after_tool_call, after_turn_complete: (runner, message) -> None

    This wrapper creates a HookContext and calls the user's hook function.

    Note: We use runner._agent to ensure
    hooks work correctly when copied to cloned agents (e.g., via spawn_detached_instance).
    """

    def hook_context(runner: ToolRunner, message: PromptMessageExtended) -> HookContext:
        return HookContext(
            runner=runner,
            agent=cast("HookAgentProtocol", runner._agent),
            message=message,
            hook_type=hook_type,
        )

    async def invoke_hook(ctx: HookContext) -> None:
        try:
            await hook_func(ctx)
        except Exception as exc:
            show_hook_failure(
                ctx,
                hook_name=hook_name or hook_func.__name__,
                hook_kind="tool",
                error=exc,
            )
            logger.exception(
                "Tool hook failed",
                hook_type=hook_type,
                hook_name=hook_name,
                hook_spec=hook_spec,
            )
            raise

    if hook_type == "before_llm_call":

        async def before_llm_wrapper(
            runner: ToolRunner, messages: list[PromptMessageExtended]
        ) -> None:
            # For before_llm_call, we use the last message if available
            message = messages[-1] if messages else PromptMessageExtended(role="user", content=[])
            await invoke_hook(hook_context(runner, message))

        return before_llm_wrapper

    # after_llm_call, before_tool_call, after_tool_call, after_turn_complete
    async def message_wrapper(runner: ToolRunner, message: PromptMessageExtended) -> None:
        await invoke_hook(hook_context(runner, message))

    return message_wrapper


def load_tool_runner_hooks(
    hooks_config: dict[str, str] | None,
    base_path: Path | None = None,
) -> ToolRunnerHooks | None:
    """
    Load hook functions from a dict config and create a ToolRunnerHooks instance.

    Args:
        hooks_config: Dict mapping hook types to string specs, e.g.:
            {
                "before_llm_call": "hooks.py:log_messages",
                "after_turn_complete": "hooks.py:trim_history"
            }
        base_path: Base path for resolving relative module paths in string specs.

    Returns:
        A ToolRunnerHooks instance with the loaded hooks, or None if no hooks.

    Raises:
        AgentConfigError: If an invalid hook type is specified or loading fails.
    """
    if not hooks_config:
        return None

    # Validate hook types
    invalid_types = set(hooks_config.keys()) - VALID_HOOK_TYPES
    if invalid_types:
        raise AgentConfigError(
            f"Invalid hook types: {invalid_types}",
            f"Valid types are: {sorted(VALID_HOOK_TYPES)}",
        )

    # Load each hook function and create wrappers
    hooks_kwargs: dict[str, Callable[..., Awaitable[None]]] = {}

    for hook_type, spec in hooks_config.items():
        hook_func = load_hook_function(spec, base_path)
        hook_name = parse_callable_file_spec(
            spec,
            invalid_message="Invalid hook spec '{spec}'. Expected format: 'module.py:function_name'",
        ).callable_name
        wrapper = _create_hook_wrapper(
            hook_func,
            hook_type,
            hook_name=hook_name,
            hook_spec=spec,
        )
        hooks_kwargs[hook_type] = wrapper

    if not hooks_kwargs:
        return None

    return ToolRunnerHooks(**hooks_kwargs)
# Logger for hook execution errors
logger = get_logger(__name__)
