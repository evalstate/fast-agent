"""
Dynamic hook loader for tool runner hooks.

Loads Python hook functions from files for use with ToolRunnerHooks.
Supports string specs like "module.py:function_name" mapped to hook types.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.hooks.hook_context import HookContext
from fast_agent.hooks.hook_messages import show_hook_failure
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fast_agent.agents.tool_runner import ToolRunner

# Valid hook types that can be specified in tool_hooks config
VALID_HOOK_TYPES = frozenset(
    {
        "before_llm_call",
        "after_llm_call",
        "before_tool_call",
        "after_tool_call",
        "after_turn_complete",
    }
)

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
    if ":" not in spec:
        raise AgentConfigError(
            f"Invalid hook spec '{spec}'. Expected format: 'module.py:function_name'"
        )

    module_path_str, func_name = spec.rsplit(":", 1)
    module_path = Path(module_path_str)

    # Resolve relative paths
    if not module_path.is_absolute():
        if base_path is not None:
            module_path = (base_path / module_path).resolve()
        else:
            module_path = Path.cwd() / module_path

    if not module_path.exists():
        raise AgentConfigError(
            f"Hook module file not found for '{spec}'",
            f"Resolved path: {module_path}",
        )

    # Generate a unique module name to avoid conflicts
    module_name = f"_hook_module_{module_path.stem}_{id(spec)}"

    # Load the module dynamically
    spec_obj = importlib.util.spec_from_file_location(module_name, module_path)
    if spec_obj is None or spec_obj.loader is None:
        raise AgentConfigError(
            f"Failed to create module spec for hook '{spec}'",
            f"Resolved path: {module_path}",
        )

    module = importlib.util.module_from_spec(spec_obj)
    try:
        spec_obj.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise AgentConfigError(
            f"Failed to import hook module for '{spec}'",
            str(exc),
        ) from exc

    # Get the function from the module
    if not hasattr(module, func_name):
        raise AgentConfigError(
            f"Hook function '{func_name}' not found for '{spec}'",
            f"Module path: {module_path}",
        )

    func = getattr(module, func_name)
    if not callable(func):
        raise AgentConfigError(
            f"Hook '{func_name}' is not callable for '{spec}'",
            f"Module path: {module_path}",
        )

    return func


def _create_hook_wrapper(
    hook_func: HookFunction,
    hook_type: str,
    agent: Any,  # noqa: ARG001 - kept for API compatibility but not used
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

    Note: We use runner._agent instead of the captured agent parameter to ensure
    hooks work correctly when copied to cloned agents (e.g., via spawn_detached_instance).
    """
    if hook_type == "before_llm_call":

        async def before_llm_wrapper(
            runner: ToolRunner, messages: list[PromptMessageExtended]
        ) -> None:
            # For before_llm_call, we use the last message if available
            message = messages[-1] if messages else PromptMessageExtended(role="user", content=[])
            ctx = HookContext(
                runner=runner,
                agent=runner._agent,  # Use runner's agent, not captured closure
                message=message,
                hook_type=hook_type,
            )
            try:
                await hook_func(ctx)
            except Exception as exc:  # noqa: BLE001
                show_hook_failure(
                    ctx,
                    hook_name=hook_name or getattr(hook_func, "__name__", None),
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

        return before_llm_wrapper
    else:
        # after_llm_call, before_tool_call, after_tool_call, after_turn_complete
        async def message_wrapper(runner: ToolRunner, message: PromptMessageExtended) -> None:
            ctx = HookContext(
                runner=runner,
                agent=runner._agent,  # Use runner's agent, not captured closure
                message=message,
                hook_type=hook_type,
            )
            try:
                await hook_func(ctx)
            except Exception as exc:  # noqa: BLE001
                show_hook_failure(
                    ctx,
                    hook_name=hook_name or getattr(hook_func, "__name__", None),
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

        return message_wrapper


def load_tool_runner_hooks(
    hooks_config: dict[str, str] | None,
    agent: Any,
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
        agent: The agent instance (for HookContext access to message_history).
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
        hook_name = spec.rsplit(":", 1)[-1]
        wrapper = _create_hook_wrapper(
            hook_func,
            hook_type,
            agent,
            hook_name=hook_name,
            hook_spec=spec,
        )
        hooks_kwargs[hook_type] = wrapper

    if not hooks_kwargs:
        return None

    return ToolRunnerHooks(**hooks_kwargs)
# Logger for hook execution errors
logger = get_logger(__name__)
