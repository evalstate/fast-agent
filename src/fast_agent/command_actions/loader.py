"""Dynamic loader for plugin command action handlers."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, cast

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.tools.python_file_loader import (
    PythonCallableLoadMessages,
    load_callable_from_file_spec,
    parse_callable_file_spec,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.command_actions.models import (
        PluginCommandActionFunction,
        PluginCommandCompletionFunction,
    )


def load_plugin_command_action_function(
    spec: str,
    base_path: Path | None = None,
) -> PluginCommandActionFunction:
    """Load an async command action function from ``path.py:function``."""
    func = load_callable_from_file_spec(
        spec,
        base_path=base_path,
        module_name_prefix="_plugin_command_action",
        messages=PythonCallableLoadMessages(
            invalid_spec="Invalid command action handler '{spec}'. Expected format: 'module.py:function_name'",
            module_not_found="Command action module file not found for '{spec}'",
            module_spec_failed="Failed to create module spec for command action '{spec}'",
            import_failed="Failed to import command action module for '{spec}'",
            callable_not_found="Command action function '{func_name}' not found for '{spec}'",
            not_callable="Command action target '{func_name}' is not callable for '{spec}'",
        ),
        register_module=True,
    )
    if not inspect.iscoroutinefunction(func):
        parsed_spec = parse_callable_file_spec(
            spec,
            invalid_message="Invalid command action handler '{spec}'. Expected format: 'module.py:function_name'",
        )
        func_name = getattr(func, "__name__", parsed_spec.callable_name)
        raise AgentConfigError(
            f"Command action function '{func_name}' must be async",
        )

    return cast("PluginCommandActionFunction", func)


def load_plugin_command_completion_function(
    spec: str,
    base_path: Path | None = None,
) -> PluginCommandCompletionFunction:
    """Load an async command completion function from ``path.py:function``."""
    func = load_callable_from_file_spec(
        spec,
        base_path=base_path,
        module_name_prefix="_plugin_command_completion",
        messages=PythonCallableLoadMessages(
            invalid_spec="Invalid command completion handler '{spec}'. Expected format: 'module.py:function_name'",
            module_not_found="Command completion module file not found for '{spec}'",
            module_spec_failed="Failed to create module spec for command completion '{spec}'",
            import_failed="Failed to import command completion module for '{spec}'",
            callable_not_found="Command completion function '{func_name}' not found for '{spec}'",
            not_callable="Command completion target '{func_name}' is not callable for '{spec}'",
        ),
        register_module=True,
    )
    if not inspect.iscoroutinefunction(func):
        parsed_spec = parse_callable_file_spec(
            spec,
            invalid_message="Invalid command completion handler '{spec}'. Expected format: 'module.py:function_name'",
        )
        func_name = getattr(func, "__name__", parsed_spec.callable_name)
        raise AgentConfigError(
            f"Command completion function '{func_name}' must be async",
        )

    return cast("PluginCommandCompletionFunction", func)
