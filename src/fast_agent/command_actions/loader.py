"""Dynamic loader for plugin command action handlers."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fast_agent.core.exceptions import AgentConfigError

if TYPE_CHECKING:
    from fast_agent.command_actions.models import PluginCommandActionFunction


def load_plugin_command_action_function(
    spec: str,
    base_path: Path | None = None,
) -> PluginCommandActionFunction:
    """Load an async command action function from ``path.py:function``."""
    if ":" not in spec:
        raise AgentConfigError(
            f"Invalid command action handler '{spec}'. Expected format: 'module.py:function_name'"
        )

    module_path_str, func_name = spec.rsplit(":", 1)
    module_path = Path(module_path_str)
    if not module_path.is_absolute():
        module_path = ((base_path or Path.cwd()) / module_path).resolve()

    if not module_path.exists():
        raise AgentConfigError(
            f"Command action module file not found for '{spec}'",
            f"Resolved path: {module_path}",
        )

    module_name = f"_plugin_command_action_{module_path.stem}_{id(spec)}"
    import_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if import_spec is None or import_spec.loader is None:
        raise AgentConfigError(
            f"Failed to create module spec for command action '{spec}'",
            f"Resolved path: {module_path}",
        )

    module = importlib.util.module_from_spec(import_spec)
    sys.modules[module_name] = module
    try:
        import_spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        sys.modules.pop(module_name, None)
        raise AgentConfigError(
            f"Failed to import command action module for '{spec}'",
            str(exc),
        ) from exc

    func = vars(module).get(func_name)
    if func is None:
        raise AgentConfigError(
            f"Command action function '{func_name}' not found in '{module_path}'"
        )
    if not callable(func):
        raise AgentConfigError(
            f"Command action target '{func_name}' in '{module_path}' is not callable"
        )
    if not inspect.iscoroutinefunction(func):
        raise AgentConfigError(
            f"Command action function '{func_name}' must be async",
            f"Resolved path: {module_path}",
        )

    return cast("PluginCommandActionFunction", func)
