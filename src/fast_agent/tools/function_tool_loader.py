"""
Dynamic function tool loader.

Loads Python functions from files for use as agent tools.
Supports both direct callables and string specs like "module.py:function_name".
"""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.fastmcp.tools.base import Tool as FastMCPTool

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


def load_function_from_spec(spec: str, base_path: Path | None = None) -> Callable[..., Any]:
    """
    Load a Python function from a spec string.

    Args:
        spec: A string in the format "module.py:function_name" or "path/to/module.py:function_name"
        base_path: Optional base path for resolving relative module paths.
                   If None, uses current working directory.

    Returns:
        The loaded callable function.

    Raises:
        AgentConfigError: If the spec format is invalid or the tool cannot be loaded.
    """
    if ":" not in spec:
        raise AgentConfigError(
            f"Invalid function tool spec '{spec}'. Expected format: 'module.py:function_name'"
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
            f"Function tool module file not found for '{spec}'",
            f"Resolved path: {module_path}",
        )

    # Generate a unique module name to avoid conflicts
    module_name = f"_function_tool_{module_path.stem}_{id(spec)}"

    # Load the module dynamically
    spec_obj = importlib.util.spec_from_file_location(module_name, module_path)
    if spec_obj is None or spec_obj.loader is None:
        raise AgentConfigError(
            f"Failed to create module spec for '{spec}'",
            f"Resolved path: {module_path}",
        )

    module = importlib.util.module_from_spec(spec_obj)
    try:
        spec_obj.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise AgentConfigError(
            f"Failed to import function tool module for '{spec}'",
            str(exc),
        ) from exc

    # Get the function from the module
    if not hasattr(module, func_name):
        raise AgentConfigError(
            f"Function '{func_name}' not found for '{spec}'",
            f"Module path: {module_path}",
        )

    func = getattr(module, func_name)
    if not callable(func):
        raise AgentConfigError(
            f"Function '{func_name}' is not callable for '{spec}'",
            f"Module path: {module_path}",
        )

    return func


def load_function_tools(
    tools_config: list[Callable[..., Any] | str] | None,
    base_path: Path | None = None,
) -> list[FastMCPTool]:
    """
    Load function tools from a config list.

    Args:
        tools_config: List of either:
            - Callable functions (used directly)
            - String specs like "module.py:function_name" (loaded dynamically)
        base_path: Base path for resolving relative module paths in string specs.

    Returns:
        List of FastMCPTool objects ready for use with an agent.
    """
    if not tools_config:
        return []

    result: list[FastMCPTool] = []

    for tool_spec in tools_config:
        try:
            if callable(tool_spec):
                # Direct callable - wrap it
                result.append(FastMCPTool.from_function(tool_spec))
            elif isinstance(tool_spec, str):
                # String spec - load and wrap
                func = load_function_from_spec(tool_spec, base_path)
                result.append(FastMCPTool.from_function(func))
            else:
                logger.warning(f"Skipping invalid function tool config: {tool_spec}")
        except Exception as e:
            logger.error(f"Failed to load function tool '{tool_spec}': {e}")
            raise

    return result
