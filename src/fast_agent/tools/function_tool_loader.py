"""
Dynamic function tool loader.

Loads Python functions from files for use as native FastMCP tools.
Supports both direct callables and string specs like "module.py:function_name".
"""

import importlib.util
import inspect
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from fastmcp.tools import FunctionTool, ToolResult

from fast_agent.agents.agent_types import ScopedFunctionToolConfig
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


def _as_default_tool_result(raw: Any) -> ToolResult:
    if isinstance(raw, ToolResult):
        return raw
    if raw is None:
        return ToolResult(content=[])
    return ToolResult(content=raw)


def _wrap_default_tool_result(fn: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def async_wrapped(*args: Any, **kwargs: Any) -> ToolResult:
            raw = await fn(*args, **kwargs)
            return _as_default_tool_result(raw)

        async_wrapped.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return async_wrapped

    @wraps(fn)
    def sync_wrapped(*args: Any, **kwargs: Any) -> ToolResult | Any:
        raw = fn(*args, **kwargs)
        if inspect.isawaitable(raw):

            async def await_and_wrap() -> ToolResult:
                awaited = await raw
                return _as_default_tool_result(awaited)

            return await_and_wrap()
        return _as_default_tool_result(raw)

    sync_wrapped.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    return sync_wrapped


def build_default_function_tool(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> FunctionTool:
    """
    Build a FastMCP FunctionTool with fast-agent's text-only-by-default policy.

    Plain callable return values are wrapped as ``ToolResult(content=...)`` so FastMCP
    preserves normal content rendering while suppressing implicit structured output.
    Explicit ``ToolResult`` returns pass through unchanged.
    """
    return FunctionTool.from_function(
        _wrap_default_tool_result(fn),
        name=name,
        description=description,
        output_schema=None,
    )


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

    module_name = f"_function_tool_{module_path.stem}_{id(spec)}"
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
    tools_config: list[Callable[..., Any] | str | ScopedFunctionToolConfig] | None,
    base_path: Path | None = None,
) -> list[FunctionTool]:
    """
    Load function tools from a config list.

    Args:
        tools_config: List of either:
            - Callable functions (used directly)
            - String specs like "module.py:function_name" (loaded dynamically)
        base_path: Base path for resolving relative module paths in string specs.

    Returns:
        List of native FunctionTool objects ready for use with an agent.
    """
    if not tools_config:
        return []

    result: list[FunctionTool] = []
    for tool_spec in tools_config:
        try:
            if isinstance(tool_spec, ScopedFunctionToolConfig):
                result.append(
                    build_default_function_tool(
                        tool_spec.function,
                        name=tool_spec.name,
                        description=tool_spec.description,
                    )
                )
            elif callable(tool_spec):
                tool_name = getattr(tool_spec, "_fast_tool_name", None)
                tool_desc = getattr(tool_spec, "_fast_tool_description", None)
                result.append(
                    build_default_function_tool(tool_spec, name=tool_name, description=tool_desc)
                )
            elif isinstance(tool_spec, str):
                result.append(
                    build_default_function_tool(load_function_from_spec(tool_spec, base_path))
                )
            else:
                logger.warning(f"Skipping invalid function tool config: {tool_spec}")
        except Exception as exc:
            logger.error(f"Failed to load function tool '{tool_spec}': {exc}")
            raise

    return result
