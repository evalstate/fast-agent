"""
Dynamic function tool loader.

Loads Python functions from files for use as native FastMCP tools.
Supports both direct callables and string specs like "module.py:function_name".
"""

import inspect
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast, runtime_checkable

from fastmcp.tools import FunctionTool, ToolResult

from fast_agent.agents.agent_types import ScopedFunctionToolConfig
from fast_agent.core.logging.logger import get_logger
from fast_agent.tools.function_tool_config import FunctionToolSpec
from fast_agent.tools.python_file_loader import (
    PythonCallableLoadMessages,
    load_callable_from_file_spec,
)

logger = get_logger(__name__)


@runtime_checkable
class _FastToolMetadataCallable(Protocol):
    _fast_tool_name: str | None
    _fast_tool_description: str | None

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


FunctionToolConfig: TypeAlias = (
    Callable[..., Any] | str | ScopedFunctionToolConfig | FunctionToolSpec
)


def _set_signature(wrapper: Callable[..., Any], source: Callable[..., Any]) -> None:
    signature_wrapper = cast("Any", wrapper)
    signature_wrapper.__signature__ = inspect.signature(source)


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

        _set_signature(async_wrapped, fn)
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

    _set_signature(sync_wrapped, fn)
    return sync_wrapped


def build_default_function_tool(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> FunctionTool:
    """
    Build a FastMCP FunctionTool with fast-agent's text-only-by-default policy.

    Plain callable return values are wrapped as ``ToolResult(content=...)`` so FastMCP
    preserves normal content rendering while suppressing implicit structured output.
    Explicit ``ToolResult`` returns pass through unchanged.
    """
    tool = FunctionTool.from_function(
        _wrap_default_tool_result(fn),
        name=name,
        description=description,
        output_schema=None,
    )
    if metadata:
        current_meta = dict(tool.meta or {})
        current_meta.update(metadata)
        tool.meta = current_meta
    return tool


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
    return load_callable_from_file_spec(
        spec,
        base_path=base_path,
        module_name_prefix="_function_tool",
        messages=PythonCallableLoadMessages(
            invalid_spec=(
                "Invalid function tool spec '{spec}'. Expected format: 'module.py:function_name'"
            ),
            module_not_found="Function tool module file not found for '{spec}'",
            module_spec_failed="Failed to create module spec for '{spec}'",
            import_failed="Failed to import function tool module for '{spec}'",
            callable_not_found="Function '{func_name}' not found for '{spec}'",
            not_callable="Function '{func_name}' is not callable for '{spec}'",
        ),
    )


def load_function_tools(
    tools_config: list[FunctionToolConfig] | None,
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
            tool = _function_tool_from_config(tool_spec, base_path)
            if tool is None:
                logger.warning(f"Skipping invalid function tool config: {tool_spec}")
                continue
            result.append(tool)
        except Exception as exc:
            logger.error(f"Failed to load function tool '{tool_spec}': {exc}")
            raise

    return result


def _function_tool_from_config(
    tool_spec: FunctionToolConfig,
    base_path: Path | None,
) -> FunctionTool | None:
    if isinstance(tool_spec, ScopedFunctionToolConfig):
        return build_default_function_tool(
            tool_spec.function,
            name=tool_spec.name,
            description=tool_spec.description,
        )

    if isinstance(tool_spec, str):
        return build_default_function_tool(load_function_from_spec(tool_spec, base_path))

    if isinstance(tool_spec, FunctionToolSpec):
        return build_default_function_tool(
            load_function_from_spec(tool_spec.entrypoint, base_path),
            metadata=tool_spec.metadata(),
        )

    if isinstance(tool_spec, _FastToolMetadataCallable):
        return build_default_function_tool(
            tool_spec,
            name=tool_spec._fast_tool_name,
            description=tool_spec._fast_tool_description,
        )

    if callable(tool_spec):
        return build_default_function_tool(tool_spec)

    return None
