"""AgentCard function tool parsing and validation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.tools.function_tool_config import (
    FunctionToolSpec,
    function_tool_entrypoint,
    parse_function_tool_card_entry,
)
from fast_agent.tools.python_file_loader import parse_callable_file_spec
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Iterable


def check_function_tool_spec(spec: str, base_path: Path) -> str | None:
    parsed = _parse_function_tool_spec(spec, base_path)
    if isinstance(parsed, str):
        return parsed

    module_path, func_name = parsed
    tree = _parse_function_tool_module(module_path)
    if isinstance(tree, str):
        return tree

    if _module_defines_function(tree, func_name):
        return None
    return f"Function '{func_name}' not found in {module_path.name}"


def _parse_function_tool_spec(spec: str, base_path: Path) -> tuple[Path, str] | str:
    try:
        parsed = parse_callable_file_spec(
            spec,
            invalid_message="Invalid function tool spec '{spec}'",
        )
    except AgentConfigError as exc:
        return str(exc)

    module_path = Path(parsed.module_path_text)
    if not module_path.is_absolute():
        module_path = (base_path / module_path).resolve()

    if not module_path.exists():
        return f"Function tool module file not found ({module_path})"
    if strip_casefold(module_path.suffix) != ".py":
        return f"Function tool module must be a .py file ({module_path})"
    return module_path, parsed.callable_name


def _parse_function_tool_module(module_path: Path) -> ast.Module | str:
    try:
        module_text = module_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return f"Failed to read tool module ({module_path}): {exc}"
    try:
        return ast.parse(module_text)
    except SyntaxError as exc:
        return f"Failed to parse tool module ({module_path}): {exc}"


def _module_defines_function(tree: ast.Module, func_name: str) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name
        for node in tree.body
    )


def iter_function_tool_specs(tool_specs: Iterable[object]) -> Iterable[str]:
    for spec in tool_specs:
        entrypoint = function_tool_entrypoint(spec)
        if entrypoint:
            yield entrypoint


def ensure_function_tool_list(
    raw_value: object,
    field_name: str,
    errors: list[str],
) -> list[str | FunctionToolSpec]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        try:
            return [
                parse_function_tool_card_entry(
                    raw_value,
                    field_path=field_name,
                )
            ]
        except ValueError as exc:
            errors.append(str(exc))
            return []
    if not isinstance(raw_value, list):
        errors.append(f"'{field_name}' must be a string or list")
        return []

    values: list[str | FunctionToolSpec] = []
    for index, entry in enumerate(raw_value):
        try:
            values.append(
                parse_function_tool_card_entry(
                    entry,
                    field_path=f"{field_name}[{index}]",
                )
            )
        except ValueError as exc:
            errors.append(str(exc))
    return values
