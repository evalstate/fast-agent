"""Shared helpers for loading Python callables from file specs."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fast_agent.core.exceptions import AgentConfigError

if TYPE_CHECKING:
    from collections.abc import Callable

_MISSING = object()


@dataclass(frozen=True, slots=True)
class PythonCallableLoadMessages:
    invalid_spec: str
    module_not_found: str
    module_spec_failed: str
    import_failed: str
    callable_not_found: str
    not_callable: str


@dataclass(frozen=True, slots=True)
class PythonCallableFileSpec:
    raw: str
    module_path_text: str
    callable_name: str


def parse_callable_file_spec(
    spec: str,
    *,
    invalid_message: str,
) -> PythonCallableFileSpec:
    if ":" not in spec:
        raise AgentConfigError(invalid_message.format(spec=spec))

    module_path_text, callable_name = spec.rsplit(":", 1)
    if not module_path_text.strip() or not callable_name.strip():
        raise AgentConfigError(invalid_message.format(spec=spec))
    return PythonCallableFileSpec(
        raw=spec,
        module_path_text=module_path_text.strip(),
        callable_name=callable_name.strip(),
    )


def load_callable_from_file_spec(
    spec: str,
    *,
    base_path: Path | None = None,
    module_name_prefix: str,
    messages: PythonCallableLoadMessages,
    register_module: bool = False,
) -> Callable[..., Any]:
    parsed_spec = parse_callable_file_spec(
        spec,
        invalid_message=messages.invalid_spec,
    )
    module_path = _resolve_module_path(parsed_spec.module_path_text, base_path)

    if not module_path.exists():
        raise AgentConfigError(
            messages.module_not_found.format(spec=spec),
            f"Resolved path: {module_path}",
        )

    module_name = f"{module_name_prefix}_{module_path.stem}_{id(spec)}"
    spec_obj = importlib.util.spec_from_file_location(module_name, module_path)
    if spec_obj is None or spec_obj.loader is None:
        raise AgentConfigError(
            messages.module_spec_failed.format(spec=spec),
            f"Resolved path: {module_path}",
        )

    module = importlib.util.module_from_spec(spec_obj)
    if register_module:
        sys.modules[module_name] = module
    try:
        spec_obj.loader.exec_module(module)
    except Exception as exc:
        if register_module:
            sys.modules.pop(module_name, None)
        raise AgentConfigError(
            messages.import_failed.format(spec=spec),
            str(exc),
        ) from exc

    func = getattr(module, parsed_spec.callable_name, _MISSING)
    if func is _MISSING:
        raise AgentConfigError(
            messages.callable_not_found.format(
                func_name=parsed_spec.callable_name,
                spec=spec,
            ),
            f"Module path: {module_path}",
        )
    if not callable(func):
        raise AgentConfigError(
            messages.not_callable.format(
                func_name=parsed_spec.callable_name,
                spec=spec,
            ),
            f"Module path: {module_path}",
        )

    return func


def _resolve_module_path(module_path_str: str, base_path: Path | None) -> Path:
    module_path = Path(module_path_str)
    if module_path.is_absolute():
        return module_path
    if base_path is not None:
        return (base_path / module_path).resolve()
    return Path.cwd() / module_path
