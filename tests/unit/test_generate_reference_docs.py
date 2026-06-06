from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.types import RequestParams

if TYPE_CHECKING:
    from types import ModuleType


def _load_generator() -> ModuleType:
    path = Path(__file__).parents[2] / "docs" / "generate_reference_docs.py"
    spec = importlib.util.spec_from_file_location("generate_reference_docs", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_field_default_text_renders_default_factories() -> None:
    generator = _load_generator()

    assert generator._field_default_text(RequestParams.model_fields["messages"]) == "`[]`"
    assert generator._field_default_text(RequestParams.model_fields["template_vars"]) == "`{}`"


def test_clean_signature_text_removes_noisy_module_prefixes() -> None:
    generator = _load_generator()

    cleaned = generator._clean_signature_text(
        "collections.abc.Callable[[typing.Any], collections.abc.Coroutine[typing.Any, typing.Any, str]]"
    )

    assert "collections.abc.Callable" not in cleaned
    assert "collections.abc.Coroutine" not in cleaned
    assert "typing.Any" not in cleaned
    assert cleaned == "Callable[[Any], Coroutine[Any, Any, str]]"


def test_workflow_return_annotation_wraps_common_decorator_return() -> None:
    generator = _load_generator()

    wrapped = generator._wrap_workflow_return_annotation(
        "Callable[[Callable[~P, Coroutine[Any, Any, +R]]], Callable[~P, Coroutine[Any, Any, +R]]]"
    )

    assert wrapped == (
        "Callable[\n"
        "    [Callable[~P, Coroutine[Any, Any, +R]]],\n"
        "    Callable[~P, Coroutine[Any, Any, +R]],\n"
        "]"
    )


def test_compact_signature_defaults_elides_long_string_defaults() -> None:
    generator = _load_generator()

    def func(instruction: str = "line one\nline two") -> None:
        pass

    compact = generator._compact_signature_defaults(inspect.signature(func))

    assert compact.parameters["instruction"].default == "..."


def test_current_model_table_uses_current_catalog_entries() -> None:
    generator = _load_generator()

    table = generator.generate_current_model_table("openai")

    assert "| `gpt-4.1` | `openai.gpt-4.1` | — |" in table
    assert "| `gpt-4.1-mini` | `openai.gpt-4.1-mini` | Fast |" in table
