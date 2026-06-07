from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Literal


def _load_generate_reference_docs() -> Any:
    path = Path(__file__).resolve().parents[2] / "docs" / "generate_reference_docs.py"
    spec = importlib.util.spec_from_file_location("generate_reference_docs", path)
    assert spec is not None
    loader = spec.loader
    assert loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_reference_docs"] = module
    loader.exec_module(module)
    return module


_docs = _load_generate_reference_docs()


def test_format_type_uses_pipe_none_for_optional_annotations() -> None:
    assert _docs._format_type(str | None) == "str | None"
    assert _docs._format_type(dict[str, Any] | None) == "dict[str, Any] | None"


def test_format_type_quotes_literal_string_values() -> None:
    assert _docs._format_type(Literal["auto", "off"] | None) == "Literal['auto', 'off'] | None"


def test_normalize_signature_text_hides_pathlib_internal_module() -> None:
    signature = "(instruction: str | pathlib._local.Path | None = None)"

    assert _docs._normalize_signature_text(signature) == (
        "(instruction: str | pathlib.Path | None = None)"
    )
