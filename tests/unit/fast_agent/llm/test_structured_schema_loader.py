from pathlib import Path

import pytest

from fast_agent.llm.structured_schema import (
    load_pydantic_model,
    resolve_local_ref,
    sanitize_structured_output_schema,
)


def _write_schema_module(tmp_path: Path) -> None:
    (tmp_path / "schema_models.py").write_text(
        "from pydantic import BaseModel\nclass Answer(BaseModel):\n    value: str\n",
        encoding="utf-8",
    )


def test_load_pydantic_model_normalizes_module_and_class_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_schema_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    model = load_pydantic_model("  schema_models  :  Answer  ")

    assert model.__name__ == "Answer"


def test_load_pydantic_model_rejects_empty_class_path_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_schema_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="Could not resolve schema model"):
        load_pydantic_model("schema_models:Answer..Nested")


def test_resolve_local_ref_handles_escaped_keys_and_missing_paths() -> None:
    root = {"$defs": {"path/with~tilde": {"type": "string"}, "nullable": None}}

    assert resolve_local_ref(root, "#/$defs/path~1with~0tilde") == {"type": "string"}
    assert resolve_local_ref(root, "#/$defs/nullable") is None
    assert resolve_local_ref(root, "#/$defs/missing") is None


def test_sanitize_structured_output_schema_resolves_plain_property_refs() -> None:
    schema = {
        "$defs": {
            "Child": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "default": None,
                    }
                },
            }
        },
        "type": "object",
        "properties": {
            "child": {
                "$ref": "#/$defs/Child",
            }
        },
    }

    sanitized = sanitize_structured_output_schema(
        schema,
        require_all_properties=True,
        additional_properties_false=True,
    )
    child_schema = sanitized["properties"]["child"]

    assert "$ref" not in child_schema
    assert child_schema["required"] == ["name"]
    assert child_schema["additionalProperties"] is False
    assert "default" not in child_schema["properties"]["name"]
