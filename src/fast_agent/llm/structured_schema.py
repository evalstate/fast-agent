from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import BaseModel

from fast_agent.io.source_resolver import read_text_source

if TYPE_CHECKING:
    from pathlib import Path

PydanticModel = type[BaseModel]
StructuredSchemaSource = dict[str, Any] | PydanticModel


@dataclass(frozen=True, slots=True)
class _SchemaSanitizeOptions:
    require_all_properties: bool
    additional_properties_false: bool
    strip_none_defaults: bool


def validate_json_schema_definition(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise TypeError("Structured schema must be a JSON object")
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    return schema


def validate_json_instance(instance: Any, schema: dict[str, Any]) -> None:
    validator_class = validator_for(schema)
    validator = validator_class(schema)
    validator.validate(instance)


def load_json_schema_file(path: str | Path) -> dict[str, Any]:
    try:
        raw_text = read_text_source(path, label="JSON schema file")
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    try:
        loaded = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON schema file {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(f"JSON schema file {path} must contain a JSON object")

    try:
        return validate_json_schema_definition(loaded)
    except SchemaError as exc:
        raise ValueError(f"Invalid JSON schema in {path}: {exc.message}") from exc


def load_pydantic_model(spec: str) -> PydanticModel:
    module_name, separator, class_path = spec.partition(":")
    module_name = module_name.strip()
    class_path = class_path.strip()
    if not module_name or separator != ":" or not class_path:
        raise ValueError("Expected --schema-model in the form module.path:ClassName")

    try:
        target: object = import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"Could not import schema model module {module_name}: {exc}") from exc

    try:
        for part in class_path.split("."):
            if not part:
                raise AttributeError(part)
            target = getattr(target, part)
    except AttributeError as exc:
        raise ValueError(f"Could not resolve schema model {spec}: missing {part}") from exc

    if not isinstance(target, type) or not issubclass(target, BaseModel):
        raise ValueError("--schema-model must point to a pydantic BaseModel subclass")

    return target


def load_structured_schema_source(
    *,
    json_schema: str | Path | None,
    schema_model: str | None,
) -> StructuredSchemaSource:
    if json_schema is not None and schema_model is not None:
        raise ValueError("--json-schema and --schema-model cannot be used together")
    if json_schema is None and schema_model is None:
        raise ValueError("One of --json-schema or --schema-model is required")
    if schema_model is not None:
        return load_pydantic_model(schema_model)
    assert json_schema is not None
    return load_json_schema_file(json_schema)


def sanitize_structured_output_schema(
    schema: dict[str, Any],
    *,
    require_all_properties: bool = False,
    additional_properties_false: bool = False,
    strip_none_defaults: bool = True,
) -> dict[str, Any]:
    """Return a provider-ready copy of a JSON Schema for structured outputs."""
    copied = deepcopy(schema)
    options = _SchemaSanitizeOptions(
        require_all_properties=require_all_properties,
        additional_properties_false=additional_properties_false,
        strip_none_defaults=strip_none_defaults,
    )
    return _sanitize_structured_output_schema_node(copied, copied, options)


def _sanitize_structured_output_schema_node(
    node: Any,
    root: dict[str, Any],
    options: _SchemaSanitizeOptions,
) -> Any:
    if isinstance(node, list):
        return _sanitize_schema_list(node, root, options)

    if not isinstance(node, dict):
        return node

    _sanitize_schema_definitions(node, root, options)
    _sanitize_schema_properties(node, root, options)
    _sanitize_schema_items(node, root, options)
    _sanitize_union_keywords(node, root, options)
    _sanitize_all_of(node, root, options)

    if options.strip_none_defaults and node.get("default") is None:
        node.pop("default", None)

    dereferenced = _sanitize_schema_ref(node, root, options)
    return node if dereferenced is None else dereferenced


def _sanitize_schema_list(
    values: list[Any],
    root: dict[str, Any],
    options: _SchemaSanitizeOptions,
) -> list[Any]:
    return [
        _sanitize_structured_output_schema_node(item, root, options)
        for item in values
    ]


def _sanitize_schema_mapping(
    values: dict[str, Any],
    root: dict[str, Any],
    options: _SchemaSanitizeOptions,
) -> dict[str, Any]:
    return {
        key: _sanitize_structured_output_schema_node(value, root, options)
        for key, value in values.items()
    }


def _sanitize_schema_definitions(
    node: dict[str, Any],
    root: dict[str, Any],
    options: _SchemaSanitizeOptions,
) -> None:
    for defs_key in ("$defs", "definitions"):
        defs = node.get(defs_key)
        if isinstance(defs, dict):
            node[defs_key] = _sanitize_schema_mapping(defs, root, options)


def _sanitize_schema_properties(
    node: dict[str, Any],
    root: dict[str, Any],
    options: _SchemaSanitizeOptions,
) -> None:
    properties = node.get("properties")
    if isinstance(properties, dict):
        if options.require_all_properties:
            node["required"] = list(properties.keys())
        node["properties"] = _sanitize_schema_mapping(properties, root, options)

    if (
        options.additional_properties_false
        and (node.get("type") == "object" or isinstance(properties, dict))
        and "additionalProperties" not in node
    ):
        node["additionalProperties"] = False


def _sanitize_schema_items(
    node: dict[str, Any],
    root: dict[str, Any],
    options: _SchemaSanitizeOptions,
) -> None:
    items = node.get("items")
    if isinstance(items, dict):
        node["items"] = _sanitize_structured_output_schema_node(items, root, options)


def _sanitize_union_keywords(
    node: dict[str, Any],
    root: dict[str, Any],
    options: _SchemaSanitizeOptions,
) -> None:
    for union_key in ("anyOf", "oneOf"):
        union = node.get(union_key)
        if isinstance(union, list):
            node[union_key] = _sanitize_schema_list(union, root, options)


def _sanitize_all_of(
    node: dict[str, Any],
    root: dict[str, Any],
    options: _SchemaSanitizeOptions,
) -> None:
    all_of = node.get("allOf")
    if not isinstance(all_of, list):
        return

    if len(all_of) == 1 and isinstance(all_of[0], dict):
        merged = _sanitize_structured_output_schema_node(all_of[0], root, options)
        node.update(merged)
        node.pop("allOf", None)
        return

    node["allOf"] = _sanitize_schema_list(all_of, root, options)


def _sanitize_schema_ref(
    node: dict[str, Any],
    root: dict[str, Any],
    options: _SchemaSanitizeOptions,
) -> Any | None:
    ref = node.get("$ref")
    if not isinstance(ref, str):
        return None

    resolved = resolve_local_ref(root, ref)
    if not isinstance(resolved, dict):
        return None

    if len(node) == 1:
        return _sanitize_structured_output_schema_node(deepcopy(resolved), root, options)

    node.update({**deepcopy(resolved), **node})
    node.pop("$ref", None)
    return _sanitize_structured_output_schema_node(node, root, options)


def resolve_local_ref(root: dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        return None

    target: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(target, dict):
            return None
        key = part.replace("~1", "/").replace("~0", "~")
        if key not in target:
            return None
        target = target[key]
    return target


_resolve_local_ref = resolve_local_ref
