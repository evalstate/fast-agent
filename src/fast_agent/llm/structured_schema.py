from __future__ import annotations

import json
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
