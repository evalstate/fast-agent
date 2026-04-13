from __future__ import annotations

from typing import Any

from jsonschema.validators import validator_for


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
