"""Validation helpers for child-owned tool input schemas on AgentCards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fast_agent.utils.text import strip_to_none


@dataclass(frozen=True, slots=True)
class ToolInputSchemaValidation:
    """Validation result for a ``tool_input_schema`` payload."""

    normalized: dict[str, Any] | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def validate_tool_input_schema(value: Any) -> ToolInputSchemaValidation:
    """Validate an optional ``tool_input_schema`` payload.

    Returns a normalized schema when valid, plus non-fatal warning messages for
    weak-but-usable schema metadata.
    """

    if value is None:
        return ToolInputSchemaValidation(normalized=None, errors=(), warnings=())

    if not isinstance(value, Mapping):
        return ToolInputSchemaValidation(
            normalized=None,
            errors=("must be a mapping",),
            warnings=(),
        )

    schema = dict(value)
    errors: list[str] = []
    warnings: list[str] = []

    _validate_schema_type(schema, errors)
    properties = _validated_properties(schema, errors)
    required_names = _validated_required_names(schema, errors)

    required_set = set(required_names)
    _validate_required_properties_exist(required_set, properties, errors)
    _validate_property_schemas(properties, required_set, errors, warnings)
    _validate_additional_properties(schema, errors)

    normalized = schema if not errors else None
    return ToolInputSchemaValidation(
        normalized=normalized,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _validate_schema_type(schema: Mapping[str, Any], errors: list[str]) -> None:
    if schema.get("type") != "object":
        errors.append("'type' must be 'object'")


def _validated_properties(schema: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    properties_value = schema.get("properties")
    if properties_value is None:
        return {}
    if isinstance(properties_value, Mapping):
        properties: dict[str, Any] = {}
        for key, prop_schema in properties_value.items():
            property_name = _non_empty_string(key)
            if property_name is None:
                errors.append("'properties' keys must be non-empty strings")
                continue
            properties[property_name] = prop_schema
        return properties
    errors.append("'properties' must be a mapping when present")
    return {}


def _validated_required_names(schema: Mapping[str, Any], errors: list[str]) -> list[str]:
    required_value = schema.get("required")
    if required_value is None:
        return []
    if not isinstance(required_value, list):
        errors.append("'required' must be a list of property names")
        return []

    required_names: list[str] = []
    for index, entry in enumerate(required_value):
        required_name = _non_empty_string(entry)
        if required_name is None:
            errors.append(f"'required[{index}]' must be a non-empty string")
            continue
        required_names.append(required_name)
    return required_names


def _validate_required_properties_exist(
    required_set: set[str],
    properties: Mapping[str, Any],
    errors: list[str],
) -> None:
    unknown_required = sorted(name for name in required_set if name not in properties)
    if unknown_required:
        errors.append(
            "'required' references undefined properties: "
            + ", ".join(unknown_required)
        )


def _validate_property_schemas(
    properties: Mapping[str, Any],
    required_set: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    for prop_name, prop_schema in properties.items():
        _validate_property_schema(
            prop_name=prop_name,
            prop_schema=prop_schema,
            required_set=required_set,
            errors=errors,
            warnings=warnings,
        )


def _validate_property_schema(
    *,
    prop_name: object,
    prop_schema: Any,
    required_set: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    property_name = _non_empty_string(prop_name)
    if property_name is None:
        return

    if not isinstance(prop_schema, Mapping):
        errors.append(f"'properties.{property_name}' must be a mapping")
        return

    if property_name in required_set:
        _warn_missing_required_description(property_name, prop_schema, warnings)


def _warn_missing_required_description(
    prop_name: str,
    prop_schema: Mapping[str, Any],
    warnings: list[str],
) -> None:
    description = prop_schema.get("description")
    if _non_empty_string(description) is None:
        warnings.append(f"required property '{prop_name}' should include a description")


def _validate_additional_properties(schema: Mapping[str, Any], errors: list[str]) -> None:
    additional_properties = schema.get("additionalProperties")
    if additional_properties is not None and not isinstance(
        additional_properties, (bool, Mapping)
    ):
        errors.append("'additionalProperties' must be a boolean or mapping")


def _non_empty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return strip_to_none(value)
