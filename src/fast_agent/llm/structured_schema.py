from __future__ import annotations

from copy import deepcopy
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


def sanitize_structured_output_schema(
    schema: dict[str, Any],
    *,
    require_all_properties: bool = False,
    additional_properties_false: bool = False,
    strip_none_defaults: bool = True,
) -> dict[str, Any]:
    """Return a provider-ready copy of a JSON Schema for structured outputs."""
    copied = deepcopy(schema)
    return _sanitize_structured_output_schema_node(
        copied,
        copied,
        require_all_properties=require_all_properties,
        additional_properties_false=additional_properties_false,
        strip_none_defaults=strip_none_defaults,
    )


def _sanitize_structured_output_schema_node(
    node: Any,
    root: dict[str, Any],
    *,
    require_all_properties: bool,
    additional_properties_false: bool,
    strip_none_defaults: bool,
) -> Any:
    if isinstance(node, list):
        return [
            _sanitize_structured_output_schema_node(
                item,
                root,
                require_all_properties=require_all_properties,
                additional_properties_false=additional_properties_false,
                strip_none_defaults=strip_none_defaults,
            )
            for item in node
        ]

    if not isinstance(node, dict):
        return node

    for defs_key in ("$defs", "definitions"):
        defs = node.get(defs_key)
        if isinstance(defs, dict):
            node[defs_key] = {
                key: _sanitize_structured_output_schema_node(
                    value,
                    root,
                    require_all_properties=require_all_properties,
                    additional_properties_false=additional_properties_false,
                    strip_none_defaults=strip_none_defaults,
                )
                for key, value in defs.items()
            }

    properties = node.get("properties")
    if isinstance(properties, dict):
        if require_all_properties:
            node["required"] = list(properties.keys())
        node["properties"] = {
            key: _sanitize_structured_output_schema_node(
                value,
                root,
                require_all_properties=require_all_properties,
                additional_properties_false=additional_properties_false,
                strip_none_defaults=strip_none_defaults,
            )
            for key, value in properties.items()
        }

    if (
        additional_properties_false
        and (node.get("type") == "object" or isinstance(properties, dict))
        and node.get("additionalProperties") is not False
    ):
        node["additionalProperties"] = False

    items = node.get("items")
    if isinstance(items, dict):
        node["items"] = _sanitize_structured_output_schema_node(
            items,
            root,
            require_all_properties=require_all_properties,
            additional_properties_false=additional_properties_false,
            strip_none_defaults=strip_none_defaults,
        )

    for union_key in ("anyOf", "oneOf"):
        union = node.get(union_key)
        if isinstance(union, list):
            node[union_key] = [
                _sanitize_structured_output_schema_node(
                    item,
                    root,
                    require_all_properties=require_all_properties,
                    additional_properties_false=additional_properties_false,
                    strip_none_defaults=strip_none_defaults,
                )
                for item in union
            ]

    all_of = node.get("allOf")
    if isinstance(all_of, list):
        if len(all_of) == 1 and isinstance(all_of[0], dict):
            merged = _sanitize_structured_output_schema_node(
                all_of[0],
                root,
                require_all_properties=require_all_properties,
                additional_properties_false=additional_properties_false,
                strip_none_defaults=strip_none_defaults,
            )
            node.update(merged)
            node.pop("allOf", None)
        else:
            node["allOf"] = [
                _sanitize_structured_output_schema_node(
                    item,
                    root,
                    require_all_properties=require_all_properties,
                    additional_properties_false=additional_properties_false,
                    strip_none_defaults=strip_none_defaults,
                )
                for item in all_of
            ]

    if strip_none_defaults and node.get("default") is None:
        node.pop("default", None)

    ref = node.get("$ref")
    if isinstance(ref, str) and len(node) > 1:
        resolved = _resolve_local_ref(root, ref)
        if isinstance(resolved, dict):
            node.update({**resolved, **node})
            node.pop("$ref", None)
            return _sanitize_structured_output_schema_node(
                node,
                root,
                require_all_properties=require_all_properties,
                additional_properties_false=additional_properties_false,
                strip_none_defaults=strip_none_defaults,
            )

    return node


def _resolve_local_ref(root: dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        return None

    target: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(target, dict):
            return None
        target = target.get(part.replace("~1", "/").replace("~0", "~"))
    return target
