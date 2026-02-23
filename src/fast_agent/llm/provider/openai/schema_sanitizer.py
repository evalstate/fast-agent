from typing import Any

_STRUCTURAL_SCHEMA_KEYS = frozenset(
    {
        "$ref",
        "allOf",
        "anyOf",
        "const",
        "contains",
        "enum",
        "if",
        "items",
        "not",
        "oneOf",
        "prefixItems",
        "properties",
        "then",
        "type",
    }
)

_STRICT_DEFAULT_MODELS = frozenset({"kimi25", "kimi-2.5"})


def _infer_json_schema_type(value: Any) -> str | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return None


def _sanitize_schema_node(node: Any) -> Any:
    if isinstance(node, list):
        return [_sanitize_schema_node(item) for item in node]

    if not isinstance(node, dict):
        return node

    default_value = node.get("default")
    sanitized: dict[str, Any] = {}

    for key, value in node.items():
        if key == "default":
            continue
        sanitized[key] = _sanitize_schema_node(value)

    if default_value is None or any(key in sanitized for key in _STRUCTURAL_SCHEMA_KEYS):
        return sanitized

    inferred_type = _infer_json_schema_type(default_value)
    if inferred_type is not None:
        sanitized["type"] = inferred_type

    return sanitized


def sanitize_tool_input_schema(input_schema: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_schema_node(input_schema)
    if isinstance(sanitized, dict):
        return sanitized
    return {"type": "object", "properties": {}}


def should_strip_tool_schema_defaults(model_name: str | None) -> bool:
    if not model_name:
        return False

    normalized = model_name.strip().lower()
    return normalized in _STRICT_DEFAULT_MODELS or "kimi-k2.5" in normalized
