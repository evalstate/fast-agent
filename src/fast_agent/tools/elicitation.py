from __future__ import annotations

import json
import uuid
from typing import Any, Awaitable, Callable, Literal, Union

from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.types import Tool as McpTool
from pydantic import BaseModel, Field

from fast_agent.constants import HUMAN_INPUT_TOOL_NAME

"""
Human-input (elicitation) tool models, schemas, and builders.

This module lives in fast_agent to avoid circular imports and provides:
- A centralized HUMAN_INPUT_TOOL_NAME constant (imported from fast_agent.constants)
- Pydantic models for a simplified FormSpec (LLM-friendly)
- MCP Tool schema builder (sanitized, provider-friendly)
- FastMCP Tool builder sharing the same schema
- A lightweight async callback registry for UI-specific elicitation handling
"""

# -----------------------
# Pydantic models
# -----------------------


class OptionItem(BaseModel):
    value: Union[str, int, float, bool]
    label: str | None = None


class FormField(BaseModel):
    name: str
    type: Literal["text", "textarea", "number", "checkbox", "radio"]
    label: str | None = None
    help: str | None = None
    default: Union[str, int, float, bool] | None = None
    required: bool | None = None
    # number constraints
    min: float | None = None
    max: float | None = None
    # select options (for radio)
    options: list[OptionItem] | None = None


class HumanFormArgs(BaseModel):
    """Simplified form spec for human elicitation.

    Preferred shape for LLMs.
    """

    title: str | None = None
    description: str | None = None
    message: str | None = None
    fields: list[FormField] = Field(default_factory=list, max_length=7)


# -----------------------
# MCP tool schema builder
# -----------------------


def _resolve_schema_refs(fragment: Any, root: dict[str, Any]) -> Any:
    """Inline local schema refs of the form ``#/$defs/Name``."""
    if not isinstance(fragment, dict):
        return fragment

    if "$ref" in fragment:
        ref_path: str = fragment["$ref"]
        if ref_path.startswith("#/$defs/") and "$defs" in root:
            key = (
                ref_path.split("/#/$defs/")[-1]
                if "/#/$defs/" in ref_path
                else ref_path[len("#/$defs/") :]
            )
            target = root.get("$defs", {}).get(key)
            if isinstance(target, dict):
                return _resolve_schema_refs(target, root)
        fragment = {k: v for k, v in fragment.items() if k != "$ref"}
        fragment.setdefault("type", "object")
        fragment.setdefault("properties", {})
        return fragment

    resolved: dict[str, Any] = {}
    for key, value in fragment.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_schema_refs(value, root)
        elif isinstance(value, list):
            resolved[key] = [
                _resolve_schema_refs(item, root) if isinstance(item, (dict, list)) else item
                for item in value
            ]
        else:
            resolved[key] = value
    return resolved


def _sanitized_elicitation_schema() -> dict[str, Any]:
    schema = HumanFormArgs.model_json_schema()
    sanitized: dict[str, Any] = {"type": "object"}
    if "properties" in schema:
        props = dict(schema["properties"])
        if "form_schema" in props:
            props["schema"] = props.pop("form_schema")
        sanitized["properties"] = _resolve_schema_refs(props, schema)
    else:
        sanitized["properties"] = {}
    if "required" in schema:
        sanitized["required"] = schema["required"]
    sanitized["additionalProperties"] = True
    return sanitized


def get_elicitation_tool() -> McpTool:
    """Build the MCP Tool schema for the elicitation-backed human input tool.

    Uses Pydantic models to derive a clean, portable JSON Schema suitable for providers.
    """

    return McpTool(
        name=HUMAN_INPUT_TOOL_NAME,
        description=(
            "Collect structured input from a human via a simple form. "
            "Provide up to 7 fields with types: text, textarea, number, checkbox, or radio. "
            "Each field may include label, help, default; numbers may include min/max; radio may include options (value/label). "
            "You may also add an optional message shown above the form."
        ),
        inputSchema=_sanitized_elicitation_schema(),
    )


# -----------------------
# Elicitation input callback registry
# -----------------------

ElicitationCallback = Callable[[dict, str | None, str | None, dict | None], Awaitable[str]]

_elicitation_input_callback: ElicitationCallback | None = None


def set_elicitation_input_callback(callback: ElicitationCallback | None) -> None:
    """Register the UI/backend-specific elicitation handler.

    The callback should accept a request dict with fields: prompt, description, request_id, metadata,
    plus optional agent_name, server_name, and server_info, and return an awaited string response.
    """

    global _elicitation_input_callback
    _elicitation_input_callback = callback


def get_elicitation_input_callback() -> ElicitationCallback | None:
    return _elicitation_input_callback


# -----------------------
# Runtime: run the elicitation
# -----------------------


def _parse_schema_json_string(value: str) -> dict[str, Any] | None:
    stripped = value.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    try:
        parsed = json.loads(stripped)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_elicitation_arguments(arguments: dict | str) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        parsed = _parse_schema_json_string(arguments)
        if parsed is not None:
            return parsed
    raise ValueError("Invalid arguments. Provide FormSpec or JSON Schema object.")


def _field_string_property(field: dict[str, Any]) -> dict[str, Any]:
    return {"type": "string"}


def _field_number_property(field: dict[str, Any]) -> dict[str, Any]:
    prop: dict[str, Any] = {"type": "number"}
    minimum = field.get("min")
    maximum = field.get("max")
    if isinstance(minimum, int | float):
        prop["minimum"] = minimum
    if isinstance(maximum, int | float):
        prop["maximum"] = maximum
    return prop


def _field_radio_property(field: dict[str, Any]) -> dict[str, Any]:
    prop: dict[str, Any] = {"type": "string"}
    options = field.get("options") or []
    enum_vals: list[Any] = []
    enum_names: list[str] = []
    for option in options:
        if isinstance(option, dict) and "value" in option:
            enum_vals.append(option["value"])
            label = option.get("label")
            if isinstance(label, str):
                enum_names.append(label)
        elif option is not None:
            enum_vals.append(option)
    if enum_vals:
        prop["enum"] = enum_vals
        if enum_names and len(enum_names) == len(enum_vals):
            prop["enumNames"] = enum_names
    return prop


def _build_form_field_property(field: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    name = field.get("name")
    field_type = field.get("type")
    if not isinstance(name, str) or not isinstance(field_type, str):
        return None

    if field_type in {"text", "textarea"}:
        prop = _field_string_property(field)
    elif field_type == "number":
        prop = _field_number_property(field)
    elif field_type == "checkbox":
        prop = {"type": "boolean"}
    elif field_type == "radio":
        prop = _field_radio_property(field)
    else:
        return None

    label = field.get("label")
    help_text = field.get("help")
    default = field.get("default")
    desc_parts: list[str] = []
    if isinstance(label, str) and label:
        desc_parts.append(label)
    if isinstance(help_text, str) and help_text:
        desc_parts.append(help_text)
    if desc_parts:
        prop["description"] = " - ".join(desc_parts)
    if default is not None:
        prop["default"] = default
    return name, prop


def _build_schema_from_fields(arguments: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    fields_value = arguments.get("fields")
    assert isinstance(fields_value, list)
    if len(fields_value) > 7:
        raise ValueError(f"Error: form requests {len(fields_value)} fields; the maximum allowed is 7.")

    properties: dict[str, Any] = {}
    required_fields: list[str] = []
    for field in fields_value:
        if not isinstance(field, dict):
            continue
        field_property = _build_form_field_property(field)
        if field_property is None:
            continue
        name, prop = field_property
        properties[name] = prop
        if field.get("required") is True:
            required_fields.append(name)

    if not properties:
        raise ValueError("Invalid form specification: no valid fields provided.")

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required_fields:
        schema["required"] = required_fields

    title = arguments.get("title") if isinstance(arguments.get("title"), str) else None
    description = (
        arguments.get("description") if isinstance(arguments.get("description"), str) else None
    )
    message = arguments.get("message") if isinstance(arguments.get("message"), str) else None
    if title:
        schema["title"] = title
    if description:
        schema["description"] = description
    return schema, message


def _merge_schema_overrides(
    schema: dict[str, Any],
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    message = arguments.get("message") if isinstance(arguments.get("message"), str) else None
    if isinstance(arguments.get("title"), str) and "title" not in schema:
        schema["title"] = arguments["title"]
    if isinstance(arguments.get("description"), str) and "description" not in schema:
        schema["description"] = arguments["description"]
    if isinstance(arguments.get("required"), list) and "required" not in schema:
        schema["required"] = arguments["required"]
    if isinstance(arguments.get("properties"), dict) and "properties" not in schema:
        schema["properties"] = arguments["properties"]
    return schema, message


def _build_schema_from_schema_argument(arguments: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    schema_value = arguments.get("schema")
    if isinstance(schema_value, str):
        parsed = _parse_schema_json_string(schema_value)
        if parsed is None:
            raise ValueError("Missing or invalid schema. Provide a JSON Schema object.")
        schema_value = parsed
    if not isinstance(schema_value, dict):
        raise ValueError("Missing or invalid schema. Provide a JSON Schema object.")
    return _merge_schema_overrides(schema_value, arguments)


def _resolve_elicitation_schema(arguments: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    fields_value = arguments.get("fields")
    if isinstance(fields_value, list):
        return _build_schema_from_fields(arguments)
    if isinstance(arguments.get("schema"), (dict, str)):
        return _build_schema_from_schema_argument(arguments)
    if ("type" in arguments and "properties" in arguments) or (
        "$schema" in arguments and "properties" in arguments
    ):
        return arguments, None
    raise ValueError("Missing or invalid schema or fields in arguments.")


def _validate_elicitation_schema_field_limit(schema: dict[str, Any]) -> None:
    properties = schema.get("properties")
    props = properties if isinstance(properties, dict) else {}
    if len(props) > 7:
        raise ValueError(f"Error: schema requests {len(props)} fields; the maximum allowed is 7.")


def _build_elicitation_request_payload(
    *,
    schema: dict[str, Any],
    message: str | None,
    agent_name: str | None,
) -> dict[str, Any]:
    resolved_agent_name = agent_name or "Unknown Agent"
    return {
        "prompt": message or schema.get("title") or "Please complete this form:",
        "description": schema.get("description"),
        "request_id": f"__human_input__{uuid.uuid4()}",
        "metadata": {
            "agent_name": resolved_agent_name,
            "requested_schema": schema,
        },
    }


async def run_elicitation_form(arguments: dict | str, agent_name: str | None = None) -> str:
    """Parse arguments into a JSON Schema or simplified fields spec and invoke the registered callback.

    Returns the response string from the callback. Raises if no callback is registered.
    """
    arguments_dict = _coerce_elicitation_arguments(arguments)
    schema_dict, message = _resolve_elicitation_schema(arguments_dict)
    _validate_elicitation_schema_field_limit(schema_dict)
    request_payload = _build_elicitation_request_payload(
        schema=schema_dict,
        message=message,
        agent_name=agent_name,
    )
    cb = get_elicitation_input_callback()
    if not cb:
        raise RuntimeError("No elicitation input callback registered")

    response_text: str = await cb(
        request_payload,
        agent_name or "Unknown Agent",
        "__human_input__",
        None,
    )

    return response_text


# -----------------------
# FastMCP tool builder
# -----------------------


def get_elicitation_fastmcp_tool() -> FastMCPTool:
    async def elicit(
        title: str | None = None,
        description: str | None = None,
        message: str | None = None,
        fields: list[FormField] = Field(default_factory=list, max_length=7),
    ) -> str:
        args = {
            "title": title,
            "description": description,
            "message": message,
            "fields": [f.model_dump() if isinstance(f, BaseModel) else f for f in fields],
        }
        return await run_elicitation_form(args)

    tool = FastMCPTool.from_function(elicit)
    tool.name = HUMAN_INPUT_TOOL_NAME
    tool.description = (
        "Collect structured input from a human via a simple form. Provide up to 7 fields "
        "(text, textarea, number, checkbox, radio). Fields can include label, help, default; "
        "numbers support min/max; radio supports options (value/label); optional message is shown above the form."
    )
    # Harmonize input schema with the sanitized MCP schema for provider compatibility
    tool.parameters = get_elicitation_tool().inputSchema
    return tool
