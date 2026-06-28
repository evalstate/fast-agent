"""Shared agent-as-tool schema and argument rendering helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from fast_agent.core.template_render import render_mapping_template
from fast_agent.llm.request_params import (
    ToolResultMode,
    response_mode_to_tool_result_mode,
    tool_result_mode_allows_response_mode,
)
from fast_agent.utils.text import strip_casefold

RESPONSE_MODE_FIELD = "response_mode"
RESPONSE_MODE_OVERRIDES: Mapping[str, ToolResultMode | None] = {
    "inherit": None,
    "postprocess": response_mode_to_tool_result_mode("postprocess"),
    "passthrough": response_mode_to_tool_result_mode("passthrough"),
}
_MISSING_RESPONSE_MODE = object()

type AgentToolArgumentRenderer = Callable[[Mapping[str, Any]], str] | str


@dataclass(frozen=True, slots=True)
class ResponseModeControl:
    arguments: dict[str, Any]
    tool_result_mode_override: ToolResultMode | None = None
    error: str | None = None


def default_agent_tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Message to send to the agent",
            },
        },
        "required": ["message"],
    }


def response_mode_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "description": "Override how the child agent returns tool results for this call.",
        "enum": list(RESPONSE_MODE_OVERRIDES),
        "default": "inherit",
    }


def tool_result_mode_allows_agent_response_mode(mode: ToolResultMode) -> bool:
    return tool_result_mode_allows_response_mode(mode)


def response_mode_control_enabled(
    configured_schema: dict[str, Any] | None,
    *,
    response_mode_enabled: bool,
) -> bool:
    if not response_mode_enabled:
        return False
    if configured_schema is None:
        return True
    if configured_schema.get("type") != "object":
        return False
    properties = configured_schema.get("properties")
    if properties is None:
        return True
    return isinstance(properties, dict) and RESPONSE_MODE_FIELD not in properties


def augment_schema_with_response_mode(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") != "object":
        return schema

    properties = schema.get("properties")
    if properties is not None and not isinstance(properties, dict):
        return schema

    updated_schema = deepcopy(schema)
    updated_properties = updated_schema.setdefault("properties", {})
    if not isinstance(updated_properties, dict):
        return schema

    updated_properties.setdefault(RESPONSE_MODE_FIELD, response_mode_schema())
    return updated_schema


def resolved_agent_tool_schema(
    configured_schema: dict[str, Any] | None,
    *,
    response_mode_control: bool,
) -> dict[str, Any]:
    schema = configured_schema if configured_schema is not None else default_agent_tool_schema()
    if response_mode_control:
        return augment_schema_with_response_mode(schema)
    return schema


def uses_structured_args(
    configured_schema: dict[str, Any] | None,
    *,
    response_mode_control: bool,
) -> bool:
    if configured_schema is None:
        return False
    properties = configured_schema.get("properties")
    if not isinstance(properties, dict):
        return True
    if response_mode_control:
        property_names = {name for name in properties if name != RESPONSE_MODE_FIELD}
    else:
        property_names = set(properties)
    return property_names != {"message"}


def render_structured_args(arguments: Mapping[str, Any]) -> str:
    return json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True, default=str)


def render_agent_tool_arguments(
    arguments: Mapping[str, Any],
    *,
    configured_schema: dict[str, Any] | None = None,
    response_mode_control: bool = False,
    render_arguments: AgentToolArgumentRenderer | None = None,
) -> str:
    if isinstance(render_arguments, str):
        result = render_mapping_template(
            render_arguments,
            arguments,
            json_placeholder="arguments_json",
        )
        if result.missing:
            fields = ", ".join(result.missing)
            raise ValueError(f"Missing template field: {fields}")
        return result.text

    if render_arguments is not None:
        return render_arguments(arguments)

    if uses_structured_args(configured_schema, response_mode_control=response_mode_control):
        return render_structured_args(arguments)
    if isinstance(arguments.get("message"), str):
        return arguments["message"]
    if isinstance(arguments.get("text"), str):
        return arguments["text"]
    return str(dict(arguments)) if arguments else ""


def split_response_mode_control(
    arguments: Mapping[str, Any],
    *,
    enabled: bool,
) -> ResponseModeControl:
    if not enabled:
        return ResponseModeControl(arguments=dict(arguments))

    sanitized_arguments = dict(arguments)
    raw_mode = sanitized_arguments.pop(RESPONSE_MODE_FIELD, _MISSING_RESPONSE_MODE)
    if raw_mode is _MISSING_RESPONSE_MODE:
        return ResponseModeControl(arguments=sanitized_arguments)
    if not isinstance(raw_mode, str):
        return ResponseModeControl(
            arguments=sanitized_arguments,
            error="response_mode must be one of: inherit, postprocess, passthrough",
        )

    normalized = strip_casefold(raw_mode)
    if normalized not in RESPONSE_MODE_OVERRIDES:
        return ResponseModeControl(
            arguments=sanitized_arguments,
            error=(
                f"Invalid response_mode '{raw_mode}'. "
                "Expected one of: inherit, postprocess, passthrough"
            ),
        )

    return ResponseModeControl(
        arguments=sanitized_arguments,
        tool_result_mode_override=RESPONSE_MODE_OVERRIDES[normalized],
    )


__all__ = [
    "RESPONSE_MODE_FIELD",
    "RESPONSE_MODE_OVERRIDES",
    "ResponseModeControl",
    "default_agent_tool_schema",
    "render_agent_tool_arguments",
    "render_structured_args",
    "resolved_agent_tool_schema",
    "response_mode_control_enabled",
    "split_response_mode_control",
    "tool_result_mode_allows_agent_response_mode",
    "uses_structured_args",
]
