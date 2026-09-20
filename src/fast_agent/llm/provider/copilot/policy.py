"""Wire-payload policy shared by the two Copilot protocol adapters."""

from collections.abc import Mapping
from typing import Any

from fast_agent.llm.provider.copilot.models import CopilotModelSpec

_PROTECTED_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "host",
    "x-initiator",
    "copilot-integration-id",
    "copilot-vision-request",
}


def reject_overrides(values: Mapping[str, object]) -> None:
    for key in ("api_key", "base_url"):
        if values.get(key) is not None:
            raise ValueError(f"Copilot {key} must come from the endpoint broker.")
    for key in ("default_headers", "extra_headers"):
        headers = values.get(key)
        if isinstance(headers, Mapping) and any(
            str(name).lower() in _PROTECTED_HEADERS for name in headers
        ):
            raise ValueError("Copilot authentication and routing headers cannot be overridden.")
    extra = values.get("extra_body")
    if isinstance(extra, Mapping):
        reject_overrides(extra)


def contains_type(value: object, types: set[str]) -> bool:
    if isinstance(value, Mapping):
        kind = value.get("type")
        return (isinstance(kind, str) and kind in types) or any(
            contains_type(value.get(key), types) for key in ("content", "output")
        )
    if isinstance(value, (list, tuple)):
        return any(contains_type(child, types) for child in value)
    return False


def reject_files(value: object) -> None:
    if isinstance(value, Mapping):
        kind = value.get("type")
        if kind in ("input_file", "file", "document") or (
            kind == "input_image" and value.get("file_id")
        ):
            raise ValueError("Copilot provider file APIs are not supported.")
        # Traverse wire content, not arbitrary local tool arguments (which may
        # legitimately contain a field named file_id or type="file").
        for key in ("content", "output", "source"):
            reject_files(value.get(key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_files(child)


def request_headers(payload: object) -> dict[str, str]:
    """Classify the latest semantic input, not the presence of old tool history."""
    latest = payload[-1] if isinstance(payload, list) and payload else payload
    agent = False
    if isinstance(latest, Mapping):
        agent = latest.get("role") == "assistant" or latest.get("type") in (
            "function_call_output",
            "custom_tool_call_output",
        )
        if latest.get("role") == "user":
            content = latest.get("content")
            agent = (
                isinstance(content, list)
                and bool(content)
                and all(
                    isinstance(block, Mapping) and block.get("type") == "tool_result"
                    for block in content
                )
            )
    headers = {"x-initiator": "agent" if agent else "user"}
    if contains_type(payload, {"image", "input_image"}):
        headers["copilot-vision-request"] = "true"
    return headers


def apply_policy(
    arguments: dict[str, Any],
    spec: CopilotModelSpec,
    broker_headers: Mapping[str, str],
) -> dict[str, Any]:
    # SDK request dictionaries are the deliberately dynamic boundary here.
    reject_overrides(arguments)
    headers = arguments.get("extra_headers") or {}
    if {name.lower() for name in headers} & {name.lower() for name in broker_headers}:
        raise ValueError("Copilot broker headers cannot be overridden.")
    extra = arguments.get("extra_body") or {}
    effective = {**arguments, **extra}
    if "service_tier" in effective:
        raise ValueError("Copilot does not support service_tier selection; omit the field.")
    if effective.get("model") != spec.model_id:
        raise ValueError("Copilot request model cannot override the bound endpoint.")
    if effective.get("mcp_servers") or effective.get("container"):
        raise ValueError("Copilot native MCP and hosted tools are not supported.")
    choice = effective.get("tool_choice")
    if not spec.supports_named_tool_choice and (
        isinstance(choice, dict)
        and choice.get("type") in {"tool", "any", "function"}
        or choice == "required"
    ):
        raise ValueError(
            f"Copilot adapter requires automatic tool selection for {spec.model_id}; "
            "forced modes are not enabled."
        )
    tools = effective.get("tools") or []
    for tool in tools:
        allowed = {None, "custom"} if spec.wire_api == "messages" else {"function", "custom"}
        if spec.wire_api == "responses" and spec.supports_web_search:
            allowed.add("web_search")
        if tool.get("type") not in allowed:
            raise ValueError(f"Copilot provider-hosted tool is not supported for {spec.model_id}.")
    if spec.wire_api == "messages" and tools:
        # The Copilot Messages gateway rejects this Anthropic tool field.
        updated = [
            {key: value for key, value in tool.items() if key != "eager_input_streaming"}
            for tool in tools
        ]
        if "tools" in extra:
            arguments["extra_body"] = {**extra, "tools": updated}
        else:
            arguments["tools"] = updated
    input_key = "messages" if spec.wire_api == "messages" else "input"
    payload = effective.get(input_key)
    reject_files(payload)
    headers = arguments.get("extra_headers") or {}
    arguments["extra_headers"] = {**headers, **request_headers(payload)}
    if spec.wire_api == "responses":
        if effective.get("previous_response_id") or effective.get("conversation"):
            raise ValueError("Copilot requires full-history replay, not stored continuations.")
        arguments["store"] = False
        if "store" in extra:
            arguments["extra_body"] = {**arguments["extra_body"], "store": False}
    return arguments
