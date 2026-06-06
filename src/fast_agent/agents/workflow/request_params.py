"""Request parameter helpers for workflow child delegation."""

from __future__ import annotations

from typing import Any

from fast_agent.types import RequestParams


def _copy_explicit_non_none(
    delegated: dict[str, Any],
    field_name: str,
    value: Any,
    explicit_fields: set[str],
) -> None:
    if field_name in explicit_fields and value is not None:
        delegated[field_name] = value


def child_request_params(request_params: RequestParams | None) -> RequestParams | None:
    """Forward workflow-control params without overriding child LLM defaults."""
    if request_params is None:
        return None

    delegated: dict[str, Any] = {}
    explicit_fields = request_params.model_fields_set
    if "use_history" in explicit_fields:
        delegated["use_history"] = request_params.use_history
    if request_params.tool_execution_handler is not None:
        delegated["tool_execution_handler"] = request_params.tool_execution_handler
    if "emit_loop_progress" in explicit_fields:
        delegated["emit_loop_progress"] = request_params.emit_loop_progress
    if "tool_result_mode" in explicit_fields:
        delegated["tool_result_mode"] = request_params.tool_result_mode
    _copy_explicit_non_none(delegated, "mcp_metadata", request_params.mcp_metadata, explicit_fields)
    _copy_explicit_non_none(
        delegated, "batch_context", request_params.batch_context, explicit_fields
    )

    return RequestParams(**delegated) if delegated else None
