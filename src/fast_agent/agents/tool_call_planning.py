"""Small primitives for planned tool-call execution."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

from mcp_types import CallToolResult

from fast_agent.types import RequestParams


@dataclass(frozen=True)
class PlannedToolCall:
    correlation_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class UnavailableToolCall:
    correlation_id: str
    name: str


@dataclass(frozen=True)
class ToolCallPlan:
    planned_calls: list[PlannedToolCall]
    unavailable_calls: list[UnavailableToolCall]


@dataclass(frozen=True)
class PlannedToolCallResult:
    result: CallToolResult
    duration_ms: float


type ToolCallExecutor = Callable[
    [PlannedToolCall, RequestParams | None],
    Awaitable[CallToolResult],
]


def plan_tool_calls(
    tool_call_items: list[tuple[str, Any]],
    *,
    available_tools: Collection[str],
    execution_tools: Mapping[str, object],
) -> ToolCallPlan:
    known_tool_names = set(available_tools) | set(execution_tools)
    casefolded_tool_names: dict[str, list[str]] = {}
    for known_tool_name in known_tool_names:
        casefolded_tool_names.setdefault(known_tool_name.casefold(), []).append(known_tool_name)

    planned_calls: list[PlannedToolCall] = []
    unavailable_calls: list[UnavailableToolCall] = []
    for correlation_id, tool_request in tool_call_items:
        requested_tool_name = tool_request.params.name
        tool_args = tool_request.params.arguments or {}
        tool_name = requested_tool_name
        if tool_name not in known_tool_names:
            casefolded_matches = casefolded_tool_names.get(tool_name.casefold(), [])
            if len(casefolded_matches) == 1:
                tool_name = casefolded_matches[0]
            else:
                unavailable_calls.append(
                    UnavailableToolCall(
                        correlation_id=correlation_id,
                        name=requested_tool_name,
                    )
                )
                continue
        planned_calls.append(
            PlannedToolCall(
                correlation_id=correlation_id,
                name=tool_name,
                arguments=tool_args,
            )
        )
    return ToolCallPlan(
        planned_calls=planned_calls,
        unavailable_calls=unavailable_calls,
    )


async def execute_planned_tool_call(
    planned_call: PlannedToolCall,
    *,
    execute_tool: ToolCallExecutor,
    request_params: RequestParams | None,
) -> PlannedToolCallResult:
    start_time = time.perf_counter()
    result = await execute_tool(planned_call, request_params)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return PlannedToolCallResult(result=result, duration_ms=duration_ms)
