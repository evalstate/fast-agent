"""Small primitives for planned tool-call execution."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

from mcp.types import CallToolResult

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
    unavailable_call: UnavailableToolCall | None = None


@dataclass(frozen=True)
class PlannedToolCallResult:
    result: CallToolResult
    duration_ms: float


type ToolCallExecutor = Callable[
    [str, dict[str, Any], RequestParams | None],
    Awaitable[CallToolResult],
]


def plan_tool_calls(
    tool_call_items: list[tuple[str, Any]],
    *,
    available_tools: Collection[str],
    execution_tools: Mapping[str, object],
) -> ToolCallPlan:
    planned_calls: list[PlannedToolCall] = []
    for correlation_id, tool_request in tool_call_items:
        tool_name = tool_request.params.name
        tool_args = tool_request.params.arguments or {}
        if tool_name not in available_tools and tool_name not in execution_tools:
            return ToolCallPlan(
                planned_calls=planned_calls,
                unavailable_call=UnavailableToolCall(
                    correlation_id=correlation_id,
                    name=tool_name,
                ),
            )
        planned_calls.append(
            PlannedToolCall(
                correlation_id=correlation_id,
                name=tool_name,
                arguments=tool_args,
            )
        )
    return ToolCallPlan(planned_calls=planned_calls)


async def execute_planned_tool_call(
    planned_call: PlannedToolCall,
    *,
    execute_tool: ToolCallExecutor,
    request_params: RequestParams | None,
) -> PlannedToolCallResult:
    start_time = time.perf_counter()
    result = await execute_tool(planned_call.name, planned_call.arguments, request_params)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return PlannedToolCallResult(result=result, duration_ms=duration_ms)
