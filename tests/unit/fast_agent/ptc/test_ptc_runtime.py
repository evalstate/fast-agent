"""Contract tests for programmatic tool calling running in a real Monty sandbox."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import pytest_asyncio
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.config import ProgrammaticToolCallingSettings
from fast_agent.ptc import PTC_TOOL_NAME, ProgrammaticToolRuntime

pytest.importorskip("pydantic_monty")

TOOLS = [
    Tool(
        name="orders__get_orders",
        description="Get orders for a customer.",
        input_schema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "status": {"type": "string", "enum": ["open", "closed"]},
            },
            "required": ["customer_id"],
        },
    ),
    Tool(
        name="list",
        description="A tool whose name shadows a builtin and has a keyword parameter.",
        input_schema={"type": "object", "properties": {"from": {"type": "string"}}},
    ),
    Tool(
        name="delete_everything",
        description="Always fails.",
        input_schema={"type": "object", "properties": {}},
    ),
    Tool(name="execute", description="Shell", input_schema={"type": "object"}),
]


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        self.calls.append((name, arguments))
        await asyncio.sleep(0)
        if name == "delete_everything":
            return CallToolResult(content=[TextContent(type="text", text="denied")], is_error=True)
        payload = {"tool": name, "arguments": arguments, "call_id": call_id}
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))])


@pytest_asyncio.fixture
async def runtime():
    runtime = ProgrammaticToolRuntime(
        ProgrammaticToolCallingSettings(enabled=True, direct_tools=["execute"])
    )
    runtime.model_tools(TOOLS)
    yield runtime
    await runtime.close()


def _text(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def test_model_sees_code_tool_plus_direct_tools_only() -> None:
    runtime = ProgrammaticToolRuntime(
        ProgrammaticToolCallingSettings(enabled=True, direct_tools=["execute"])
    )

    tools = runtime.model_tools(TOOLS)

    assert [tool.name for tool in tools] == [PTC_TOOL_NAME, "execute"]
    description = tools[0].description or ""
    assert "async def orders__get_orders(*, customer_id: str" in description
    assert "async def list_tool(*, from_: str | None = None)" in description


@pytest.mark.asyncio
async def test_concurrent_calls_and_state_persists_between_executions(runtime) -> None:
    dispatch = RecordingDispatcher()

    first = await runtime.execute(
        {
            "code": (
                "import asyncio\n"
                "rs = await asyncio.gather("
                "*[orders__get_orders(customer_id=c) for c in ['a', 'b', 'c']])\n"
                "print(len(rs))\n"
                "[r['arguments']['customer_id'] for r in rs]"
            )
        },
        dispatch,
    )
    second = await runtime.execute({"code": "rs[0]['call_id']"}, dispatch)

    assert not first.is_error, _text(first)
    assert "3" in _text(first) and '["a", "b", "c"]' in _text(first)
    assert len(dispatch.calls) == 3
    assert not second.is_error and "ptc-1-" in _text(second)


@pytest.mark.asyncio
async def test_sanitized_names_map_back_and_unset_optionals_are_dropped(runtime) -> None:
    dispatch = RecordingDispatcher()

    result = await runtime.execute(
        {"code": "await list_tool(from_='x')\nawait orders__get_orders(customer_id='a')"},
        dispatch,
    )

    assert not result.is_error, _text(result)
    assert dispatch.calls == [("list", {"from": "x"}), ("orders__get_orders", {"customer_id": "a"})]


@pytest.mark.asyncio
async def test_type_errors_reject_code_before_any_tool_runs(runtime) -> None:
    dispatch = RecordingDispatcher()

    result = await runtime.execute(
        {"code": "await list_tool(from_='x')\nawait orders__get_orders(status='bogus')"},
        dispatch,
    )

    assert result.is_error
    assert "NOT executed" in _text(result)
    assert dispatch.calls == []


@pytest.mark.asyncio
async def test_tool_errors_are_catchable_and_failures_report_completed_calls(runtime) -> None:
    dispatch = RecordingDispatcher()

    caught = await runtime.execute(
        {"code": "try:\n    await delete_everything()\nexcept RuntimeError as e:\n    print(e)"},
        dispatch,
    )
    failed = await runtime.execute(
        {"code": "await orders__get_orders(customer_id='a')\nawait delete_everything()"},
        dispatch,
    )

    assert not caught.is_error and "denied" in _text(caught)
    assert failed.is_error
    assert "orders__get_orders" in _text(failed) and "denied" in _text(failed)
