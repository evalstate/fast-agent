"""Contract tests for programmatic tool calling running in a real Monty sandbox."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import pytest_asyncio
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings
from fast_agent.context import Context
from fast_agent.mcp.prompt import Prompt
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
    assert "async def list_tool(*, from_: str | None = None, **kwargs: Any)" in description


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


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_print_capture_is_bounded_on_success_and_failure(runtime, monkeypatch, fails) -> None:
    monkeypatch.setattr("fast_agent.ptc.runtime._MAX_PRINT_CHARS", 128)
    dispatch = RecordingDispatcher()
    code = "for i in range(1000):\n    print('α' * 100, end='')\n"
    code += "raise ValueError('failed after printing')" if fails else "'finished'"

    result = await runtime.execute({"code": code}, dispatch)

    assert bool(result.is_error) == fails, _text(result)
    text = _text(result)
    assert "α" * 128 in text
    assert "α" * 129 not in text
    assert text.count("[Printed output truncated]") == 1
    assert "failed after printing" in text if fails else "finished" in text
    assert len(text) < 1000

    next_result = await runtime.execute({"code": "print('fresh output')"}, dispatch)
    assert not next_result.is_error, _text(next_result)
    assert "fresh output" in _text(next_result)
    assert "truncated" not in _text(next_result)


@pytest.mark.asyncio
async def test_tool_documentation_is_safe_python_source(runtime) -> None:
    runtime.model_tools(
        [
            Tool(
                name="documented",
                description="Read C:\\Users\\name, \"\"\"quoted\"\"\" and '''quoted'''.\nNext line.\x00",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Use C:\\Users\\name"}
                    },
                },
            )
        ]
    )
    dispatch = RecordingDispatcher()

    result = await runtime.execute({"code": "1 + 1"}, dispatch)
    called = await runtime.execute({"code": "await documented(path='example')"}, dispatch)

    assert not result.is_error, _text(result)
    assert not called.is_error, _text(called)
    assert dispatch.calls == [("documented", {"path": "example"})]


@pytest.mark.asyncio
async def test_optional_nullable_argument_distinguishes_omission_from_explicit_null(
    runtime,
) -> None:
    runtime.model_tools(
        [
            Tool(
                name="update",
                input_schema={
                    "type": "object",
                    "properties": {"from": {"type": ["string", "null"], "default": "default"}},
                },
            )
        ]
    )
    dispatch = RecordingDispatcher()

    result = await runtime.execute(
        {"code": "await update()\nawait update(from_=None)\nawait update(from_='new')"}, dispatch
    )

    assert not result.is_error, _text(result)
    assert dispatch.calls == [
        ("update", {}),
        ("update", {"from": None}),
        ("update", {"from": "new"}),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("additional", [True, {}, {"type": "integer"}, False, None])
async def test_named_properties_respect_additional_properties(runtime, additional) -> None:
    schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}, "kwargs": {"type": "string"}},
        "required": ["id"],
    }
    if additional is not None:
        schema["additionalProperties"] = additional
    runtime.model_tools([Tool(name="update", input_schema=schema)])
    dispatch = RecordingDispatcher()

    result = await runtime.execute(
        {"code": "await update(id='x', kwargs='named', value=1)"}, dispatch
    )

    if additional is False:
        assert result.is_error
        assert "NOT executed" in _text(result)
        assert dispatch.calls == []
    else:
        assert not result.is_error, _text(result)
        assert dispatch.calls == [("update", {"id": "x", "kwargs": "named", "value": 1})]


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", ["clear", "clear_prompts", "load_history"])
@pytest.mark.parametrize("type_check", [True, False])
async def test_conversation_resets_discard_sandbox_state(reset, type_check) -> None:
    agent = McpAgent(
        AgentConfig(name="ptc-reset", servers=[]),
        context=Context(
            config=Settings(
                ptc=ProgrammaticToolCallingSettings(enabled=True, type_check=type_check)
            )
        ),
    )
    try:
        await agent.list_tools()
        seeded = await agent.call_tool(PTC_TOOL_NAME, {"code": "saved: int = 42\nsaved"})
        assert not seeded.is_error, _text(seeded)

        # Ordinary history replacement (including compaction) keeps the REPL intact.
        agent.load_message_history([Prompt.user("updated history")])
        await agent.list_tools()
        retained = await agent.call_tool(PTC_TOOL_NAME, {"code": "saved"})
        assert not retained.is_error, _text(retained)
        assert "42" in _text(retained)

        if reset == "load_history":
            agent.load_message_history([Prompt.user("new conversation")], clear_state=True)
        else:
            agent.clear(clear_prompts=reset == "clear_prompts")
        await agent.list_tools()
        missing = await agent.call_tool(PTC_TOOL_NAME, {"code": "saved"})
        assert missing.is_error, _text(missing)

        # A new annotation must also work: old type-check state is gone.
        fresh = await agent.call_tool(PTC_TOOL_NAME, {"code": "saved: str = 'fresh'\nsaved"})
        assert not fresh.is_error, _text(fresh)
        assert "fresh" in _text(fresh)
    finally:
        await agent.shutdown()
