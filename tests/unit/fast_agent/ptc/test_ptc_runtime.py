"""Contract tests for programmatic tool calling running in a real Monty sandbox."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import pytest_asyncio
from mcp_types import CallToolResult, ImageContent, TextContent, Tool
from PIL import Image as PILImage

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.tool_result_channels import build_tool_result_message
from fast_agent.config import ProgrammaticToolCallingSettings, Settings, ShellSettings
from fast_agent.constants import (
    FAST_AGENT_PTC_CALLS,
    FAST_AGENT_PTC_VALUE,
    FAST_AGENT_URL_ELICITATION_CHANNEL,
)
from fast_agent.context import Context
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.tool_result_metadata import set_url_elicitation_required_payload
from fast_agent.mcp.url_elicitation_required import (
    URLElicitationDisplayItem,
    URLElicitationRequiredDisplayPayload,
)
from fast_agent.ptc import PTC_TOOL_NAME, ProgrammaticToolRuntime
from fast_agent.tools.invocation_context import is_programmatic_tool_call, outer_tool_call_id
from fast_agent.tools.local_shell_executor import LocalEnvironment

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
    assert "async def list_tool(*, from_: str = ...)" in description


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
@pytest.mark.parametrize("catch_error", [False, True])
async def test_gather_settles_sibling_side_effects_before_reporting(runtime, catch_error) -> None:
    write_started = asyncio.Event()
    failure_returned = asyncio.Event()
    release_write = asyncio.Event()
    writes: list[str] = []

    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        if name == "delete_everything":
            await write_started.wait()
            failure_returned.set()
            return CallToolResult(content=[TextContent(type="text", text="denied")], is_error=True)
        write_started.set()
        await release_write.wait()
        writes.append(call_id)
        return CallToolResult(content=[TextContent(type="text", text="written")])

    expression = "await asyncio.gather(orders__get_orders(customer_id='a'), delete_everything())"
    code = "import asyncio\n" + (
        f"try:\n    {expression}\nexcept RuntimeError:\n    pass" if catch_error else expression
    )
    execution = asyncio.create_task(runtime.execute({"code": code}, dispatch))
    try:
        await asyncio.wait_for(failure_returned.wait(), timeout=5)
        done, _ = await asyncio.wait({execution}, timeout=0.05)
        assert not done, "Execution returned while a sibling write was still pending"
        release_write.set()
        result = await asyncio.wait_for(execution, timeout=5)

        assert bool(result.is_error) is not catch_error
        assert len(writes) == 1
        assert "2 tool calls, 1 failed" in _text(result)
        records = (result.meta or {})[FAST_AGENT_PTC_CALLS]
        assert {record["tool"] for record in records} == {
            "orders__get_orders",
            "delete_everything",
        }
        if not catch_error:
            assert "avoid repeating side effects" in _text(result)
            assert "orders__get_orders" in _text(result)
    finally:
        release_write.set()
        await execution


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during_drain", [False, True])
async def test_cancelled_execution_settles_tools_and_can_run_again(
    runtime, cancel_during_drain
) -> None:
    write_started = asyncio.Event()
    failure_returned = asyncio.Event()
    write_settled = asyncio.Event()

    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        if name == "delete_everything":
            await write_started.wait()
            failure_returned.set()
            return CallToolResult(content=[TextContent(type="text", text="denied")], is_error=True)
        write_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            write_settled.set()
        return CallToolResult(content=[])

    code = "await orders__get_orders(customer_id='a')"
    if cancel_during_drain:
        code = f"import asyncio\nawait asyncio.gather({code.removeprefix('await ')}, delete_everything())"
    execution = asyncio.create_task(runtime.execute({"code": code}, dispatch))
    try:
        await asyncio.wait_for(write_started.wait(), timeout=5)
        if cancel_during_drain:
            await asyncio.wait_for(failure_returned.wait(), timeout=5)
            await asyncio.wait({execution}, timeout=0.05)
    finally:
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(execution, timeout=5)

    assert write_settled.is_set()
    dispatch_again = RecordingDispatcher()
    recovered = await runtime.execute({"code": "saved = 42\nsaved"}, dispatch_again)
    reused = await runtime.execute({"code": "saved + 1"}, dispatch_again)
    assert not recovered.is_error and _text(recovered).endswith("Result:\n42")
    assert not reused.is_error and _text(reused).endswith("Result:\n43")


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_in_context", [False, True])
async def test_agent_context_exit_closes_ptc_resources(raise_in_context) -> None:
    agent = McpAgent(
        AgentConfig(name="ptc-lifecycle", servers=[]),
        connection_persistence=False,
        context=Context(config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=True))),
    )
    try:
        try:
            async with agent:
                await agent.list_tools()
                result = await agent.call_tool(PTC_TOOL_NAME, {"code": "42"}, "call-1")
                assert not result.is_error, _text(result)
                runtime = agent._ptc_runtime
                assert runtime is not None
                assert runtime._session is not None and runtime._pool is not None
                if raise_in_context:
                    raise ValueError("context failed")
        except ValueError as exc:
            assert raise_in_context and str(exc) == "context failed"
        runtime = agent._ptc_runtime
        assert runtime is not None
        assert runtime._session is None
        assert runtime._pool is None
    finally:
        await agent.shutdown()


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

    if additional in (False, None):
        # Absent additionalProperties is treated as closed so misspelled arguments are caught.
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


def _url_payload(server_name: str, url: str) -> URLElicitationRequiredDisplayPayload:
    return URLElicitationRequiredDisplayPayload(
        server_name=server_name,
        request_method="tools/call",
        elicitations=[URLElicitationDisplayItem(message="Sign in", url=url, elicitation_id=url)],
        issues=[],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_nested_url_elicitations_reach_the_tool_result_message(runtime, fails) -> None:
    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        del arguments, call_id
        is_error = name == "delete_everything"
        result = CallToolResult(content=[TextContent(type="text", text="{}")], is_error=is_error)
        set_url_elicitation_required_payload(result, _url_payload(name, f"https://{name}.test"))
        return result

    tail = "await delete_everything()" if fails else "None"
    code = (
        "try:\n    await delete_everything()\nexcept RuntimeError:\n    pass\n"
        f"await orders__get_orders(customer_id='c1')\n{tail}"
    )
    result = await runtime.execute({"code": code}, dispatch)
    assert result.is_error is fails

    message = build_tool_result_message({"call-1": result})
    assert message.channels is not None
    [block] = message.channels[FAST_AGENT_URL_ELICITATION_CHANNEL]
    assert isinstance(block, TextContent)
    urls = [item["url"] for payload in json.loads(block.text) for item in payload["elicitations"]]
    expected = ["https://delete_everything.test", "https://orders__get_orders.test"]
    assert urls == expected + (["https://delete_everything.test"] if fails else [])


@pytest.mark.asyncio
async def test_misspelled_optional_argument_is_rejected_without_open_schema(runtime) -> None:
    runtime.model_tools(
        [
            Tool(
                name="search",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["query"],
                },
            )
        ]
    )
    dispatch = RecordingDispatcher()

    typo = await runtime.execute({"code": "await search(query='x', limt=5)"}, dispatch)
    valid = await runtime.execute({"code": "await search(query='x', limit=5)"}, dispatch)

    assert typo.is_error and "NOT executed" in _text(typo)
    assert not valid.is_error, _text(valid)
    assert dispatch.calls == [("search", {"query": "x", "limit": 5})]


@pytest.mark.asyncio
async def test_explicit_none_is_only_accepted_for_nullable_optionals(runtime) -> None:
    runtime.model_tools(
        [
            Tool(
                name="filter",
                input_schema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string"},
                        "status": {"anyOf": [{"enum": ["open", "closed"]}, {"type": "null"}]},
                    },
                },
            )
        ]
    )
    dispatch = RecordingDispatcher()

    rejected = await runtime.execute({"code": "await filter_tool(region=None)"}, dispatch)
    accepted = await runtime.execute(
        {"code": "await filter_tool()\nawait filter_tool(status=None)"}, dispatch
    )

    assert rejected.is_error and "NOT executed" in _text(rejected)
    assert not accepted.is_error, _text(accepted)
    assert dispatch.calls == [("filter", {}), ("filter", {"status": None})]


def test_nullable_unions_render_each_member_once() -> None:
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(enabled=True))
    schema = {
        "type": "object",
        "properties": {
            "region": {"anyOf": [{"enum": ["EMEA", "AMER"]}, {"type": "null"}]},
            "tags": {"type": ["array", "null"], "items": {"type": ["string", "null"]}},
        },
    }

    description = runtime.model_tools([Tool(name="lookup", input_schema=schema)])[0].description

    assert 'region: Literal["EMEA", "AMER"] | None = ...' in (description or "")
    assert "tags: list[str | None] | None = ..." in (description or "")


_PIXEL = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
)


@pytest.mark.asyncio
async def test_referenced_definitions_are_typed_and_checked(runtime) -> None:
    runtime.model_tools(
        [
            Tool(
                name="label_run",
                input_schema={
                    "type": "object",
                    "$defs": {
                        "Colour": {"enum": ["red", "blue"]},
                        "Label": {"type": "string", "maxLength": 32},
                    },
                    "properties": {
                        "colour": {"$ref": "#/$defs/Colour"},
                        "label": {"anyOf": [{"$ref": "#/$defs/Label"}, {"type": "null"}]},
                    },
                    "required": ["colour"],
                },
            )
        ]
    )
    dispatch = RecordingDispatcher()

    description = runtime.tool().description or ""
    rejected = await runtime.execute({"code": "await label_run(colour='green')"}, dispatch)
    accepted = await runtime.execute(
        {"code": "await label_run(colour='red', label=None)"}, dispatch
    )

    assert 'colour: Literal["red", "blue"], label: str | None = ...' in description
    assert rejected.is_error and "NOT executed" in _text(rejected)
    assert not accepted.is_error, _text(accepted)


def test_tools_declare_their_exposure_and_program_return_type() -> None:
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(enabled=True))
    control = Tool(
        name="slash_command",
        input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
        meta={"fast_agent": {"ptc": {"expose": "direct"}}},
    )
    returns = {"type": "object", "properties": {"answer": {"type": "string"}}}
    delegate = Tool(
        name="delegate",
        input_schema={"type": "object", "properties": {}},
        meta={"fast_agent": {"ptc": {"returns": returns}}},
    )

    code_tool, *direct = runtime.model_tools([control, delegate])

    assert [tool.name for tool in direct] == ["slash_command"]
    assert "async def delegate() -> dict[str, Any]:" in (code_tool.description or "")
    assert '"answer"' in (code_tool.description or "")
    assert "slash_command" not in (code_tool.description or "")


@pytest.mark.asyncio
async def test_media_reaches_the_model_only_when_the_script_returns_its_handle(runtime) -> None:
    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        del name, arguments, call_id
        return CallToolResult(
            content=[
                TextContent(type="text", text="screenshot"),
                ImageContent(type="image", data=_PIXEL, mime_type="image/png"),
            ]
        )

    printed = await runtime.execute(
        {"code": "caption, shot = await orders__get_orders(customer_id='a')\nprint(shot)"},
        dispatch,
    )
    returned = await runtime.execute({"code": "{'caption': caption, 'image': shot}"}, dispatch)

    assert not printed.is_error, _text(printed)
    assert "media_id" in _text(printed)
    assert [block.type for block in printed.content] == ["text"]
    assert not returned.is_error, _text(returned)
    assert [block.type for block in returned.content] == ["text", "image"]
    image = returned.content[1]
    assert isinstance(image, ImageContent) and image.data == _PIXEL


@pytest.mark.asyncio
async def test_program_values_replace_model_text_only_for_the_sandbox_caller(runtime) -> None:
    seen_ids: list[str] = []

    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        del name, arguments
        seen_ids.append(call_id)
        result = CallToolResult(content=[TextContent(type="text", text="answer + notice")])
        if is_programmatic_tool_call(call_id):
            result.meta = {FAST_AGENT_PTC_VALUE: {"answer": "answer"}}
        return result

    result = await runtime.execute(
        {"code": "(await orders__get_orders(customer_id='a'))['answer']"},
        dispatch,
        parent_call_id="call-7",
    )

    assert not result.is_error, _text(result)
    assert _text(result).endswith("Result:\nanswer")
    assert [outer_tool_call_id(call_id) for call_id in seen_ids] == ["call-7"]
    assert not is_programmatic_tool_call(seen_ids[0])


@pytest.mark.asyncio
async def test_tool_surface_changes_restart_the_session_with_a_notice(runtime) -> None:
    dispatch = RecordingDispatcher()
    seeded = await runtime.execute({"code": "saved = 1\nsaved"}, dispatch)
    unchanged = await runtime.execute({"code": "saved"}, dispatch)

    runtime.model_tools([*TOOLS, Tool(name="extra", input_schema={"type": "object"})])
    restarted = await runtime.execute({"code": "saved"}, dispatch)

    assert not seeded.is_error and not unchanged.is_error
    assert "restarted" not in _text(unchanged)
    assert restarted.is_error
    assert "sandbox session was restarted" in _text(restarted)


@pytest.mark.asyncio
async def test_attach_media_from_code_stages_media_and_returns_a_handle_free_value(
    tmp_path,
) -> None:
    PILImage.new("RGB", (4, 4), (0, 0, 255)).save(tmp_path / "pixel.png")
    environment = LocalEnvironment(logger=get_logger(__name__), working_directory=tmp_path)
    agent = McpAgent(
        AgentConfig(name="ptc-attach", servers=[], shell=True),
        shell_environment=environment,
        context=Context(
            config=Settings(
                ptc=ProgrammaticToolCallingSettings(enabled=True),
                shell_execution=ShellSettings(enable_attach_media="on"),
            )
        ),
    )
    try:
        await agent.list_tools()
        from_code = await agent.call_tool(
            PTC_TOOL_NAME, {"code": "await attach_media(source='pixel.png')"}, "call-1"
        )
        staged_from_code = agent._consume_pending_media_attachments()
        direct = await agent.call_tool("attach_media", {"source": "pixel.png"}, "call-2")

        assert not from_code.is_error, _text(from_code)
        assert '"staged": true' in _text(from_code)
        assert [block.type for block in from_code.content] == ["text"]
        assert [block.type for block in staged_from_code] == ["image"]
        assert FAST_AGENT_PTC_VALUE not in (direct.meta or {})
        assert "Staged pixel.png" in _text(direct)
    finally:
        await agent.shutdown()
        await environment.close()
