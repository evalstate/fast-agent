"""Contract tests for the freeform (Responses custom tool) input format of execute_python."""

from __future__ import annotations

from typing import Any

import pytest
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings
from fast_agent.context import Context
from fast_agent.ptc import PTC_TOOL_NAME, ProgrammaticToolRuntime
from fast_agent.tools.apply_patch_tool import get_openai_responses_custom_tool_payload
from fast_agent.ui.streaming.segments import StreamSegmentAssembler

pytest.importorskip("pydantic_monty")

ECHO = Tool(
    name="echo",
    description="Echo a value.",
    input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
)


async def _listed_code_tool(model: str, **ptc: Any) -> Tool:
    agent = McpAgent(
        AgentConfig(name="ptc-format", servers=[]),
        context=Context(
            config=Settings(
                default_model=model,
                ptc=ProgrammaticToolCallingSettings(enabled=True, **ptc),
            )
        ),
    )
    tools = (await agent.list_tools()).tools
    return next(tool for tool in tools if tool.name == PTC_TOOL_NAME)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "settings", "freeform"),
    [
        ("gpt-6-astra", {}, True),
        ("gpt-6-luna", {}, True),
        ("gpt-5.6", {}, False),
        ("gpt-5.6", {"tool_format": "freeform"}, True),
        ("gpt-6-astra", {"tool_format": "json"}, False),
    ],
)
async def test_code_tool_format_follows_model_default_and_setting(
    model: str, settings: dict[str, Any], freeform: bool
) -> None:
    tool = await _listed_code_tool(model, **settings)

    payload = get_openai_responses_custom_tool_payload(tool)
    assert (payload is not None) is freeform
    assert ("FREEFORM" in (tool.description or "")) is freeform
    # Providers without custom tools fall back to the JSON schema, so it must stay valid.
    assert tool.input_schema["required"] == ["input" if freeform else "code"]
    if payload is not None:
        assert payload["format"]["syntax"] == "lark"


async def _dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
    del call_id
    return CallToolResult(content=[TextContent(type="text", text=f"{name}:{arguments['value']}")])


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["input", "code"])
async def test_execute_accepts_freeform_and_json_arguments(argument: str) -> None:
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(enabled=True))
    runtime.model_tools([ECHO], tool_format="freeform")
    try:
        result = await runtime.execute({argument: 'await echo(value="hi")'}, _dispatch)
    finally:
        await runtime.close()

    text = result.content[0]
    assert isinstance(text, TextContent)
    assert not result.is_error, text.text
    assert text.text.endswith("Result:\necho:hi")


def test_freeform_code_streams_as_a_python_preview() -> None:
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(enabled=True))
    runtime.model_tools([ECHO], tool_format="freeform")
    assembler = StreamSegmentAssembler(
        base_kind="markdown",
        tool_prefix="->",
        tool_metadata_resolver=lambda name: runtime.tool_meta() if name == PTC_TOOL_NAME else None,
    )

    for chunk in ("import json\n", 'x = await echo(value="a")\n', "x"):
        assembler.handle_tool_event(
            "delta", {"tool_name": PTC_TOOL_NAME, "tool_use_id": "call-1", "chunk": chunk}
        )

    preview = assembler.segments[0].code_preview
    assert preview is not None
    assert preview.language == "python"
    assert preview.code == 'import json\nx = await echo(value="a")\nx'
