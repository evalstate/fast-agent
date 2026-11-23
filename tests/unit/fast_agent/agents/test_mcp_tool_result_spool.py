import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
from mcp_types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings
from fast_agent.context import Context
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp import tool_result_spool
from fast_agent.mcp.mcp_aggregator import NamespacedTool
from fast_agent.ptc import PTC_TOOL_NAME
from fast_agent.tools.local_shell_executor import LocalEnvironment
from fast_agent.types import PromptMessageExtended

_BYTE_LIMIT = 200
_PATH_PATTERN = re.compile(r"available during this session at (\S+?\.(?:json|txt))\.")
_STORE_KEY_PATTERN = re.compile(r"under key '(truncated_output_\d+)'")


def _agent(tmp_path: Path, *, shell: bool, result: CallToolResult) -> McpAgent:
    config = AgentConfig(
        name="test", instruction="Instruction", servers=[], shell=shell, cwd=tmp_path
    )
    environment = (
        LocalEnvironment(logger=get_logger(__name__), working_directory=tmp_path) if shell else None
    )
    agent = McpAgent(config=config, context=Context(), shell_environment=environment)
    tool = Tool(name="lookup", input_schema={"type": "object"})
    namespaced = NamespacedTool(tool=tool, server_name="demo", namespaced_tool_name="demo__lookup")
    agent._aggregator._namespaced_tool_map = {namespaced.namespaced_tool_name: namespaced}
    agent._aggregator._server_to_tool_map = {namespaced.server_name: [namespaced]}

    async def fake_list_tools() -> ListToolsResult:
        return ListToolsResult(
            tools=[tool.model_copy(update={"name": namespaced.namespaced_tool_name})]
        )

    async def fake_call_tool(name: str, *args: object, **kwargs: object) -> CallToolResult:
        del name, args, kwargs
        return result

    async def fake_get_app_integration_config(server_name: str) -> None:
        del server_name

    agent._aggregator.list_tools = cast("Any", fake_list_tools)
    agent._aggregator.call_tool = cast("Any", fake_call_tool)
    agent._aggregator.get_app_integration_config = cast("Any", fake_get_app_integration_config)
    agent._model_tool_output_byte_limit = cast("Any", lambda _llm=None: _BYTE_LIMIT)
    return agent


async def _model_text(agent: McpAgent) -> str:
    request = PromptMessageExtended(
        role="assistant",
        content=[],
        tool_calls={
            "call-1": CallToolRequest(
                params=CallToolRequestParams(name="demo__lookup", arguments={})
            )
        },
    )
    response = await agent.run_tools(request)
    assert response.tool_results is not None
    content = response.tool_results["call-1"].content
    assert isinstance(content[0], TextContent)
    return content[0].text


def _structured_result() -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text="summary " * 100)],
        structured_content={
            "entries": [{"id": index, "title": f"Entry {index}"} for index in range(50)]
        },
    )


@pytest.mark.asyncio
async def test_oversized_structured_result_is_retained_as_complete_json(tmp_path: Path) -> None:
    result = _structured_result()
    agent = _agent(tmp_path, shell=True, result=result)

    text = await _model_text(agent)

    assert "[Tool result truncated:" in text
    match = _PATH_PATTERN.search(text)
    assert match is not None
    retained = Path(match.group(1)).read_text(encoding="utf-8")
    assert json.loads(retained) == result.structured_content
    assert len(retained.splitlines()) > 50

    await agent.shutdown()
    assert not Path(match.group(1)).exists()


@pytest.mark.asyncio
async def test_structured_result_over_retention_limit_falls_back_to_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool_result_spool, "TOOL_RESULT_ARTIFACT_MAX_BYTES", 1000)
    result = _structured_result()
    agent = _agent(tmp_path, shell=True, result=result)

    text = await _model_text(agent)

    match = _PATH_PATTERN.search(text)
    assert match is not None
    assert match.group(1).endswith(".txt")
    assert "text version of the demo__lookup result" in text
    assert Path(match.group(1)).read_text(encoding="utf-8") == "summary " * 100

    await agent.shutdown()


@pytest.mark.asyncio
async def test_result_is_not_retained_without_a_model_readable_environment(tmp_path: Path) -> None:
    agent = _agent(tmp_path, shell=False, result=_structured_result())

    text = await _model_text(agent)

    assert "[Tool result truncated:" in text
    assert "Use a narrower query" in text
    assert _PATH_PATTERN.search(text) is None

    await agent.shutdown()


@pytest.mark.asyncio
async def test_result_within_budget_is_not_retained(tmp_path: Path) -> None:
    result = CallToolResult(content=[TextContent(type="text", text="small")])
    agent = _agent(tmp_path, shell=True, result=result)

    assert await _model_text(agent) == "small"
    store = agent.transient_artifact_store()
    assert store is not None
    assert store._artifacts == []

    await agent.shutdown()


@pytest.mark.asyncio
async def test_oversized_file_read_is_paged_at_a_line_boundary(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_text(
        "".join(f"line {index:03d}\n" for index in range(1, 101)), encoding="utf-8"
    )
    agent = _agent(tmp_path, shell=True, result=_structured_result())

    async def read(arguments: dict[str, object]) -> str:
        request = PromptMessageExtended(
            role="assistant",
            content=[],
            tool_calls={
                "call-1": CallToolRequest(
                    params=CallToolRequestParams(name="read_text_file", arguments=arguments)
                )
            },
        )
        response = await agent.run_tools(request)
        assert response.tool_results is not None
        content = response.tool_results["call-1"].content
        assert isinstance(content[0], TextContent)
        return content[0].text

    first = await read({"path": "data.txt"})
    # 9-byte lines: 22 fit in the 200-byte budget.
    assert first.startswith("line 001\n")
    assert "line 022\n[File read truncated: showing lines 1-22 of 1-100" in first
    assert "Continue with line=23" in first
    assert "Tool result truncated" not in first

    second = await read({"path": "data.txt", "line": 23, "limit": 30})
    assert second.startswith("line 023\n")
    assert "showing lines 23-44 of 23-52" in second

    await agent.shutdown()


def _ptc_agent(
    tmp_path: Path,
    *,
    shell: bool = False,
    output_byte_limit: int = 256,
    direct_tools: list[str] | None = None,
    result: CallToolResult | None = None,
) -> McpAgent:
    config = AgentConfig(
        name="ptc-test", instruction="Instruction", servers=[], shell=shell, cwd=tmp_path
    )
    environment = (
        LocalEnvironment(logger=get_logger(__name__), working_directory=tmp_path) if shell else None
    )
    settings = Settings(
        ptc=ProgrammaticToolCallingSettings(
            enabled=True,
            output_byte_limit=output_byte_limit,
            direct_tools=direct_tools or [],
        )
    )
    agent = McpAgent(config=config, context=Context(config=settings), shell_environment=environment)
    if result is not None:
        tool = Tool(name="lookup", input_schema={"type": "object"})
        namespaced = NamespacedTool(
            tool=tool, server_name="demo", namespaced_tool_name="demo__lookup"
        )
        agent._aggregator._namespaced_tool_map = {namespaced.namespaced_tool_name: namespaced}
        agent._aggregator._server_to_tool_map = {namespaced.server_name: [namespaced]}

        async def fake_list_tools() -> ListToolsResult:
            return ListToolsResult(
                tools=[tool.model_copy(update={"name": namespaced.namespaced_tool_name})]
            )

        async def fake_call_tool(name: str, *args: object, **kwargs: object) -> CallToolResult:
            del name, args, kwargs
            return result

        async def fake_get_app_integration_config(server_name: str) -> None:
            del server_name

        agent._aggregator.list_tools = cast("Any", fake_list_tools)
        agent._aggregator.call_tool = cast("Any", fake_call_tool)
        agent._aggregator.get_app_integration_config = cast("Any", fake_get_app_integration_config)
        agent._model_tool_output_byte_limit = cast("Any", lambda _llm=None: output_byte_limit)
    return agent


async def _run_tool(agent: McpAgent, name: str, arguments: dict[str, Any]) -> CallToolResult:
    request = PromptMessageExtended(
        role="assistant",
        content=[],
        tool_calls={
            "call-1": CallToolRequest(params=CallToolRequestParams(name=name, arguments=arguments))
        },
    )
    response = await agent.run_tools(request)
    assert response.tool_results is not None
    return response.tool_results["call-1"]


@pytest.mark.asyncio
async def test_ptc_without_shell_retains_oversized_output_in_sandbox_store(tmp_path: Path) -> None:
    pytest.importorskip("pydantic_monty")
    agent = _ptc_agent(tmp_path, shell=False, output_byte_limit=256)
    try:
        await agent.list_tools()
        first = await _run_tool(
            agent,
            PTC_TOOL_NAME,
            {"code": ("print('HEAD-' * 40)\nprint('SECRET_MIDDLE_TOKEN_42')\nprint('TAIL-' * 40)")},
        )
        assert not first.is_error
        assert isinstance(first.content[0], TextContent)
        text = first.content[0].text
        assert "[Tool result truncated:" in text
        assert "SECRET_MIDDLE_TOKEN_42" not in text
        match = _STORE_KEY_PATTERN.search(text)
        assert match is not None
        key = match.group(1)

        recovered = await _run_tool(
            agent,
            PTC_TOOL_NAME,
            {
                "code": (
                    f"text = await load({key!r})\n"
                    "print([line for line in text.splitlines() if 'SECRET_MIDDLE' in line])"
                )
            },
        )
        assert not recovered.is_error
        assert isinstance(recovered.content[0], TextContent)
        assert "SECRET_MIDDLE_TOKEN_42" in recovered.content[0].text
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_ptc_without_shell_preserves_is_error_and_retains_oversized_failure(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pydantic_monty")
    agent = _ptc_agent(tmp_path, shell=False, output_byte_limit=256)
    try:
        await agent.list_tools()
        failed = await _run_tool(
            agent,
            PTC_TOOL_NAME,
            {
                "code": (
                    "print('BEFORE-' * 40)\n"
                    "print('HIDDEN_DIAGNOSTIC_99')\n"
                    "print('AFTER-' * 40)\n"
                    "raise RuntimeError('boom')"
                )
            },
        )
        assert failed.is_error
        assert isinstance(failed.content[0], TextContent)
        text = failed.content[0].text
        assert "[Tool result truncated:" in text
        match = _STORE_KEY_PATTERN.search(text)
        assert match is not None

        recovered = await _run_tool(
            agent,
            PTC_TOOL_NAME,
            {
                "code": (
                    f"text = await load({match.group(1)!r})\nprint('HIDDEN_DIAGNOSTIC_99' in text)"
                )
            },
        )
        assert not recovered.is_error
        assert isinstance(recovered.content[0], TextContent)
        assert "True" in recovered.content[0].text
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_ptc_without_shell_retains_direct_structured_tool_result_in_store(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pydantic_monty")
    result = _structured_result()
    agent = _ptc_agent(
        tmp_path,
        shell=False,
        output_byte_limit=256,
        direct_tools=["demo__lookup"],
        result=result,
    )
    try:
        await agent.list_tools()
        direct = await _run_tool(agent, "demo__lookup", {})
        assert not direct.is_error
        assert isinstance(direct.content[0], TextContent)
        text = direct.content[0].text
        assert "[Tool result truncated:" in text
        match = _STORE_KEY_PATTERN.search(text)
        assert match is not None

        recovered = await _run_tool(
            agent,
            PTC_TOOL_NAME,
            {"code": f"data = await load({match.group(1)!r})\nprint(len(data['entries']))"},
        )
        assert not recovered.is_error
        assert isinstance(recovered.content[0], TextContent)
        assert "50" in recovered.content[0].text
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_ptc_without_shell_falls_back_to_generic_guidance_when_store_is_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pydantic_monty")
    monkeypatch.setattr("fast_agent.ptc.store.MAX_STORE_BYTES", 100)
    agent = _ptc_agent(tmp_path, shell=False, output_byte_limit=256)
    try:
        await agent.list_tools()
        res = await _run_tool(agent, PTC_TOOL_NAME, {"code": "print('x' * 500)"})
        assert not res.is_error
        assert isinstance(res.content[0], TextContent)
        text = res.content[0].text
        assert "[Tool result truncated:" in text
        assert "Use a narrower query" in text
        assert _STORE_KEY_PATTERN.search(text) is None
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_ptc_with_shell_prefers_spool_file_over_sandbox_store(tmp_path: Path) -> None:
    pytest.importorskip("pydantic_monty")
    agent = _ptc_agent(tmp_path, shell=True, output_byte_limit=256)
    try:
        await agent.list_tools()
        res = await _run_tool(agent, PTC_TOOL_NAME, {"code": "print('x' * 500)"})
        assert not res.is_error
        assert isinstance(res.content[0], TextContent)
        text = res.content[0].text
        assert _PATH_PATTERN.search(text) is not None
        assert _STORE_KEY_PATTERN.search(text) is None
        assert agent.ptc_store is not None
        assert len(agent.ptc_store) == 0
    finally:
        await agent.shutdown()
