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
from fast_agent.context import Context
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp import tool_result_spool
from fast_agent.mcp.mcp_aggregator import NamespacedTool
from fast_agent.tools.local_shell_executor import LocalEnvironment
from fast_agent.types import PromptMessageExtended

_BYTE_LIMIT = 200
_PATH_PATTERN = re.compile(r"available during this session at (\S+?\.(?:json|txt))\.")


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
