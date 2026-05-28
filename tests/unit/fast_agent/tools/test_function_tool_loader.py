import asyncio
import time

import pytest
from fastmcp.tools import FunctionTool, ToolResult

from fast_agent.agents.agent_types import ScopedFunctionToolConfig
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.tools.function_tool_config import FunctionToolSpec
from fast_agent.tools.function_tool_loader import build_default_function_tool, load_function_tools


@pytest.mark.asyncio
async def test_sync_function_tool_runs_off_event_loop() -> None:
    def blocking_add(a: int, b: int) -> int:
        time.sleep(0.05)
        return a + b

    tool = build_default_function_tool(blocking_add)

    async def probe() -> float:
        started = time.perf_counter()
        await asyncio.sleep(0.01)
        return time.perf_counter() - started

    result, probe_elapsed = await asyncio.gather(
        tool.run({"a": 2, "b": 3}),
        probe(),
    )

    assert get_text(result.content[0]) == "5"
    assert result.structured_content is None
    assert probe_elapsed < 0.03


@pytest.mark.asyncio
async def test_loader_returns_text_only_function_tool_for_dict_result() -> None:
    def shout(value: str) -> dict[str, str]:
        return {"value": value.upper()}

    tool = load_function_tools([shout])[0]

    result = await tool.run({"value": "hello"})

    assert isinstance(tool, FunctionTool)
    assert get_text(result.content[0]) == '{"value":"HELLO"}'
    assert result.structured_content is None


@pytest.mark.asyncio
async def test_loader_uses_scoped_function_tool_metadata() -> None:
    def shout(value: str) -> dict[str, str]:
        return {"value": value.upper()}

    tool = load_function_tools(
        [
            ScopedFunctionToolConfig(
                function=shout,
                name="custom_shout",
                description="Uppercase a value",
            )
        ]
    )[0]

    result = await tool.run({"value": "hello"})

    assert tool.name == "custom_shout"
    assert tool.description == "Uppercase a value"
    assert get_text(result.content[0]) == '{"value":"HELLO"}'
    assert result.structured_content is None


@pytest.mark.asyncio
async def test_async_function_tool_still_runs_inline() -> None:
    async def async_add(a: int, b: int) -> int:
        await asyncio.sleep(0)
        return a + b

    tool = build_default_function_tool(async_add)

    result = await tool.run({"a": 4, "b": 5})

    assert get_text(result.content[0]) == "9"
    assert result.structured_content is None


@pytest.mark.asyncio
async def test_default_function_tool_preserves_explicit_structured_tool_result() -> None:
    def summarize() -> ToolResult:
        return ToolResult(
            content={"status": "ok"},
            structured_content={"status": "ok"},
        )

    tool = build_default_function_tool(summarize)

    result = await tool.run({})

    assert get_text(result.content[0]) == '{"status":"ok"}'
    assert result.structured_content == {"status": "ok"}


def test_loader_applies_metadata_for_structured_function_tool_spec(tmp_path) -> None:
    module_path = tmp_path / "tools.py"
    module_path.write_text(
        "def run_query(code: str, limit: int = 1):\n    return code\n",
        encoding="utf-8",
    )

    tool = load_function_tools(
        [
            FunctionToolSpec(
                entrypoint="tools.py:run_query",
                variant="code",
                code_arg="code",
                language="python",
            )
        ],
        tmp_path,
    )[0]

    assert tool.meta == {
        "variant": "code",
        "code_arg": "code",
        "language": "python",
    }
