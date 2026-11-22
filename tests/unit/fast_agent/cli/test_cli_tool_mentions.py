from __future__ import annotations

import pytest
import typer
from mcp_types import TextContent, Tool

from fast_agent.cli.runtime.agent_setup import (
    _build_cli_message_payload,
    _build_cli_prompt_file_payload,
)
from fast_agent.ptc.mentions import TOOL_DEFINITIONS_META
from fast_agent.tools.tool_surface import (
    ProgrammaticSurface,
    SurfaceTool,
    ToolExposure,
    ToolSurface,
)
from fast_agent.types import PromptMessageExtended


class _Agent:
    def __init__(self, *, programmatic: bool) -> None:
        self.programmatic = programmatic

    async def tool_surface(self) -> ToolSurface:
        if not self.programmatic:
            return ToolSurface(tools=())
        return ToolSurface(
            programmatic=ProgrammaticSurface(
                tool_name="exec_sandbox",
                vocabulary=1,
                tool_format="json",
                instructions="",
                type_check=False,
                max_duration_seconds=30,
                max_memory_mb=64,
            ),
            tools=(
                SurfaceTool(
                    tool=Tool(name="jobs__list", input_schema={}),
                    exposure=ToolExposure.PROGRAMMATIC,
                    context_tokens=0,
                    python_name="jobs_list",
                    stub="def jobs_list(): ...",
                ),
            ),
        )


@pytest.mark.asyncio
async def test_go_message_expands_tool_mentions_for_ptc_agents() -> None:
    payload = await _build_cli_message_payload(
        _Agent(programmatic=True), "List failed jobs ^tool:jobs_list", None
    )

    assert isinstance(payload, PromptMessageExtended)
    assert payload.first_text() == "List failed jobs"
    definitions = payload.content[-1]
    assert isinstance(definitions, TextContent)
    assert (definitions.meta or {}).get(TOOL_DEFINITIONS_META) == ["jobs_list"]


@pytest.mark.asyncio
async def test_go_message_leaves_mentions_as_text_without_ptc() -> None:
    message = "List failed jobs ^tool:jobs_list"
    assert await _build_cli_message_payload(_Agent(programmatic=False), message, None) == message


@pytest.mark.asyncio
async def test_go_prompt_file_expands_user_mentions_and_rejects_unknown_names() -> None:
    prompt = [
        PromptMessageExtended(
            role="user", content=[TextContent(type="text", text="Use ^tool:jobs_list now")]
        )
    ]

    [expanded] = await _build_cli_prompt_file_payload(_Agent(programmatic=True), prompt, None)
    assert expanded.first_text() == "Use `jobs_list` now"
    assert prompt[0].first_text() == "Use ^tool:jobs_list now"

    with pytest.raises(typer.Exit):
        await _build_cli_message_payload(_Agent(programmatic=True), "Use ^tool:typo", None)
