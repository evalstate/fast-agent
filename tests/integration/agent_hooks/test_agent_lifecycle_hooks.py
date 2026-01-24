"""Integration tests for lifecycle hooks on agent clones."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest
from mcp import CallToolRequest, Tool
from mcp.types import CallToolRequestParams

from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.llm.request_params import RequestParams
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class ChildToolCallerLlm(PassthroughLLM):
    """LLM that calls a child agent tool then returns a final response."""

    def __init__(self, child_tool_name: str, **kwargs):
        super().__init__(**kwargs)
        self._child_tool_name = child_tool_name

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        if any(message.tool_results for message in multipart_messages):
            return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)

        return Prompt.assistant(
            "calling child agent",
            stop_reason=LlmStopReason.TOOL_USE,
            tool_calls={
                "call_child": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name=self._child_tool_name,
                        arguments={"input": "test from parent"},
                    ),
                )
            },
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_lifecycle_hooks_for_detached_clones(fast_agent, tmp_path: Path) -> None:
    marker_file = tmp_path / "lifecycle_marker.jsonl"

    hook_file = tmp_path / "lifecycle_hooks_test.py"
    child_card = tmp_path / "agent_child_lifecycle.md"
    parent_card = tmp_path / "agent_parent_lifecycle.md"

    hook_file.write_text(
        f'''
import json
from fast_agent.hooks.lifecycle_hook_context import AgentLifecycleContext

async def record_lifecycle(ctx: AgentLifecycleContext) -> None:
    marker_path = {str(marker_file)!r}
    payload = {{
        "agent_name": ctx.agent_name,
        "agent_id": id(ctx.agent),
        "hook_type": ctx.hook_type,
    }}
    with open(marker_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
'''
    )

    child_card.write_text(
        '''---
name: child_lifecycle
model: passthrough
type: agent
instruction: Child agent with lifecycle hooks.
lifecycle_hooks:
  on_start: lifecycle_hooks_test.py:record_lifecycle
  on_shutdown: lifecycle_hooks_test.py:record_lifecycle
---
'''
    )

    parent_card.write_text(
        '''---
name: parent_lifecycle
model: passthrough
type: agent
agents: [child_lifecycle]
instruction: Parent agent that calls child.
---
'''
    )

    try:
        fast_agent.load_agents(str(child_card))
        fast_agent.load_agents(str(parent_card))

        async with fast_agent.run() as agents:
            parent = agents.parent_lifecycle
            parent._llm = ChildToolCallerLlm(child_tool_name="agent__child_lifecycle")

            await asyncio.gather(
                parent.generate("request one"),
                parent.generate("request two"),
            )

        assert marker_file.exists(), "Lifecycle hooks should write marker file."

        events = [
            json.loads(line)
            for line in marker_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        tool_events = [event for event in events if event["agent_name"].endswith("[tool]")]

        agent_ids = {event["agent_id"] for event in tool_events}
        assert len(agent_ids) == 2

        for agent_id in agent_ids:
            hook_types = [
                event["hook_type"]
                for event in tool_events
                if event["agent_id"] == agent_id
            ]
            assert hook_types.count("on_start") == 1
            assert hook_types.count("on_shutdown") == 1
            assert hook_types.index("on_start") < hook_types.index("on_shutdown")
    finally:
        hook_file.unlink(missing_ok=True)
        child_card.unlink(missing_ok=True)
        parent_card.unlink(missing_ok=True)
