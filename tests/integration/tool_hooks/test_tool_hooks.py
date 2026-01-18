"""Integration tests for tool hooks feature."""

import pytest
from mcp import CallToolRequest, Tool
from mcp.types import CallToolRequestParams

from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


class MultiToolCallLlm(PassthroughLLM):
    """LLM that makes 3 tool calls before returning final response."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._call_count = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self._call_count += 1

        if self._call_count <= 3:
            # Return tool call
            return Prompt.assistant(
                f"calling tool {self._call_count}",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    f"call_{self._call_count}": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="dummy_tool", arguments={"input": f"test{self._call_count}"}),
                    )
                },
            )
        else:
            # Final response
            return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trim_tool_history_from_card(fast_agent):
    """Test that agent with trim_tool_history=true trims history after tool loop."""
    fast_agent.load_agents("agent_trim.md")

    async with fast_agent.run() as agents:
        agent = agents.trim_agent

        # Inject custom LLM that makes multiple tool calls
        test_llm = MultiToolCallLlm()
        agent._llm = test_llm

        # Run a turn with multiple tool calls
        result = await agent.generate("test")
        assert result.last_text() == "done"

        # Check that history was trimmed
        # Should have: user message + last tool call + result + final = 4 messages
        history = agent.message_history
        assert len(history) == 4, f"Expected 4 messages after trimming, got {len(history)}"

        # Verify structure: user, assistant(tool), user(result), assistant(final)
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert history[1].stop_reason == LlmStopReason.TOOL_USE
        assert history[2].role == "user"
        assert history[3].role == "assistant"
        assert history[3].stop_reason == LlmStopReason.END_TURN


@pytest.mark.integration
@pytest.mark.asyncio
async def test_custom_tool_hooks_from_card(fast_agent):
    """Test that custom tool_hooks from agent card are loaded and called."""
    fast_agent.load_agents("agent_custom_hooks.md")

    async with fast_agent.run() as agents:
        agent = agents.hooks_agent

        # Verify tool_hooks config was loaded
        assert agent.config.tool_hooks is not None
        assert "after_turn_complete" in agent.config.tool_hooks

        # Verify hooks were applied to the agent
        assert agent.tool_runner_hooks is not None
        assert agent.tool_runner_hooks.after_turn_complete is not None

        # Inject custom LLM that makes tool calls
        test_llm = MultiToolCallLlm()
        agent._llm = test_llm

        # Run a turn - this exercises the hook (verified by no errors)
        result = await agent.generate("test")
        assert result.last_text() == "done"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_card_with_trim_tool_history_parses(fast_agent):
    """Test that agent card with trim_tool_history loads correctly."""
    fast_agent.load_agents("agent_trim.md")

    async with fast_agent.run() as agents:
        agent = agents.trim_agent

        # Verify the config has trim_tool_history set
        assert agent.config.trim_tool_history is True

        # Verify hooks are applied
        assert agent.tool_runner_hooks is not None
        assert agent.tool_runner_hooks.after_turn_complete is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_card_with_tool_hooks_parses(fast_agent):
    """Test that agent card with tool_hooks loads correctly."""
    fast_agent.load_agents("agent_custom_hooks.md")

    async with fast_agent.run() as agents:
        agent = agents.hooks_agent

        # Verify the config has tool_hooks set
        assert agent.config.tool_hooks is not None
        assert "after_turn_complete" in agent.config.tool_hooks

        # Verify hooks are applied
        assert agent.tool_runner_hooks is not None
        assert agent.tool_runner_hooks.after_turn_complete is not None


class ChildAgentCallerLlm(PassthroughLLM):
    """LLM that calls a child agent tool then returns final response."""

    def __init__(self, child_tool_name: str, **kwargs):
        super().__init__(**kwargs)
        self._child_tool_name = child_tool_name
        self._call_count = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self._call_count += 1

        if self._call_count == 1:
            # Call the child agent tool
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
        else:
            # Final response
            return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_child_agent_hooks_fire_via_parent(fast_agent, tmp_path):
    """Test that tool_hooks fire when child agent is invoked via parent."""
    import json
    from pathlib import Path

    # Create a marker file that the hook will write to
    marker_file = tmp_path / "hook_called.json"

    # Create a custom hook file that writes to our marker
    hooks_dir = Path(__file__).parent
    child_hooks_file = hooks_dir / "child_hooks_test.py"
    child_hooks_file.write_text(f'''
"""Test hook that writes to a marker file."""
from fast_agent.hooks import HookContext
import json

async def mark_hook_called(ctx: HookContext) -> None:
    """Write hook call info to marker file."""
    marker_path = "{marker_file}"
    data = {{
        "hook_type": ctx.hook_type,
        "iteration": ctx.iteration,
        "called": True,
    }}
    with open(marker_path, "w") as f:
        json.dump(data, f)
''')

    # Create child agent card that uses our test hook
    child_card = hooks_dir / "agent_child_test.md"
    child_card.write_text('''---
type: agent
name: child_test
model: passthrough
tool_hooks:
  after_turn_complete: child_hooks_test.py:mark_hook_called
instruction: Child agent with test hooks.
---
''')

    # Create parent agent card
    parent_card = hooks_dir / "agent_parent_test.md"
    parent_card.write_text('''---
type: agent
name: parent_test
model: passthrough
agents: [child_test]
instruction: Parent agent that uses child.
---
''')

    try:
        fast_agent.load_agents("agent_child_test.md")
        fast_agent.load_agents("agent_parent_test.md")

        async with fast_agent.run() as agents:
            parent = agents.parent_test
            child = agents.child_test

            # Verify child has hooks configured
            assert child.tool_runner_hooks is not None
            assert child.tool_runner_hooks.after_turn_complete is not None

            # Inject LLM that calls the child agent tool
            parent._llm = ChildAgentCallerLlm(child_tool_name="agent__child_test")

            # Run parent - this should invoke child via tool call
            result = await parent.generate("test")
            assert result.last_text() == "done"

            # Verify child's hook was called by checking marker file
            assert marker_file.exists(), "Child agent's after_turn_complete hook should have fired"
            with open(marker_file) as f:
                hook_data = json.load(f)
            assert hook_data["called"] is True
            assert hook_data["hook_type"] == "after_turn_complete"
    finally:
        # Cleanup temp files
        child_hooks_file.unlink(missing_ok=True)
        child_card.unlink(missing_ok=True)
        parent_card.unlink(missing_ok=True)
