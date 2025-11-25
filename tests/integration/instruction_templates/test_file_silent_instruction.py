from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_file_silent_reaches_llm_request_params(fast_agent):
    """Ensure {{file_silent:...}} is resolved before the LLM sees the system prompt."""
    fast = fast_agent
    file_text = Path("FOO.md").read_text(encoding="utf-8").strip()

    @fast.agent(
        name="file_template_agent",
        instruction="System prompt start. {{file_silent:FOO.md}}",
        model="passthrough",
    )
    async def agent_function():
        async with fast.run() as agent_app:
            agent = agent_app.file_template_agent

            # Agent-facing instruction should have the file contents applied
            assert "{{file_silent:FOO.md}}" not in agent.instruction
            assert file_text in agent.instruction

            # The LLM request params (what the provider sees) should also be resolved
            request_params = agent.llm.get_request_params()
            assert request_params.systemPrompt is not None
            assert "{{file_silent:FOO.md}}" not in request_params.systemPrompt
            assert file_text in request_params.systemPrompt

            # Default params should stay in sync for future calls
            assert file_text in agent.llm.default_request_params.systemPrompt

            response = await agent.send("ping")
            assert "ping" in response

    await agent_function()
