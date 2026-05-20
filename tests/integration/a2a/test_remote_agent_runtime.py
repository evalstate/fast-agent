from __future__ import annotations

import pytest
from mcp.types import TextContent

from fast_agent.a2a.config import A2AAgentConfig
from fast_agent.a2a.remote_agent import A2ARemoteAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.types import PromptMessageExtended


async def _send_text(base_url: str, transport: str) -> A2ARemoteAgent:
    agent = A2ARemoteAgent(
        config=AgentConfig(
            name=f"remote_{transport.lower().replace('+', '_')}",
            agent_type=AgentType.A2A,
            use_history=False,
        ),
        a2a_config=A2AAgentConfig(url=base_url, transport=transport),
    )
    await agent.initialize()
    try:
        response = await agent.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text=f"hello over {transport}")],
                )
            ]
        )
        assert f"echo: hello over {transport}" in response.all_text()
        assert agent.remote_card is not None
        assert agent.remote_card.name == "fast-agent test A2A server"
        assert agent.context_id
        assert agent.last_task_state == "TASK_STATE_COMPLETED"
        assert agent.current_task_id is None
        return agent
    except Exception:
        await agent.shutdown()
        raise


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["JSONRPC", "HTTP+JSON"])
async def test_a2a_remote_agent_sends_text_over_supported_transports(
    a2a_test_server, transport: str
) -> None:
    agent = await _send_text(a2a_test_server.base_url, transport)
    await agent.shutdown()
