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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a2a_remote_agent_emits_stream_chunks(a2a_test_server) -> None:
    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote_stream", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url=a2a_test_server.base_url, transport="JSONRPC"),
    )
    chunks: list[str] = []
    agent.add_stream_listener(lambda chunk: chunks.append(chunk.text))
    await agent.initialize()
    try:
        response = await agent.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="please stream")],
                )
            ]
        )
    finally:
        await agent.shutdown()

    assert "stream chunk one" in response.all_text()
    assert "stream chunk two" in response.all_text()
    assert chunks == ["stream chunk one", "stream chunk two"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a2a_remote_agent_renders_file_url_data_and_raw_parts(a2a_test_server) -> None:
    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote_files", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url=a2a_test_server.base_url, transport="JSONRPC"),
    )
    await agent.initialize()
    try:
        response = await agent.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="respond with files")],
                )
            ]
        )
    finally:
        await agent.shutdown()

    text = response.all_text()
    assert "file response" in text
    assert "[report.pdf](https://example.com/report.pdf) (application/pdf)" in text
    assert '"ok": true' in text
    assert '"count": 2.0' in text
    assert "[note.txt: 3 bytes text/plain]" in text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a2a_remote_agent_sends_url_and_raw_parts(a2a_test_server) -> None:
    from mcp.types import ImageContent, ResourceLink
    from pydantic import AnyUrl

    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote_attachments", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url=a2a_test_server.base_url, transport="JSONRPC"),
    )
    await agent.initialize()
    try:
        response = await agent.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[
                        TextContent(type="text", text="inspect attachment"),
                        ResourceLink(
                            type="resource_link",
                            name="report.pdf",
                            uri=AnyUrl("https://example.com/report.pdf"),
                            mimeType="application/pdf",
                        ),
                        ImageContent(type="image", data="YWJj", mimeType="image/png"),
                    ],
                )
            ]
        )
    finally:
        await agent.shutdown()

    assert "echo: inspect attachment [text,url,raw]" in response.all_text()
    assert a2a_test_server.executor.seen_part_kinds[-1] == ["text", "url", "raw"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a2a_connect_command_adds_runtime_agent(a2a_test_server) -> None:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.ui.command_payloads import A2ACommand
    from fast_agent.ui.interactive.command_dispatch import dispatch_command_payload
    from fast_agent.ui.interactive_prompt import InteractivePrompt

    initial = A2ARemoteAgent(
        config=AgentConfig(name="initial", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url=a2a_test_server.base_url, transport="JSONRPC"),
    )
    await initial.initialize()
    app = AgentApp({"initial": initial})
    owner = InteractivePrompt(agent_types={"initial": AgentType.A2A})
    try:
        result = await dispatch_command_payload(
            owner,
            A2ACommand(
                action="connect",
                argument=f"{a2a_test_server.base_url} --transport http --name connected",
            ),
            prompt_provider=app,
            agent="initial",
            available_agents=["initial"],
            available_agents_set={"initial"},
            merge_pinned_agents=lambda names: names,
        )
        assert result.next_agent == "connected"
        assert result.available_agents_set == {"initial", "connected"}
        connected = app.get_agent("connected")
        assert isinstance(connected, A2ARemoteAgent)
        response = await connected.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="hello connected")],
                )
            ]
        )
        assert "echo: hello connected" in response.all_text()
    finally:
        for remote in app.registered_agents().values():
            if isinstance(remote, A2ARemoteAgent):
                await remote.shutdown()
