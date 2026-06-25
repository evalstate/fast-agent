from __future__ import annotations

import pytest
from mcp.types import TextContent

from fast_agent.a2a.config import A2AAgentConfig
from fast_agent.a2a.remote_agent import A2ARemoteAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.types import LlmStopReason, PromptMessageExtended
from tests.integration.a2a.conftest import FAKE_A2A_HELP


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
        assert agent.last_task_state is None
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
async def test_a2a_remote_agent_aggregates_task_artifacts_without_stream_chunks(
    a2a_test_server,
) -> None:
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
    assert chunks == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a2a_remote_agent_aggregates_long_task_artifacts_without_stream_chunks(
    a2a_test_server,
) -> None:
    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote_long_stream", agent_type=AgentType.A2A, use_history=False),
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
                    content=[TextContent(type="text", text="please long stream")],
                )
            ]
        )
    finally:
        await agent.shutdown()

    assert chunks == []
    assert "Starting the remote analysis task." in response.all_text()
    assert "Step 1 — Reading the request and identifying the goal." in response.all_text()
    assert "Remote analysis complete." in response.all_text()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a2a_fake_server_help_lists_available_prompts(a2a_test_server) -> None:
    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote_help", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url=a2a_test_server.base_url, transport="JSONRPC"),
    )
    await agent.initialize()
    try:
        response = await agent.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="help")],
                )
            ]
        )
    finally:
        await agent.shutdown()

    assert response.all_text() == FAKE_A2A_HELP
    assert agent.context_id
    assert agent.current_task_id is None
    assert agent.last_task_state is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a2a_remote_agent_preserves_input_required_task_for_follow_up(
    a2a_test_server,
) -> None:
    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote_input", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url=a2a_test_server.base_url, transport="JSONRPC"),
    )
    await agent.initialize()
    try:
        first = await agent.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="need input")],
                )
            ]
        )
        input_task_id = agent.current_task_id

        second = await agent.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="blue")],
                )
            ]
        )
    finally:
        await agent.shutdown()

    assert first.all_text() == "A2A task TASK_STATE_INPUT_REQUIRED: Please provide the missing value."
    assert first.stop_reason == LlmStopReason.PAUSE
    assert input_task_id
    assert "input received: blue" in second.all_text()
    assert second.stop_reason == LlmStopReason.END_TURN
    assert agent.current_task_id is None
    assert agent.last_task_state == "TASK_STATE_COMPLETED"
    assert a2a_test_server.executor.seen_queries[-2:] == ["need input", "blue"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a2a_fake_server_help_does_not_complete_input_required_task(
    a2a_test_server,
) -> None:
    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote_input_help", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url=a2a_test_server.base_url, transport="JSONRPC"),
    )
    await agent.initialize()
    try:
        await agent.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="need input")],
                )
            ]
        )
        input_task_id = agent.current_task_id
        help_response = await agent.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="help")],
                )
            ]
        )
    finally:
        await agent.shutdown()

    assert input_task_id
    assert agent.current_task_id == input_task_id
    assert agent.last_task_state == "TASK_STATE_INPUT_REQUIRED"
    assert "Fake A2A server commands:" in help_response.all_text()
    assert "Current task is still waiting for input." in help_response.all_text()


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
async def test_a2a_remote_agent_honors_artifact_append_semantics(a2a_test_server) -> None:
    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote_append", agent_type=AgentType.A2A, use_history=False),
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
                    content=[TextContent(type="text", text="artifact append")],
                )
            ]
        )
    finally:
        await agent.shutdown()

    assert response.all_text() == "final\nrepeat\nrepeat"
    assert chunks == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a2a_remote_agent_clone_preserves_remote_config(a2a_test_server) -> None:
    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote_clone", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url=a2a_test_server.base_url, transport="HTTP+JSON"),
    )
    await agent.initialize()
    clone: A2ARemoteAgent | None = None
    try:
        clone = await agent.spawn_detached_instance(name="remote_clone[tool]")
        response = await clone.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="hello clone")],
                )
            ]
        )
    finally:
        if clone is not None:
            await clone.shutdown()
        await agent.shutdown()

    assert clone.a2a_config == agent.a2a_config
    assert "echo: hello clone" in response.all_text()


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
