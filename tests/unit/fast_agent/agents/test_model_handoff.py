from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import AsyncMock

import pytest
from mcp_types import CreateMessageRequestParams

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.config import MCPSamplingSettings, MCPServerSettings, OpenAISettings, Settings
from fast_agent.context import Context
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.model_factory import ModelFactory, with_run_model_endpoint
from fast_agent.llm.provider.bedrock.llm_bedrock import BedrockLLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

Change = Literal["switch", "subagent"]
_SOURCE = "responses.gpt-6.2-astra"
_DUMMY_SOURCE_KEY = "dummy-source-key"


@asynccontextmanager
async def _change_model(
    agent: ToolAgent, target: str, action: Change
) -> "AsyncGenerator[ToolAgent]":
    if action == "switch":
        await agent.set_model(target)
        yield agent
    else:
        child = await agent.spawn_isolated_instance(model=target, for_subagent=True)
        try:
            yield child
        finally:
            await child.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["switch", "subagent"])
@pytest.mark.parametrize(
    ("target", "expected_limit"),
    [
        ("codexresponses.gpt-6.2-astra", None),
        ("copilot.claude-fable-5.2?max_tokens=32000", 32_000),
    ],
)
async def test_model_handoff_retains_harness_controls_not_old_request_options(
    target: str, expected_limit: int | None, action: Change
) -> None:
    params = RequestParams(max_tokens=64_000, use_history=False, metadata={"extra_body": {"x": 1}})
    agent = ToolAgent(AgentConfig("agent", use_history=False, default_request_params=params))
    old_llm = await agent.attach_llm(
        ModelFactory.create_factory(_SOURCE),
        request_params=RequestParams(max_iterations=8, streaming_timeout=None, service_tier="fast"),
    )
    try:
        async with _change_model(agent, target, action) as changed:
            assert changed.llm is not None
            actual = changed.llm.get_request_params()
            assert actual.max_tokens == expected_limit
            assert actual.service_tier is None
            assert actual.metadata is None
            assert actual.use_history is False
            assert actual.max_iterations == 8
            assert actual.streaming_timeout is None
        assert old_llm.default_request_params.max_tokens == 64_000
        assert old_llm.default_request_params.service_tier == "fast"
        assert old_llm.default_request_params.metadata == {"extra_body": {"x": 1}}
    finally:
        await agent.shutdown()


def _require_destination_key(api_key: str) -> None:
    # Never compare in an assert: pytest would print a key that is not the dummy.
    if api_key != "dummy-destination-key":
        pytest.fail("handoff did not use the dummy destination credential", pytrace=False)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["switch", "subagent"])
async def test_model_handoff_uses_destination_credentials_and_connection(action: Change) -> None:
    source = "anthropic.claude-sonnet-4-6"
    destination = OpenAISettings(
        api_key="dummy-destination-key", base_url="https://destination.example.invalid/v1"
    )
    # Configured keys outrank the environment: never let the operator's settings leak in.
    context = Context(config=Settings(_env_file=None, responses=destination, openai=destination))
    agent = ToolAgent(
        AgentConfig("agent", model=source, api_key=_DUMMY_SOURCE_KEY), context=context
    )
    await agent.attach_llm(
        ModelFactory.create_factory(source),
        api_key=_DUMMY_SOURCE_KEY,
        base_url="https://source.example.invalid/v1",
        default_headers={"X-Source-Only": "dummy"},
    )
    try:
        async with _change_model(agent, _SOURCE, action) as changed:
            assert changed.config.api_key == _DUMMY_SOURCE_KEY
            assert isinstance(changed.llm, ResponsesLLM)
            async with changed.llm._responses_client() as client:
                _require_destination_key(client.api_key)
                assert client.base_url.host == "destination.example.invalid"
                assert "X-Source-Only" not in client.default_headers
            # A descendant using exactly this destination cannot resurrect the source key.
            async with _change_model(changed, _SOURCE, "subagent") as grandchild:
                assert isinstance(grandchild.llm, ResponsesLLM)
                async with grandchild.llm._responses_client() as client:
                    _require_destination_key(client.api_key)
    finally:
        await agent.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_model_handoff_preserves_connection_when_only_model_changes() -> None:
    agent = ToolAgent(AgentConfig("agent"))
    await agent.attach_llm(
        ModelFactory.create_factory(_SOURCE),
        request_params=RequestParams(max_tokens=64_000, service_tier="fast"),
        api_key=_DUMMY_SOURCE_KEY,
        base_url="https://same.example.invalid/v1",
    )
    try:
        await agent.set_model("responses.gpt-5.6-sol?max_tokens=1024")
        assert isinstance(agent.llm, ResponsesLLM)
        assert agent.llm.get_request_params().max_tokens == 1024
        assert agent.llm.get_request_params().service_tier is None
        async with agent.llm._responses_client() as client:
            assert client.api_key == _DUMMY_SOURCE_KEY
            assert client.base_url.host == "same.example.invalid"
    finally:
        await agent.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["switch", "subagent"])
async def test_model_handoff_still_rejects_explicit_bad_destination_setting(action: Change) -> None:
    agent = ToolAgent(AgentConfig("agent", model=_SOURCE))
    old_llm = await agent.attach_llm(
        ModelFactory.create_factory(_SOURCE), request_params=RequestParams(max_tokens=64_000)
    )
    try:
        with pytest.raises(ModelConfigError, match="does not support max_tokens"):
            async with _change_model(
                agent, "codexresponses.gpt-6.2-astra?max_tokens=64000", action
            ):
                pytest.fail("Explicit unsupported setting accepted")
        assert agent.llm is old_llm
        assert agent.config.model == _SOURCE
        assert agent.llm.get_request_params().max_tokens == 64_000
    finally:
        await agent.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_url_stays_only_with_its_original_connection() -> None:
    agent = ToolAgent(AgentConfig("agent"))
    url = "https://same.example.invalid/v1"
    await agent.attach_llm(
        with_run_model_endpoint(
            ModelFactory.create_factory(_SOURCE), agent_name=agent.name, base_url=url
        ),
        api_key=_DUMMY_SOURCE_KEY,
    )
    try:
        await agent.set_model("responses.gpt-5.6-sol")
        assert agent.llm is not None
        assert agent.llm.run_model_base_url == url
        for model in (None, "responses.gpt-5.6-sol"):
            child = await agent.spawn_isolated_instance(
                name="child", model=model, for_subagent=True
            )
            try:
                assert child.llm is not None
                assert child.llm.run_model_base_url is None
                assert child._active_attachment_api_key() is None
            finally:
                await child.shutdown()

        await agent.set_model("anthropic.claude-sonnet-4-6")
        assert agent.llm is not None
        assert agent.llm.run_model_base_url is None
        await agent.set_model(_SOURCE)
        assert agent.llm is not None
        assert agent.llm.run_model_base_url is None
    finally:
        await agent.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["switch", "subagent"])
async def test_bedrock_model_handoff_does_not_change_aws_connection(action: Change) -> None:
    pytest.importorskip("boto3")
    agent = ToolAgent(AgentConfig("agent"))
    await agent.attach_llm(
        ModelFactory.create_factory("bedrock.anthropic.claude-sonnet-4-6"),
        profile="explicit-test-profile",
        region="us-west-2",
    )
    try:
        async with _change_model(agent, "bedrock.anthropic.claude-haiku-4-5", action) as changed:
            assert isinstance(changed.llm, BedrockLLM)
            assert changed.llm.aws_profile == "explicit-test-profile"
            assert changed.llm.aws_region == "us-west-2"
    finally:
        await agent.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_default_can_be_reselected_without_resetting_request_settings() -> None:
    agent = ToolAgent(AgentConfig("agent"))
    await agent.attach_llm(
        ModelFactory.create_factory("moonshot"), request_params=RequestParams(max_tokens=1024)
    )
    try:
        async with _change_model(agent, "moonshot", "subagent") as child:
            assert agent.llm is not None and child.llm is not None
            assert child.llm.model_name == agent.llm.model_name
            assert child.llm.get_request_params().max_tokens == 1024
    finally:
        await agent.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_existing_mcp_callback_uses_current_attachment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = McpAgent(
        AgentConfig("agent", model=_SOURCE, api_key=_DUMMY_SOURCE_KEY, servers=[], skills=[])
    )
    sample = AsyncMock(return_value=None)
    monkeypatch.setattr("fast_agent.mcp.client_callback_runtime.sample", sample)
    monkeypatch.setattr(
        agent._aggregator,
        "_server_config",
        lambda _name: MCPServerSettings(sampling=MCPSamplingSettings()),
    )
    await agent.attach_llm(ModelFactory.create_factory(_SOURCE), api_key=_DUMMY_SOURCE_KEY)
    try:
        runtime = agent._aggregator._create_callback_runtime("server")
        assert runtime.sampling_callback is not None
        request_context: Any = None
        params = CreateMessageRequestParams(max_tokens=64, messages=[])
        await runtime.sampling_callback(request_context, params)
        assert sample.await_args is not None
        sampling_target = sample.await_args.kwargs["agent_target"]
        assert sampling_target.model.provider == Provider.RESPONSES
        if sampling_target.api_key != _DUMMY_SOURCE_KEY:
            pytest.fail("sampling did not use the attachment key", pytrace=False)

        await agent.set_model("anthropic.claude-sonnet-4-6")
        await runtime.sampling_callback(request_context, params)
        assert sample.await_args is not None
        sampling_target = sample.await_args.kwargs["agent_target"]
        assert sampling_target.model.provider == Provider.ANTHROPIC
        assert sampling_target.api_key is None
        assert sample.await_args.kwargs["api_key"] is None
        assert agent.config.api_key == _DUMMY_SOURCE_KEY
    finally:
        await agent.shutdown()
