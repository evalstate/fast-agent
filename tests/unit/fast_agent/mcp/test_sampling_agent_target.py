"""Automatic MCP sampling uses the agent's attached model together with its key."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import CreateMessageRequestParams

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import MCPSamplingSettings, MCPServerSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory, with_run_model_endpoint
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fast_agent.llm.resolved_model import ResolvedModelSpec

_CONFIGURED = "openai.gpt-4.1"
_ANTHROPIC_KEY = "dummy-anthropic-key"
_RESPONSES_KEY = "dummy-responses-key"


class _SamplingCalls:
    """The model and key each sampling request built its client with."""

    def __init__(self, agent: McpAgent, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[Provider, str | None]] = []

        def create_sampling_llm(
            params: Any, model: ResolvedModelSpec, api_key: str | None, app_context: Any
        ) -> Any:
            self.calls.append((model.provider, api_key))
            raise RuntimeError("no provider call in tests")

        monkeypatch.setattr("fast_agent.mcp.sampling.create_sampling_llm", create_sampling_llm)
        runtime = agent._aggregator._create_callback_runtime("server")
        assert runtime.sampling_callback is not None
        self._callback: Callable[..., Awaitable[object]] = runtime.sampling_callback

    async def sample(self) -> tuple[Provider, str | None]:
        await self._callback(None, CreateMessageRequestParams(max_tokens=8, messages=[]))
        return self.calls[-1]


def _agent(monkeypatch: pytest.MonkeyPatch, sampling: MCPSamplingSettings) -> McpAgent:
    agent = McpAgent(
        AgentConfig("agent", model=_CONFIGURED, servers=[], skills=[]),
        context=Context(config=Settings(_env_file=None)),
    )
    monkeypatch.setattr(
        agent._aggregator, "_server_config", lambda _name: MCPServerSettings(sampling=sampling)
    )
    return agent


@pytest.mark.asyncio
async def test_sampling_pairs_the_attached_model_with_its_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(monkeypatch, MCPSamplingSettings())
    calls = _SamplingCalls(agent, monkeypatch)
    try:
        # Before an attachment, the configured model never inherits a key.
        assert await calls.sample() == (Provider.OPENAI, None)

        # A bare wire name still selects the attachment's provider.
        await agent.attach_llm(
            ModelFactory.create_factory("claude-sonnet-4-6"), api_key=_ANTHROPIC_KEY
        )
        assert agent.config.model == _CONFIGURED
        assert await calls.sample() == (Provider.ANTHROPIC, _ANTHROPIC_KEY)

        # A custom preset is not re-resolved by its name, which selects OpenAI by default.
        await agent.attach_llm(
            ModelFactory.create_factory(
                "gpt-4.1", presets={"gpt-4.1": "anthropic.claude-sonnet-4-6"}
            ),
            api_key=_ANTHROPIC_KEY,
        )
        assert await calls.sample() == (Provider.ANTHROPIC, _ANTHROPIC_KEY)

        await agent.attach_llm(
            ModelFactory.create_factory("responses.gpt-5.6-sol"), api_key=_RESPONSES_KEY
        )
        assert await calls.sample() == (Provider.RESPONSES, _RESPONSES_KEY)

        # Switching provider leaves the previous attachment's key behind.
        await agent.set_model("anthropic.claude-sonnet-4-6")
        assert await calls.sample() == (Provider.ANTHROPIC, None)

        # An attachment-specific endpoint keeps its key to itself.
        await agent.attach_llm(
            with_run_model_endpoint(
                ModelFactory.create_factory("responses.gpt-5.6-sol"),
                agent_name=agent.name,
                base_url="https://run.example.invalid/v1",
            ),
            api_key=_RESPONSES_KEY,
        )
        assert await calls.sample() == (Provider.RESPONSES, None)
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_server_sampling_model_does_not_inherit_the_attachment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(monkeypatch, MCPSamplingSettings(model="responses.gpt-5.6-sol"))
    calls = _SamplingCalls(agent, monkeypatch)
    try:
        await agent.attach_llm(
            ModelFactory.create_factory("anthropic.claude-sonnet-4-6"), api_key=_ANTHROPIC_KEY
        )
        assert await calls.sample() == (Provider.RESPONSES, None)
    finally:
        await agent.shutdown()
