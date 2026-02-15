from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.provider.openai.responses_websocket import (
    ManagedWebSocketConnection,
    WebSocketConnectionManager,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.interfaces import LLMFactoryProtocol


def _websocket_test_models() -> list[str]:
    if configured_models := os.environ.get("TEST_RESPONSES_WS_MODELS"):
        return [model.strip() for model in configured_models.split(",") if model.strip()]
    if configured_model := os.environ.get("TEST_RESPONSES_WS_MODEL"):
        value = configured_model.strip()
        return [value] if value else []
    return []


def _ensure_websocket_transport(model_name: str) -> str:
    if "transport=" in model_name:
        return model_name
    separator = "&" if "?" in model_name else "?"
    return f"{model_name}{separator}transport=websocket"


def _resolve_codex_websocket_model(model_name: str) -> str:
    resolved = _ensure_websocket_transport(model_name)
    model_config = ModelFactory.parse_model_string(resolved)
    if model_config.provider != Provider.CODEX_RESPONSES:
        pytest.skip(
            "Responses websocket reuse e2e currently targets codexresponses models only "
            f"(got provider '{model_config.provider.value}')."
        )
    if model_config.transport != "websocket":
        pytest.skip(
            "Responses websocket reuse e2e requires websocket transport in model string "
            f"(got '{model_config.transport}')."
        )
    return resolved


WEBSOCKET_MODELS = _websocket_test_models()
if not WEBSOCKET_MODELS:
    pytest.skip(
        "Set TEST_RESPONSES_WS_MODEL or TEST_RESPONSES_WS_MODELS to run websocket Responses e2e tests",
        allow_module_level=True,
    )


@pytest_asyncio.fixture
async def websocket_agent(model_name: str) -> LlmAgent:
    model_spec = _resolve_codex_websocket_model(model_name)
    config_path = Path(__file__).parent / "fastagent.config.yaml"

    core = Core(settings=config_path)
    await core.initialize()

    agent = LlmAgent(AgentConfig("test"), core.context)
    await agent.attach_llm(ModelFactory.create_factory(model_spec))
    return agent


async def _generate_and_assert_websocket(agent: LlmAgent, prompt: str) -> None:
    result = await agent.generate(
        prompt,
        request_params=RequestParams(maxTokens=200),
    )
    assert result.stop_reason is LlmStopReason.END_TURN
    assert result.last_text()

    assert getattr(agent.llm, "active_transport", None) == "websocket"


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", WEBSOCKET_MODELS)
async def test_responses_websocket_two_turns_without_fallback(
    websocket_agent: LlmAgent,
    model_name: str,
) -> None:
    del model_name
    agent = websocket_agent

    assert getattr(agent.llm, "configured_transport", None) == "websocket"

    await _generate_and_assert_websocket(
        agent,
        "Reply with a short greeting and one emoji.",
    )
    await _generate_and_assert_websocket(
        agent,
        "Now reply with exactly three words.",
    )


class _AlwaysKeepConnectionManager(WebSocketConnectionManager):
    """Test-only manager that keeps reusable sockets after successful turns."""

    async def release(
        self,
        connection: ManagedWebSocketConnection,
        *,
        reusable: bool,
        keep: bool,
    ) -> None:
        del keep
        await super().release(connection, reusable=reusable, keep=True)


class _ReuseProbeCodexResponsesLLM(CodexResponsesLLM):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.connections_created = 0
        self._ws_connections = _AlwaysKeepConnectionManager(idle_timeout_seconds=300.0)

    async def _create_websocket_connection(
        self,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float | None,
    ) -> ManagedWebSocketConnection:
        self.connections_created += 1
        return await super()._create_websocket_connection(url, headers, timeout_seconds)


class _ReuseProbeFactory:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    def __call__(self, agent: Any, **kwargs: Any) -> _ReuseProbeCodexResponsesLLM:
        kwargs.pop("provider", None)
        kwargs.pop("model", None)
        kwargs.pop("transport", None)
        return _ReuseProbeCodexResponsesLLM(
            provider=Provider.CODEX_RESPONSES,
            model=self._model_name,
            agent=agent,
            transport="websocket",
            **kwargs,
        )


@pytest_asyncio.fixture
async def websocket_reuse_probe_agent(model_name: str) -> tuple[LlmAgent, _ReuseProbeCodexResponsesLLM]:
    model_spec = _resolve_codex_websocket_model(model_name)
    model_config = ModelFactory.parse_model_string(model_spec)
    config_path = Path(__file__).parent / "fastagent.config.yaml"

    core = Core(settings=config_path)
    await core.initialize()

    agent = LlmAgent(AgentConfig("test"), core.context)
    reuse_probe_factory: LLMFactoryProtocol = _ReuseProbeFactory(model_config.model_name)
    await agent.attach_llm(reuse_probe_factory)

    probe_llm = agent.llm
    assert isinstance(probe_llm, _ReuseProbeCodexResponsesLLM)
    return agent, probe_llm


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", WEBSOCKET_MODELS)
async def test_responses_websocket_reuse_probe(
    websocket_reuse_probe_agent: tuple[LlmAgent, _ReuseProbeCodexResponsesLLM],
    model_name: str,
) -> None:
    del model_name
    if os.environ.get("TEST_RESPONSES_WS_REUSE_PROBE") != "1":
        pytest.skip(
            "Set TEST_RESPONSES_WS_REUSE_PROBE=1 to run the websocket connection-reuse probe."
        )

    agent, probe_llm = websocket_reuse_probe_agent

    try:
        await _generate_and_assert_websocket(
            agent,
            "Give a one-sentence fact about Saturn.",
        )
        await _generate_and_assert_websocket(
            agent,
            "Now give a different one-sentence fact about Saturn.",
        )
    finally:
        await probe_llm._ws_connections.close()

    if probe_llm.connections_created != 1:
        pytest.xfail(
            "WebSocket connection reuse not observed; expected one socket for two turns "
            f"but created {probe_llm.connections_created}."
        )

    assert probe_llm.connections_created == 1
