from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from acp.exceptions import RequestError

from fast_agent.acp.server.agent_acp_server import (
    ACP_AUTH_CONFIG_FILE,
    ACP_AUTH_DOCS_URL,
    ACP_AUTH_METHOD_ID,
    AgentACPServer,
)
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.fastagent import AgentInstance
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    instruction = ""
    acp_commands: dict[str, object] = {}

    def __init__(self) -> None:
        self.config = SimpleNamespace(default=False, model=None)


class _AuthAgent(_Agent):
    class _Llm:
        provider = Provider.OPENAI

    llm = _Llm()


class _App:
    def _agent(self, _name: str):
        return _Agent()

    def agent_names(self):
        return ["main"]

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):
        del namespace, agent_name
        return {}


def _build_server(agent: AgentProtocol | None = None) -> AgentACPServer:
    active_agent = agent or cast("AgentProtocol", _Agent())
    app = _App()
    instance = AgentInstance(
        app=cast("Any", app),
        agents={"main": active_agent},
        registry_version=0,
    )

    async def create_instance() -> AgentInstance:
        return instance

    async def dispose_instance(_instance: AgentInstance) -> None:
        return None

    return AgentACPServer(
        bootstrap_instance=instance,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="test",
        permissions_enabled=False,
    )


@pytest.mark.asyncio
async def test_authenticate_returns_setup_metadata() -> None:
    server = _build_server()

    response = await server.authenticate(method_id=ACP_AUTH_METHOD_ID)

    assert response is not None
    meta = getattr(response, "field_meta", None) or getattr(response, "_meta", None)
    assert isinstance(meta, dict)
    assert meta["configFile"] == ACP_AUTH_CONFIG_FILE
    assert meta["docsUrl"] == ACP_AUTH_DOCS_URL
    assert "fast-agent model setup" in meta["recommendedCommands"]


@pytest.mark.asyncio
async def test_authenticate_rejects_unknown_method_id() -> None:
    server = _build_server()

    with pytest.raises(RequestError) as exc_info:
        await server.authenticate(method_id="unknown-method")

    assert exc_info.value.code == -32602
    assert exc_info.value.data == {
        "methodId": "unknown-method",
        "supported": [ACP_AUTH_METHOD_ID],
    }


def test_build_auth_required_data_includes_provider_hints() -> None:
    server = _build_server(cast("AgentProtocol", _AuthAgent()))

    data = server._build_auth_required_data(
        ProviderKeyError("OpenAI API key not configured", ""),
        agent=cast("AgentProtocol", _AuthAgent()),
    )

    assert data["methodId"] == ACP_AUTH_METHOD_ID
    assert data["configFile"] == ACP_AUTH_CONFIG_FILE
    assert data["docsUrl"] == ACP_AUTH_DOCS_URL
    assert data["envVars"] == ["OPENAI_API_KEY"]
    assert data["provider"] == "OpenAI"
