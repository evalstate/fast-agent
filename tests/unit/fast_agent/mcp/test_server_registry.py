from fast_agent.config import MCPServerSettings
from fast_agent.mcp_server_registry import ServerRegistry


def test_default_server_registry_state_is_per_instance() -> None:
    first = ServerRegistry()
    second = ServerRegistry()

    first.registry["demo"] = MCPServerSettings(name="demo", transport="stdio", command="echo")

    assert second.registry == {}
