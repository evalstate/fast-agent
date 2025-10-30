from fast_agent.mcp.server.agent_server import AgentMCPServer


class _DummyLLM:
    def __init__(self) -> None:
        self.message_history: list = []


class _DummyAgent:
    def __init__(self) -> None:
        self._llm = _DummyLLM()

    async def send(self, message: str) -> str:
        return message


def test_tool_description_supports_agent_placeholder():
    agent_app = type("App", (), {"_agents": {"worker": _DummyAgent()}})()

    server = AgentMCPServer(agent_app=agent_app, tool_description="Use {agent}")

    tool = server.mcp_server._tool_manager._tools["worker_send"]
    assert tool.description == "Use worker"


def test_tool_description_defaults_when_not_provided():
    agent_app = type("App", (), {"_agents": {"writer": _DummyAgent()}})()

    server = AgentMCPServer(agent_app=agent_app, tool_description="Custom text")

    tool = server.mcp_server._tool_manager._tools["writer_send"]
    assert tool.description == "Custom text"
