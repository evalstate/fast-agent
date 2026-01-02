import pytest
from mcp import Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core.prompt import Prompt
from fast_agent.types import PromptMessageExtended, RequestParams


class DummyChild(LlmAgent):
    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config=config, context=None)
        self.spawned: list[DummyChild] = []
        self.history_loads: list[list[PromptMessageExtended] | None] = []
        self.generated: list[list[PromptMessageExtended]] = []

    def load_message_history(self, messages: list[PromptMessageExtended] | None) -> None:
        super().load_message_history(messages)
        self.history_loads.append(messages)

    async def spawn_detached_instance(self, *, name: str | None = None) -> "DummyChild":
        clone_config = AgentConfig(
            name=name or self.config.name,
            instruction=self.instruction,
        )
        clone = DummyChild(clone_config)
        clone.initialized = True
        self.spawned.append(clone)
        return clone

    async def generate_impl(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        self.generated.append(messages)
        return Prompt.assistant("ok")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_agent_tool_uses_stateless_clone_history() -> None:
    parent = ToolAgent(AgentConfig("parent"), [])
    child = DummyChild(AgentConfig("child", instruction="child"))
    child.load_message_history([Prompt.user("seed")])

    tool_name = parent.add_agent_tool(child)
    tool = parent._execution_tools[tool_name]

    assert await tool.run({"text": "hello"}) == "ok"
    assert len(child.spawned) == 1
    clone_one = child.spawned[0]
    assert clone_one.history_loads == [[]]
    assert clone_one.message_history == []

    assert await tool.run({"text": "again"}) == "ok"
    assert len(child.spawned) == 2
    clone_two = child.spawned[1]
    assert clone_two.history_loads == [[]]
    assert clone_two.message_history == []
    assert clone_two is not clone_one
