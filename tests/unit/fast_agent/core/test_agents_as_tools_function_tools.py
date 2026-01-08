from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.core import Core
from fast_agent.core.direct_factory import create_agents_in_dependency_order
from fast_agent.llm.model_factory import ModelFactory

if TYPE_CHECKING:
    from fast_agent.interfaces import LLMFactoryProtocol, ModelFactoryFunctionProtocol


@pytest.mark.asyncio
async def test_agents_as_tools_loads_function_tools_from_agent_data(
    tmp_path,
) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    tool_path = tmp_path / "tools.py"
    tool_path.write_text(
        "async def mini_rag(text: str) -> str:\n    return text\n", encoding="utf-8"
    )

    source_path = tmp_path / "vertex-rag.md"
    source_path.write_text("", encoding="utf-8")

    parent_config = AgentConfig(
        name="vertex-rag",
        instruction="Use mini_rag.",
    )
    child_config = AgentConfig(
        name="sizer",
        instruction="Return size.",
    )

    agents_dict = {
        "vertex-rag": {
            "config": parent_config,
            "type": AgentType.BASIC.value,
            "child_agents": ["sizer"],
            "function_tools": ["tools.py:mini_rag"],
            "source_path": source_path,
        },
        "sizer": {
            "config": child_config,
            "type": AgentType.BASIC.value,
        },
    }

    def model_factory_func(model: str | None = None) -> LLMFactoryProtocol:
        return ModelFactory.create_factory("passthrough")

    model_factory = cast("ModelFactoryFunctionProtocol", model_factory_func)

    core = Core(settings=str(config_path))
    await core.initialize()
    try:
        agents = await create_agents_in_dependency_order(core, agents_dict, model_factory)
        parent = agents["vertex-rag"]
        tools = await parent.list_tools()
        tool_names = {tool.name for tool in tools.tools}

        assert "mini_rag" in tool_names
        assert "agent__sizer" in tool_names
    finally:
        await core.cleanup()
