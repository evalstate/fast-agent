import pytest

from fast_agent.commands.context import StaticAgentProvider


def test_static_agent_provider_exposes_mapping_backed_agents() -> None:
    agents = {"alpha": object(), "beta": object()}
    provider = StaticAgentProvider(agents)

    assert provider._agent("alpha") is agents["alpha"]
    assert list(provider.visible_agent_names()) == ["alpha", "beta"]
    assert list(provider.registered_agent_names()) == ["alpha", "beta"]
    assert provider.registered_agents() == agents


@pytest.mark.asyncio
async def test_static_agent_provider_list_prompts_defaults_to_empty_mapping() -> None:
    provider = StaticAgentProvider()

    assert await provider.list_prompts(namespace=None) == {}
