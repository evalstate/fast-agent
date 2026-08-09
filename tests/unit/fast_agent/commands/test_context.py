import gc
import weakref
from dataclasses import dataclass

import pytest

from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)


@dataclass
class _UnhashableProvider(StaticAgentProvider):
    pass


def _context(
    provider: StaticAgentProvider,
    *,
    acp_session_id: str | None = None,
) -> CommandContext:
    return CommandContext(
        agent_provider=provider,
        current_agent_name="main",
        io=NonInteractiveCommandIOBase(),
        acp_session_id=acp_session_id,
    )


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


def test_skill_source_override_persists_for_provider_lifetime() -> None:
    provider = StaticAgentProvider()
    _context(provider).set_active_skill_source("main", "mcp://skills")

    assert _context(provider).active_skill_source("main") == "mcp://skills"
    assert _context(StaticAgentProvider()).active_skill_source("main") is None


def test_skill_source_override_supports_unhashable_provider() -> None:
    provider = _UnhashableProvider()
    context = _context(provider)

    context.set_active_skill_source("main", "mcp://skills")

    assert context.active_skill_source("main") == "mcp://skills"


def test_skill_source_override_does_not_retain_provider() -> None:
    provider = StaticAgentProvider()
    provider_ref = weakref.ref(provider)
    context = _context(provider)
    context.set_active_skill_source("main", "mcp://skills")

    del context, provider
    gc.collect()

    assert provider_ref() is None


def test_skill_source_override_persists_for_acp_session() -> None:
    _context(StaticAgentProvider(), acp_session_id="session-1").set_active_skill_source(
        "main", "mcp://skills"
    )

    assert (
        _context(
            StaticAgentProvider(),
            acp_session_id="session-1",
        ).active_skill_source("main")
        == "mcp://skills"
    )
    assert (
        _context(
            StaticAgentProvider(),
            acp_session_id="session-2",
        ).active_skill_source("main")
        is None
    )
