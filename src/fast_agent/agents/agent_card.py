"""Shared AgentCard construction helpers."""

from __future__ import annotations

from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

DEFAULT_CAPABILITIES = AgentCapabilities(streaming=False, push_notifications=False)


def build_fast_agent_card(
    *,
    name: str,
    description: str,
    skills: list[AgentSkill],
) -> AgentCard:
    return AgentCard(
        skills=skills,
        name=name,
        description=description,
        supported_interfaces=[
            AgentInterface(
                url=f"fast-agent://agents/{name}/",
                protocol_binding="fast-agent",
            )
        ],
        version="0.1",
        capabilities=DEFAULT_CAPABILITIES,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        provider=None,
        documentation_url=None,
    )
