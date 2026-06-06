"""Shared AgentCard construction helpers."""

from __future__ import annotations

from a2a.types import AgentCard, AgentInterface, AgentSkill

from fast_agent.agents.llm_agent import DEFAULT_CAPABILITIES


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
