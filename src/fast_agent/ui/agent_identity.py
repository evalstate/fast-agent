"""Shared agent identity display rules."""


def is_default_agent_name(agent_name: str | None, *, default_agent_name: str | None) -> bool:
    return (
        agent_name is not None
        and default_agent_name is not None
        and agent_name == default_agent_name
    )
