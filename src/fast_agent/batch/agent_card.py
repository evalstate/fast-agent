"""AgentCard loading and target selection for batch runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.core.exceptions import AgentConfigError

if TYPE_CHECKING:
    from fast_agent.core.fastagent import FastAgent


@dataclass(frozen=True)
class BatchAgentCardSelection:
    target_name: str
    loaded_names: list[str]


def load_batch_agent_card(
    fast: FastAgent,
    source: str,
    requested_agent: str | None,
) -> BatchAgentCardSelection:
    """Load a batch AgentCard source and select the worker agent."""
    try:
        loaded_names = (
            fast.load_agents_from_url(source) if _is_url(source) else fast.load_agents(source)
        )
    except AgentConfigError as exc:
        raise ValueError(str(exc)) from exc

    loaded_names = sorted(loaded_names)
    _reject_human_input_agents(fast, loaded_names)

    if requested_agent is not None:
        if requested_agent not in loaded_names:
            raise ValueError(
                f"Agent '{requested_agent}' was not loaded from AgentCard source {source}."
            )
        if _is_tool_only(fast, requested_agent):
            raise ValueError(
                f"Agent '{requested_agent}' is tool_only and cannot be used as the batch worker."
            )
        return BatchAgentCardSelection(target_name=requested_agent, loaded_names=loaded_names)

    runnable = [name for name in loaded_names if not _is_tool_only(fast, name)]
    if not runnable:
        raise ValueError(f"AgentCard source {source} loaded no runnable agents.")
    if len(runnable) == 1:
        return BatchAgentCardSelection(target_name=runnable[0], loaded_names=loaded_names)

    default_runnable = [name for name in runnable if _is_default(fast, name)]
    if len(default_runnable) == 1:
        return BatchAgentCardSelection(target_name=default_runnable[0], loaded_names=loaded_names)

    candidates = ", ".join(runnable)
    raise ValueError(
        f"AgentCard source {source} loaded multiple runnable agents: {candidates}. "
        "Specify --agent <name>."
    )


def force_loaded_card_history_off(fast: FastAgent, loaded_names: list[str]) -> None:
    """Disable persisted/chat history for loaded card agents before runtime init."""
    for name in loaded_names:
        config = _agent_config(fast, name)
        if config is None:
            continue
        config.use_history = False
        if config.default_request_params is not None:
            config.default_request_params.use_history = False


def override_selected_agent_model(fast: FastAgent, target_name: str, model: str) -> None:
    config = _agent_config(fast, target_name)
    if config is None:
        raise ValueError(f"Agent '{target_name}' is missing AgentConfig")
    config.model = model


def _is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"}


def _agent_config(fast: FastAgent, name: str) -> AgentConfig | None:
    agent_data = fast.agents.get(name)
    if agent_data is None:
        return None
    config = agent_data.get("config")
    return config if isinstance(config, AgentConfig) else None


def _is_tool_only(fast: FastAgent, name: str) -> bool:
    agent_data = fast.agents.get(name)
    if agent_data is None:
        return False
    return bool(agent_data.get("tool_only", False))


def _is_default(fast: FastAgent, name: str) -> bool:
    config = _agent_config(fast, name)
    return bool(config.default) if config is not None else False


def _reject_human_input_agents(fast: FastAgent, loaded_names: list[str]) -> None:
    human_input_names = [
        name
        for name in loaded_names
        if (config := _agent_config(fast, name)) is not None and config.human_input
    ]
    if human_input_names:
        names = ", ".join(human_input_names)
        raise ValueError(
            f"AgentCard batch mode does not support human_input agents: {names}."
        )
