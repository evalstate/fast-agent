"""Typed structures for AgentCard data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.agents.agent_types import AgentConfig


class AgentCardData(TypedDict, total=False):
    config: AgentConfig
    type: str
    func: object | None
    source_path: str
    tool_only: bool
    schema_version: str
    message_files: list[Path]
    child_agents: list[str]
    mcp_connect: list[dict[str, str]]
    agents_as_tools_options: dict[str, Any]
    function_tools: list[str] | str | None
    sequence: list[str]
    cumulative: bool
    fan_out: list[str]
    fan_in: str | None
    include_request: bool
    generator: str
    evaluator: str
    min_rating: str
    max_refinements: int
    refinement_instruction: str | None
    router_agents: list[str]
    instruction: str
    plan_type: str
    plan_iterations: int
    worker: str
    k: int
    max_samples: int
    match_strategy: str
    red_flag_max_length: int | None
    agent_class: type | None
    cls: type | None
