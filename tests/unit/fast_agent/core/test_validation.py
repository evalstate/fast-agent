from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest
import pytest_asyncio

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core import Core
from fast_agent.core.direct_factory import (
    create_agents_in_dependency_order,
    create_basic_agents_in_dependency_order,
)
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.validation import validate_workflow_references
from fast_agent.llm.model_factory import ModelFactory

if TYPE_CHECKING:
    from fast_agent.interfaces import LLMFactoryProtocol, ModelFactoryFunctionProtocol


def _passthrough_model_factory(model: str | None = None) -> "LLMFactoryProtocol":
    return ModelFactory.create_factory("passthrough")


@pytest_asyncio.fixture(autouse=True)
async def cleanup_logging():
    yield
    from fast_agent.core.logging.logger import LoggingConfig
    from fast_agent.core.logging.transport import AsyncEventBus

    await LoggingConfig.shutdown()
    bus = AsyncEventBus._instance
    if bus is not None:
        await bus.stop()
    AsyncEventBus.reset()
    pending = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and getattr(task.get_coro(), "__qualname__", "") == "AsyncEventBus._process_events"
    ]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


class _CustomAgent(LlmAgent):
    pass


@pytest.mark.asyncio
async def test_create_basic_agents_accepts_llm_enum_type(tmp_path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    agents_dict = {
        "decorated_agent": {
            "config": AgentConfig(
                name="decorated_agent",
                instruction="Be helpful.",
                model="passthrough",
            ),
            "type": AgentType.LLM,
            "func": None,
        }
    }

    core = Core(settings=str(config_path))
    await core.initialize()
    try:
        agents = await create_basic_agents_in_dependency_order(
            core.context,
            agents_dict,
            cast("ModelFactoryFunctionProtocol", _passthrough_model_factory),
        )
    finally:
        await core.cleanup()

    assert set(agents) == {"decorated_agent"}


def test_validate_workflow_references_accepts_basic_like_and_custom_children() -> None:
    agents = {
        "llm_child": {
            "config": AgentConfig(name="llm_child", model="passthrough"),
            "type": AgentType.LLM,
            "func": None,
        },
        "custom_agent_class_child": {
            "config": AgentConfig(name="custom_agent_class_child", model="passthrough"),
            "type": AgentType.CUSTOM.value,
            "func": None,
            "agent_class": _CustomAgent,
        },
        "custom_cls_child": {
            "config": AgentConfig(name="custom_cls_child", model="passthrough"),
            "type": AgentType.CUSTOM.value,
            "func": None,
            "cls": _CustomAgent,
        },
        "orchestrator": {
            "config": AgentConfig(name="orchestrator", model="passthrough"),
            "type": AgentType.ORCHESTRATOR.value,
            "func": None,
            "child_agents": [
                "llm_child",
                "custom_agent_class_child",
                "custom_cls_child",
            ],
        },
    }

    validate_workflow_references(agents)


def test_validate_workflow_references_accepts_nested_workflow_children() -> None:
    agents = {
        "worker": {
            "config": AgentConfig(name="worker", model="passthrough"),
            "type": AgentType.BASIC.value,
            "func": None,
        },
        "nested_orchestrator": {
            "config": AgentConfig(name="nested_orchestrator", model="passthrough"),
            "type": AgentType.ORCHESTRATOR.value,
            "func": None,
            "child_agents": ["worker"],
        },
        "iterative_planner_child": {
            "config": AgentConfig(name="iterative_planner_child", model="passthrough"),
            "type": AgentType.ITERATIVE_PLANNER.value,
            "func": None,
            "child_agents": ["worker"],
        },
        "maker_child": {
            "config": AgentConfig(name="maker_child", model="passthrough"),
            "type": AgentType.MAKER.value,
            "func": None,
            "worker": "worker",
        },
        "orchestrator": {
            "config": AgentConfig(name="orchestrator", model="passthrough"),
            "type": AgentType.ORCHESTRATOR.value,
            "func": None,
            "child_agents": [
                "nested_orchestrator",
                "iterative_planner_child",
                "maker_child",
            ],
        },
    }

    validate_workflow_references(agents)


def test_validate_workflow_references_rejects_missing_maker_worker() -> None:
    agents = {
        "maker_child": {
            "config": AgentConfig(name="maker_child", model="passthrough"),
            "type": AgentType.MAKER.value,
            "func": None,
            "worker": "missing_worker",
        }
    }

    with pytest.raises(AgentConfigError, match="non-existent worker"):
        validate_workflow_references(agents)


def test_validate_workflow_references_accepts_parallel_empty_fan_in() -> None:
    agents = {
        "worker": {
            "config": AgentConfig(name="worker", model="passthrough"),
            "type": AgentType.BASIC.value,
            "func": None,
        },
        "parallel": {
            "config": AgentConfig(name="parallel", model="passthrough"),
            "type": AgentType.PARALLEL.value,
            "func": None,
            "fan_out": ["worker"],
            "fan_in": "",
        },
    }

    validate_workflow_references(agents)


@pytest.mark.asyncio
async def test_create_custom_agent_accepts_cls_alias(tmp_path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    agents_dict = {
        "custom_agent": {
            "config": AgentConfig(
                name="custom_agent",
                instruction="Be helpful.",
                model="passthrough",
            ),
            "type": AgentType.CUSTOM.value,
            "func": None,
            "cls": _CustomAgent,
        }
    }

    core = Core(settings=str(config_path))
    await core.initialize()
    try:
        agents = await create_agents_in_dependency_order(
            core,
            agents_dict,
            cast("ModelFactoryFunctionProtocol", _passthrough_model_factory),
        )
    finally:
        await core.cleanup()

    assert isinstance(agents["custom_agent"], _CustomAgent)


@pytest.mark.asyncio
async def test_create_maker_agents_in_dependency_order_respects_worker_dependencies(
    tmp_path,
) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    agents_dict = {
        "outer_maker": {
            "config": AgentConfig(name="outer_maker", instruction="Vote.", model="passthrough"),
            "type": AgentType.MAKER.value,
            "func": None,
            "worker": "inner_maker",
        },
        "inner_maker": {
            "config": AgentConfig(name="inner_maker", instruction="Vote.", model="passthrough"),
            "type": AgentType.MAKER.value,
            "func": None,
            "worker": "worker",
        },
        "worker": {
            "config": AgentConfig(name="worker", instruction="Work.", model="passthrough"),
            "type": AgentType.BASIC.value,
            "func": None,
        },
    }

    core = Core(settings=str(config_path))
    await core.initialize()
    try:
        agents = await create_agents_in_dependency_order(
            core,
            agents_dict,
            cast("ModelFactoryFunctionProtocol", _passthrough_model_factory),
        )
    finally:
        await core.cleanup()

    assert set(agents) == {"outer_maker", "inner_maker", "worker"}


@pytest.mark.asyncio
async def test_create_custom_agent_requires_class_reference(tmp_path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    agents_dict = {
        "custom_agent": {
            "config": AgentConfig(
                name="custom_agent",
                instruction="Be helpful.",
                model="passthrough",
            ),
            "type": AgentType.CUSTOM.value,
            "func": None,
        }
    }

    core = Core(settings=str(config_path))
    await core.initialize()
    try:
        with pytest.raises(AgentConfigError, match="missing class reference"):
            await create_agents_in_dependency_order(
                core,
                agents_dict,
                cast("ModelFactoryFunctionProtocol", _passthrough_model_factory),
            )
    finally:
        await core.cleanup()
