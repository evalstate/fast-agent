"""Unit tests for lifecycle hook loading and execution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_decorator import LlmDecorator
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.hooks.lifecycle_hook_context import AgentLifecycleContext
from fast_agent.hooks.lifecycle_hook_loader import (
    VALID_LIFECYCLE_HOOK_TYPES,
    load_lifecycle_hooks,
)
from fast_agent.hooks.lifecycle_hook_types import lifecycle_hook_descriptor

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
def test_lifecycle_hooks_invalid_type_raises(tmp_path: Path) -> None:
    hook_file = tmp_path / "hooks.py"
    hook_file.write_text(
        """
async def on_start(ctx):
    pass
"""
    )

    with pytest.raises(AgentConfigError) as exc_info:
        load_lifecycle_hooks({"invalid": f"{hook_file}:on_start"})

    assert "Invalid lifecycle hook types" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifecycle_hooks_loads_and_calls(tmp_path: Path) -> None:
    marker_file = tmp_path / "lifecycle_marker.json"
    hook_file = tmp_path / "hooks.py"
    hook_file.write_text(
        f"""
import json
from fast_agent.hooks.lifecycle_hook_context import AgentLifecycleContext

async def start_hook(ctx: AgentLifecycleContext) -> None:
    marker_path = {str(marker_file)!r}
    payload = {{
        "agent_name": ctx.agent_name,
        "has_context": ctx.has_context,
        "config_name": ctx.config.name,
        "hook_type": ctx.hook_type,
    }}
    with open(marker_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
"""
    )

    config = AgentConfig(
        "test-agent",
        lifecycle_hooks={"on_start": f"{hook_file}:start_hook"},
    )
    agent = LlmDecorator(config=config)

    await agent.initialize()

    payload = json.loads(marker_file.read_text(encoding="utf-8"))
    assert payload["agent_name"] == "test-agent"
    assert payload["has_context"] is False
    assert payload["config_name"] == "test-agent"
    assert payload["hook_type"] == "on_start"


@pytest.mark.unit
def test_valid_lifecycle_hook_types_constant() -> None:
    assert VALID_LIFECYCLE_HOOK_TYPES == {"on_start", "on_shutdown"}


@pytest.mark.unit
def test_lifecycle_hook_descriptors_define_failure_policy() -> None:
    startup = lifecycle_hook_descriptor("on_start")
    shutdown = lifecycle_hook_descriptor("on_shutdown")

    assert startup.progress_kind == "agent_startup"
    assert startup.raises_on_failure is True
    assert shutdown.progress_kind == "agent_shutdown"
    assert shutdown.raises_on_failure is False


@pytest.mark.unit
def test_lifecycle_context_exposes_agent_registry() -> None:
    primary = LlmDecorator(config=AgentConfig("primary"))
    peer = LlmDecorator(config=AgentConfig("peer"))
    registry = {"primary": primary, "peer": peer}
    primary.set_agent_registry(registry)

    ctx = AgentLifecycleContext(
        agent=primary,
        context=None,
        config=primary.config,
        hook_type="on_start",
    )

    assert ctx.agent_registry == registry
    assert ctx.get_agent("peer") is peer


class _CloseTrackingLLM:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _CountingCloseLLM:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_closes_llm_resources_when_supported() -> None:
    config = AgentConfig("test-agent")
    agent = LlmDecorator(config=config)
    llm = _CloseTrackingLLM()
    agent._llm = cast("Any", llm)

    await agent.shutdown()

    assert llm.closed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_is_idempotent() -> None:
    agent = LlmDecorator(config=AgentConfig("test-agent"))
    llm = _CountingCloseLLM()
    agent._llm = cast("Any", llm)

    await agent.shutdown()
    await agent.shutdown()

    assert llm.close_calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_lifecycle_hook_failure_does_not_raise(tmp_path: Path) -> None:
    hook_file = tmp_path / "hooks.py"
    hook_file.write_text(
        """
async def shutdown_hook(ctx):
    raise RuntimeError("shutdown failed")
"""
    )
    config = AgentConfig(
        "test-agent",
        lifecycle_hooks={"on_shutdown": f"{hook_file}:shutdown_hook"},
    )
    agent = LlmDecorator(config=config)
    llm = _CloseTrackingLLM()
    agent._llm = cast("Any", llm)

    await agent.shutdown()

    assert llm.closed is True
