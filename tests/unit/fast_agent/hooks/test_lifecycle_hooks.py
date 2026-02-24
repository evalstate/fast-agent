"""Unit tests for lifecycle hook loading and execution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_decorator import LlmDecorator
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.hooks.lifecycle_hook_loader import (
    VALID_LIFECYCLE_HOOK_TYPES,
    load_lifecycle_hooks,
)

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


class _CloseTrackingLLM:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_closes_llm_resources_when_supported() -> None:
    config = AgentConfig("test-agent")
    agent = LlmDecorator(config=config)
    llm = _CloseTrackingLLM()
    agent._llm = cast("Any", llm)

    await agent.shutdown()

    assert llm.closed
