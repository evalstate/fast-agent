from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from fast_agent.agents.smart_agent import _run_smart_command_call
from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.skills import SKILLS_DEFAULT


@dataclass
class _AgentConfig:
    model: str | None = None
    tool_only: bool = False
    skills: object = SKILLS_DEFAULT


class _SmartAgentStub:
    def __init__(self, *, settings: Settings) -> None:
        self.name = "main"
        self.config = _AgentConfig()
        self.context = Context(config=settings)


@pytest.mark.asyncio
async def test_run_smart_command_models_doctor_returns_markdown(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = await _run_smart_command_call(
            agent,
            operation="models.doctor",
            argument=None,
        )
    finally:
        os.chdir(previous_cwd)

    assert "# models.doctor" in result
    assert "models doctor" in result


@pytest.mark.asyncio
async def test_run_smart_command_check_rejects_invalid_argument_syntax(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    with pytest.raises(AgentConfigError, match="Invalid check arguments"):
        await _run_smart_command_call(
            agent,
            operation="check",
            argument='"',
        )


@pytest.mark.asyncio
async def test_run_smart_command_check_returns_markdown_heading(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = await _run_smart_command_call(
            agent,
            operation="check",
            argument=None,
        )
    finally:
        os.chdir(previous_cwd)

    assert "# check" in result
