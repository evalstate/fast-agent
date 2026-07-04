from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers.skills import (
    REMOTE_ENVIRONMENT_SKILLS_WARNING,
    handle_list_skills,
)
from fast_agent.config import Settings, SkillsSettings
from fast_agent.tools.execution_environment import ShellRuntimeInfo

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _ShellRuntime:
    kind: str

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(name="bash", kind=self.kind)


@dataclass
class _Agent:
    shell_runtime: _ShellRuntime
    config: AgentConfig


def _write_skill(root: Path, name: str = "alpha") -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} skill\n---\nUse {name}.\n",
        encoding="utf-8",
    )


def _ctx(tmp_path: Path, *, runtime_kind: str) -> CommandContext:
    skills_root = tmp_path / "skills"
    _write_skill(skills_root)
    agent = _Agent(
        shell_runtime=_ShellRuntime(runtime_kind),
        config=AgentConfig(name="main", instruction="Instruction", servers=[]),
    )
    return CommandContext(
        agent_provider=StaticAgentProvider({"main": agent}),
        current_agent_name="main",
        io=NonInteractiveCommandIOBase(),
        settings=Settings(skills=SkillsSettings(directories=[str(skills_root)])),
    )


@pytest.mark.asyncio
async def test_list_skills_warns_when_runtime_environment_is_remote(tmp_path: Path) -> None:
    outcome = await handle_list_skills(_ctx(tmp_path, runtime_kind="remote"), agent_name="main")

    assert outcome.messages[-1].channel == "warning"
    assert outcome.messages[-1].plain_text() == REMOTE_ENVIRONMENT_SKILLS_WARNING


@pytest.mark.asyncio
async def test_list_skills_does_not_warn_when_runtime_environment_is_local(tmp_path: Path) -> None:
    outcome = await handle_list_skills(_ctx(tmp_path, runtime_kind="local"), agent_name="main")

    assert REMOTE_ENVIRONMENT_SKILLS_WARNING not in [
        message.plain_text() for message in outcome.messages
    ]
