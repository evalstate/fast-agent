"""Contract tests: run skills follow the active environment's filesystem."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from fast_agent.core.fastagent import FastAgent
from fast_agent.skills.registry import SkillRegistry
from fast_agent.tools.execution_environment import (
    EnvironmentFileEntry,
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionOptions,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellRuntimeInfo,
)
from fast_agent.tools.local_shell_executor import LocalEnvironment

SKILL_MARKDOWN = "---\nname: alpha\ndescription: Alpha skill\n---\nUse alpha.\n"


class ShellOnlyEnvironment:
    """Environment with shell execution but no filesystem."""

    async def open(self) -> None:
        return None

    @property
    def cwd(self) -> str:
        return "/workspace"

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(name="sh", kind="remote", provider="shell-only")

    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        del callbacks
        return ShellExecution(
            result=ShellExecutionResult(stdout="", stderr="", exit_code=0),
            options=ShellExecutionOptions(),
        )

    async def close(self) -> None:
        return None


class PosixFilesystemEnvironment(ShellOnlyEnvironment):
    """In-memory posix environment filesystem simulator."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def resolve_path(self, path: str) -> str:
        if path.startswith("/"):
            return path
        return f"{self.cwd}/{path}"

    async def read_text(self, path: str) -> str:
        return self.files[self.resolve_path(path)]

    async def write_text(self, path: str, content: str) -> None:
        self.files[self.resolve_path(path)] = content

    async def exists(self, path: str) -> bool:
        resolved = self.resolve_path(path)
        return resolved in self.files or any(
            name.startswith(f"{resolved}/") for name in self.files
        )

    async def list_dir(self, path: str) -> list[EnvironmentFileEntry]:
        resolved = self.resolve_path(path).rstrip("/")
        if not await self.exists(resolved):
            raise FileNotFoundError(resolved)
        entries: dict[str, EnvironmentFileEntry] = {}
        for file_path in sorted(self.files):
            if not file_path.startswith(f"{resolved}/"):
                continue
            relative = file_path[len(resolved) + 1 :]
            name = relative.split("/", 1)[0]
            kind = "directory" if "/" in relative else "file"
            entries.setdefault(
                name,
                EnvironmentFileEntry(path=f"{resolved}/{name}", name=name, kind=kind),
            )
        return list(entries.values())

    async def mkdir(self, path: str) -> None:
        return None

    async def remove(self, path: str) -> None:
        self.files.pop(self.resolve_path(path), None)


def _build_fast_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAgent:
    monkeypatch.chdir(tmp_path)
    fast = FastAgent("test-environment-skills", parse_cli_args=False)

    @fast.agent(name="main", model="passthrough", default=True)
    async def main() -> None: ...

    return fast


def _agent_manifests(fast: FastAgent) -> list:
    config = fast.agents["main"]["config"]
    return list(config.skill_manifests or [])


def _host_manifest():
    manifest, error = SkillRegistry.parse_manifest_text(
        SKILL_MARKDOWN, path=Path("/host/skills/alpha/SKILL.md")
    )
    assert error is None and manifest is not None
    return manifest


@pytest.mark.asyncio
async def test_environment_filesystem_skills_replace_host_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = _build_fast_agent(tmp_path, monkeypatch)
    fast._apply_skills_to_agent_configs([_host_manifest()])

    env = PosixFilesystemEnvironment()
    env.files["/workspace/.fast-agent/skills/beta/SKILL.md"] = (
        "---\nname: beta\ndescription: Beta skill\n---\nUse beta.\n"
    )
    await fast._apply_environment_skills(env)

    manifests = _agent_manifests(fast)
    assert [manifest.name for manifest in manifests] == ["beta"]
    assert str(manifests[0].path) == "/workspace/.fast-agent/skills/beta/SKILL.md"


@pytest.mark.asyncio
async def test_prompt_context_skills_are_cleared_when_environment_has_no_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = _build_fast_agent(tmp_path, monkeypatch)
    fast._apply_skills_to_agent_configs([_host_manifest()])
    context = {
        "agentSkills": "host leak: /host/skills/alpha/SKILL.md",
    }

    await fast._apply_environment_skills(PosixFilesystemEnvironment())
    fast._refresh_prompt_context_skills(context)

    assert context["agentSkills"] == ""


@pytest.mark.asyncio
async def test_prompt_context_skills_use_environment_manifest_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = _build_fast_agent(tmp_path, monkeypatch)
    fast._apply_skills_to_agent_configs([_host_manifest()])
    context = {
        "agentSkills": "host leak: /host/skills/alpha/SKILL.md",
    }
    env = PosixFilesystemEnvironment()
    env.files["/workspace/.fast-agent/skills/beta/SKILL.md"] = (
        "---\nname: beta\ndescription: Beta skill\n---\nUse beta.\n"
    )

    await fast._apply_environment_skills(env)
    fast._refresh_prompt_context_skills(context)

    assert "/workspace/.fast-agent/skills/beta/SKILL.md" in context["agentSkills"]
    assert "/host/skills/alpha/SKILL.md" not in context["agentSkills"]


@pytest.mark.asyncio
async def test_shell_only_environment_disables_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = _build_fast_agent(tmp_path, monkeypatch)
    fast._apply_skills_to_agent_configs([_host_manifest()])
    assert _agent_manifests(fast)

    await fast._apply_environment_skills(ShellOnlyEnvironment())

    assert _agent_manifests(fast) == []


@pytest.mark.asyncio
async def test_local_environment_keeps_host_skill_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = _build_fast_agent(tmp_path, monkeypatch)
    host_manifest = _host_manifest()
    fast._apply_skills_to_agent_configs([host_manifest])

    local_env = LocalEnvironment(
        logger=logging.getLogger("test-local"), working_directory=tmp_path
    )
    await fast._apply_environment_skills(local_env)

    assert _agent_manifests(fast) == [host_manifest]
