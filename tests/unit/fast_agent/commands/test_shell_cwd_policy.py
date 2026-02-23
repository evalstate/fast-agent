from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.cli.runtime.shell_cwd_policy import (
    can_prompt_for_missing_cwd,
    collect_shell_cwd_issues,
    collect_shell_cwd_issues_from_runtime_agents,
    create_missing_shell_cwd_directories,
    effective_missing_shell_cwd_policy,
    resolve_missing_shell_cwd_policy,
)

if TYPE_CHECKING:
    from fast_agent.core.agent_card_types import AgentCardData


def test_collect_shell_cwd_issues_only_for_shell_enabled_agents(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-dir.txt"
    file_path.write_text("x", encoding="utf-8")

    agents: dict[str, "AgentCardData"] = {
        "shell-missing": {
            "config": AgentConfig(
                name="shell-missing",
                instruction="x",
                servers=[],
                shell=True,
                cwd=Path("missing"),
            )
        },
        "shell-file": {
            "config": AgentConfig(
                name="shell-file",
                instruction="x",
                servers=[],
                shell=True,
                cwd=Path("not-a-dir.txt"),
            )
        },
        "no-shell": {
            "config": AgentConfig(
                name="no-shell",
                instruction="x",
                servers=[],
                shell=False,
                cwd=Path("missing-2"),
            )
        },
    }

    issues = collect_shell_cwd_issues(
        agents,
        shell_runtime_requested=False,
        cwd=tmp_path,
    )

    assert len(issues) == 2
    assert [issue.agent_name for issue in issues] == ["shell-file", "shell-missing"]
    assert issues[0].kind == "not_directory"
    assert issues[1].kind == "missing"


def test_collect_shell_cwd_issues_respects_shell_flag_request(tmp_path: Path) -> None:
    agents: dict[str, "AgentCardData"] = {
        "no-shell": {
            "config": AgentConfig(
                name="no-shell",
                instruction="x",
                servers=[],
                shell=False,
                cwd=Path("missing"),
            )
        }
    }

    without_flag = collect_shell_cwd_issues(
        agents,
        shell_runtime_requested=False,
        cwd=tmp_path,
    )
    with_flag = collect_shell_cwd_issues(
        agents,
        shell_runtime_requested=True,
        cwd=tmp_path,
    )

    assert without_flag == []
    assert len(with_flag) == 1
    assert with_flag[0].kind == "missing"


def test_collect_shell_cwd_issues_from_runtime_agents(tmp_path: Path) -> None:
    class RuntimeAgent:
        def __init__(self, shell_runtime_enabled: bool, cwd: Path | None) -> None:
            self.shell_runtime_enabled = shell_runtime_enabled
            self.config = AgentConfig(name="x", instruction="x", servers=[], cwd=cwd)

    file_path = tmp_path / "not-dir.txt"
    file_path.write_text("x", encoding="utf-8")

    agents: dict[str, object] = {
        "enabled-missing": RuntimeAgent(True, Path("missing")),
        "enabled-file": RuntimeAgent(True, Path("not-dir.txt")),
        "disabled": RuntimeAgent(False, Path("missing2")),
    }

    issues = collect_shell_cwd_issues_from_runtime_agents(agents, cwd=tmp_path)

    assert len(issues) == 2
    assert [issue.agent_name for issue in issues] == ["enabled-file", "enabled-missing"]
    assert issues[0].kind == "not_directory"
    assert issues[1].kind == "missing"


def test_create_missing_shell_cwd_directories_creates_unique_paths(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b"
    agents: dict[str, "AgentCardData"] = {
        "one": {
            "config": AgentConfig(
                name="one",
                instruction="x",
                servers=[],
                shell=True,
                cwd=Path("a/b"),
            )
        },
        "two": {
            "config": AgentConfig(
                name="two",
                instruction="x",
                servers=[],
                shell=True,
                cwd=Path("a/b"),
            )
        },
    }

    issues = collect_shell_cwd_issues(agents, shell_runtime_requested=False, cwd=tmp_path)
    created, errors = create_missing_shell_cwd_directories(issues)

    assert errors == []
    assert created == [target]
    assert target.is_dir()


def test_missing_shell_cwd_policy_resolution_prefers_cli_override() -> None:
    assert resolve_missing_shell_cwd_policy(cli_override="error", configured_policy="warn") == "error"
    assert resolve_missing_shell_cwd_policy(cli_override=None, configured_policy="create") == "create"
    assert resolve_missing_shell_cwd_policy(cli_override=None, configured_policy=None) == "warn"


def test_effective_policy_degrades_ask_when_prompt_unavailable() -> None:
    assert effective_missing_shell_cwd_policy("ask", can_prompt=False) == "warn"
    assert effective_missing_shell_cwd_policy("ask", can_prompt=True) == "ask"


def test_can_prompt_for_missing_cwd_requires_interactive_tty() -> None:
    assert (
        can_prompt_for_missing_cwd(
            mode="interactive",
            message=None,
            prompt_file=None,
            stdin_is_tty=True,
            tty_device_available=False,
        )
        is True
    )

    assert (
        can_prompt_for_missing_cwd(
            mode="serve",
            message=None,
            prompt_file=None,
            stdin_is_tty=True,
            tty_device_available=False,
        )
        is False
    )

    assert (
        can_prompt_for_missing_cwd(
            mode="interactive",
            message="run",
            prompt_file=None,
            stdin_is_tty=True,
            tty_device_available=False,
        )
        is False
    )

    assert (
        can_prompt_for_missing_cwd(
            mode="interactive",
            message=None,
            prompt_file=None,
            stdin_is_tty=False,
            tty_device_available=True,
        )
        is True
    )

