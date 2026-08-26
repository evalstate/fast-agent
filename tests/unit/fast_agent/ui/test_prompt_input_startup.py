from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest
from rich.markup import render
from rich.text import Text

from fast_agent.session.session_manager import SessionManager, display_session_name
from fast_agent.tools.execution_environment import ShellRuntimeInfo
from fast_agent.ui.prompt import input as prompt_input
from fast_agent.ui.prompt import input_startup

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from fast_agent.core.agent_app import AgentApp


@dataclass(frozen=True, slots=True)
class _ProcessStatus:
    state: str


@dataclass(frozen=True, slots=True)
class _ProcessSpec:
    process_id: str
    origin_session_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ProcessSnapshot:
    status: _ProcessStatus
    spec: _ProcessSpec
    session_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ManagedProcessSnapshot:
    process_id: str


class _StartupProcessRuntime:
    def __init__(self, snapshots: tuple[_ProcessSnapshot, ...], cwd: Path) -> None:
        self._snapshots = snapshots
        self._cwd = cwd

    async def discover_durable_processes(self) -> tuple[_ProcessSnapshot, ...]:
        return self._snapshots

    async def process_snapshots(self) -> tuple[_ManagedProcessSnapshot, ...]:
        return ()

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(name="bash")

    def working_directory(self) -> Path:
        return self._cwd


@pytest.mark.parametrize("supports_clipboard_image_paste", [False, True])
def test_input_help_banner_has_balanced_markup(
    monkeypatch: pytest.MonkeyPatch,
    supports_clipboard_image_paste: bool,
) -> None:
    monkeypatch.setattr(
        input_startup,
        "rich_print",
        lambda markup: render(markup),
    )

    input_startup.show_input_help_banner(
        is_human_input=False,
        supports_clipboard_image_paste=supports_clipboard_image_paste,
    )


@pytest.mark.asyncio
async def test_input_startup_shows_home_summary_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    provider = object()

    monkeypatch.setattr(prompt_input, "help_message_shown", False)
    monkeypatch.setattr(prompt_input, "rich_print", lambda *args, **kwargs: None)
    monkeypatch.setattr(prompt_input, "_show_model_shortcut_hints", lambda **kwargs: None)
    monkeypatch.setattr(
        prompt_input,
        "_show_fast_agent_home_summary",
        lambda agent_provider: calls.append(agent_provider),
    )

    await prompt_input._show_input_startup(
        agent_name="agent",
        default="",
        show_stop_hint=False,
        is_human_input=False,
        shell_context=prompt_input.ShellInputContext(enabled=False),
        shell_agent=None,
        agent_provider=cast("AgentApp", provider),
        supports_clipboard_image_paste=False,
    )

    assert calls == [provider]


@pytest.mark.asyncio
async def test_input_startup_renders_resume_preview_for_human_prompt_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    provider = object()

    def capture_print(*args: object, **kwargs: object) -> None:
        del kwargs
        printed.append(args[0] if args else "")

    monkeypatch.setattr(prompt_input, "help_message_shown", False)
    monkeypatch.setattr(prompt_input, "rich_print", capture_print)
    monkeypatch.setattr(prompt_input, "_show_model_shortcut_hints", lambda **kwargs: None)
    prompt_input._startup_notices.clear()
    prompt_input.queue_startup_markdown_notice(
        "last assistant response",
        title="Last assistant message",
        agent_name="agent",
    )

    await prompt_input._show_input_startup(
        agent_name="agent",
        default="",
        show_stop_hint=False,
        is_human_input=True,
        shell_context=prompt_input.ShellInputContext(enabled=False),
        shell_agent=None,
        agent_provider=cast("AgentApp", provider),
        supports_clipboard_image_paste=False,
    )

    assert "last assistant response" in printed
    assert prompt_input._startup_notices == []


@pytest.mark.asyncio
async def test_shell_startup_shows_available_process_session_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    printed: list[Text] = []
    manager = SessionManager(cwd=tmp_path, home_override=tmp_path / "home")
    session = manager.create_session(metadata={"title": "Process work"})
    linked_session = manager.create_session(metadata={"title": "Linked work"})
    runtime = _StartupProcessRuntime(
        (
            _ProcessSnapshot(
                _ProcessStatus("running"),
                _ProcessSpec("process-running", session.info.name),
                (session.info.name, linked_session.info.name),
            ),
        ),
        tmp_path,
    )

    def capture_print(value: object) -> None:
        if isinstance(value, Text):
            printed.append(value)

    async def display_agents(
        _agents: Iterable[str],
        _provider: AgentApp | None,
    ) -> None:
        return None

    monkeypatch.setattr(input_startup, "rich_print", capture_print)

    await input_startup.show_shell_startup(
        agent_name="agent",
        agent_provider=None,
        shell_context=prompt_input.ShellInputContext(enabled=True, runtime=runtime),
        shell_agent=None,
        session_manager=manager,
        is_human_input=False,
        available_agents=[],
        display_all_agents_with_hierarchy=display_agents,
    )

    output = "\n".join(item.plain for item in printed)
    assert "Resume an associated session to inherit its processes" in output
    assert display_session_name(session.info.name) in output
    assert "Process work" in output
    assert display_session_name(linked_session.info.name) in output
    assert "Linked work" in output


@pytest.mark.asyncio
async def test_input_startup_renders_queued_resume_preview_after_help_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    provider = object()

    def capture_print(*args: object, **kwargs: object) -> None:
        del kwargs
        printed.append(args[0] if args else "")

    monkeypatch.setattr(prompt_input, "help_message_shown", True)
    monkeypatch.setattr(prompt_input, "rich_print", capture_print)
    prompt_input._startup_notices.clear()
    prompt_input.queue_startup_markdown_notice(
        "last assistant response",
        title="Last assistant message",
        agent_name="agent",
    )

    await prompt_input._show_input_startup(
        agent_name="agent",
        default="",
        show_stop_hint=False,
        is_human_input=True,
        shell_context=prompt_input.ShellInputContext(enabled=False),
        shell_agent=None,
        agent_provider=cast("AgentApp", provider),
        supports_clipboard_image_paste=False,
    )

    assert "last assistant response" in printed
    assert prompt_input._startup_notices == []
