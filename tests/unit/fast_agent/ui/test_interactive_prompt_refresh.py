from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.agents.agent_types import AgentType
from fast_agent.cli.runtime.shell_cwd_policy import ShellCwdCreationError, ShellCwdCreationResult
from fast_agent.core.agent_app import AgentRefreshResult
from fast_agent.ui import enhanced_prompt, interactive_prompt, notification_tracker
from fast_agent.ui.interactive_prompt import InteractivePrompt

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp


class _FakeAgent:
    agent_type = AgentType.BASIC


class _FakeAgentApp:
    def __init__(self) -> None:
        self._agents: dict[str, _FakeAgent] = {"vertex-rag": _FakeAgent()}
        self._refreshed = False
        self.noenv_mode = False
        self.missing_shell_cwd_policy_override: str | None = None

    async def refresh_if_needed(self) -> bool:
        if self._refreshed:
            return False
        self._agents["sizer"] = _FakeAgent()
        self._refreshed = True
        return True

    def latest_refresh_result(self) -> AgentRefreshResult:
        return AgentRefreshResult(changed=self._refreshed)

    def visible_agent_names(self, *, force_include: str | None = None) -> list[str]:
        del force_include
        return list(self._agents.keys())

    def visible_agent_types(self, *, force_include: str | None = None) -> dict[str, AgentType]:
        del force_include
        return {name: agent.agent_type for name, agent in self._agents.items()}

    def registered_agent_names(self) -> list[str]:
        return list(self._agents.keys())

    def registered_agents(self) -> dict[str, _FakeAgent]:
        return self._agents

    def resolve_target_agent_name(self, agent_name: str | None = None) -> str | None:
        return agent_name if agent_name is not None else next(iter(self._agents), None)

    def can_load_agent_cards(self) -> bool:
        return False

    def can_reload_agents(self) -> bool:
        return False


class _FakeAgentAppRemove:
    def __init__(self) -> None:
        self._agents: dict[str, _FakeAgent] = {
            "vertex-rag": _FakeAgent(),
            "sizer": _FakeAgent(),
        }
        self._refreshed = False
        self.noenv_mode = False
        self.missing_shell_cwd_policy_override: str | None = None

    async def refresh_if_needed(self) -> bool:
        if self._refreshed:
            return False
        self._agents.pop("sizer", None)
        self._refreshed = True
        return True

    def latest_refresh_result(self) -> AgentRefreshResult:
        return AgentRefreshResult(changed=self._refreshed)

    def visible_agent_names(self, *, force_include: str | None = None) -> list[str]:
        del force_include
        return list(self._agents.keys())

    def visible_agent_types(self, *, force_include: str | None = None) -> dict[str, AgentType]:
        del force_include
        return {name: agent.agent_type for name, agent in self._agents.items()}

    def registered_agent_names(self) -> list[str]:
        return list(self._agents.keys())

    def registered_agents(self) -> dict[str, _FakeAgent]:
        return self._agents

    def resolve_target_agent_name(self, agent_name: str | None = None) -> str | None:
        return agent_name if agent_name is not None else next(iter(self._agents), None)

    def can_load_agent_cards(self) -> bool:
        return False

    def can_reload_agents(self) -> bool:
        return False


class _FakeToolOnlyAgentApp:
    def __init__(self) -> None:
        self._agents: dict[str, _FakeAgent] = {
            "tool-only": _FakeAgent(),
            "vertex-rag": _FakeAgent(),
        }
        self._tool_only = {"tool-only"}
        self._refreshed = False
        self.noenv_mode = False
        self.missing_shell_cwd_policy_override: str | None = None

    async def refresh_if_needed(self) -> bool:
        if self._refreshed:
            return False
        self._refreshed = True
        return True

    def latest_refresh_result(self) -> AgentRefreshResult:
        return AgentRefreshResult(changed=self._refreshed)

    def visible_agent_names(self, *, force_include: str | None = None) -> list[str]:
        names = [name for name in self._agents if name not in self._tool_only]
        if force_include and force_include in self._agents and force_include not in names:
            return [force_include, *names]
        return names

    def visible_agent_types(self, *, force_include: str | None = None) -> dict[str, AgentType]:
        visible_names = set(self.visible_agent_names(force_include=force_include))
        return {
            name: agent.agent_type for name, agent in self._agents.items() if name in visible_names
        }

    def registered_agent_names(self) -> list[str]:
        return list(self._agents.keys())

    def registered_agents(self) -> dict[str, _FakeAgent]:
        return self._agents

    def resolve_target_agent_name(self, agent_name: str | None = None) -> str | None:
        if agent_name is not None:
            return agent_name
        visible = self.visible_agent_names()
        return visible[0] if visible else next(iter(self._agents), None)

    def can_load_agent_cards(self) -> bool:
        return False

    def can_reload_agents(self) -> bool:
        return False


class _FakeShellCwdAgentApp:
    def registered_agents(self) -> dict[str, object]:
        return {"agent": object()}


class _WarningRefreshAgentApp(_FakeAgentApp):
    async def refresh_if_needed(self) -> bool:
        self._refreshed = True
        return True

    def latest_refresh_result(self) -> AgentRefreshResult:
        return AgentRefreshResult(
            changed=True,
            warnings=["Reload warning [card]"],
        )


class _FailingAgentProvider:
    def _agent(self, name: str) -> object:
        del name
        raise RuntimeError("missing")


@pytest.mark.asyncio
async def test_shell_cwd_startup_prompt_formats_issue_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []

    async def fake_selection(*_args: object, **_kwargs: object) -> str:
        return "n"

    monkeypatch.setattr(
        interactive_prompt,
        "collect_shell_cwd_issues_from_runtime_agents",
        lambda runtime_agents, *, cwd: [object(), object()],
    )
    monkeypatch.setattr(interactive_prompt, "get_selection_input", fake_selection)
    monkeypatch.setattr(interactive_prompt, "rich_print", lambda text: printed.append(text))

    await InteractivePrompt()._maybe_prompt_for_shell_cwd_startup_once(
        runtime_state=interactive_prompt.PromptLoopRuntimeState(),
        prompt_provider=cast("AgentApp", _FakeShellCwdAgentApp()),
        shell_cwd_policy="ask",
    )

    assert printed == ["[yellow]Shell cwd startup check:[/yellow] 2 issues found."]


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (" YES ", True),
        ("Y", True),
        (" nO ", False),
    ],
)
@pytest.mark.asyncio
async def test_shell_cwd_creation_confirmation_normalizes_selection_case(
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    expected: bool,
) -> None:
    async def fake_selection(*_args: object, **_kwargs: object) -> str:
        return selection

    monkeypatch.setattr(interactive_prompt, "get_selection_input", fake_selection)

    assert await InteractivePrompt._confirm_shell_cwd_creation() is expected


def test_get_agent_or_warn_prints_bracketed_agent_name_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    monkeypatch.setattr(interactive_prompt, "rich_print", printed.append)

    agent = InteractivePrompt()._get_agent_or_warn(
        cast("AgentApp", _FailingAgentProvider()),
        "[draft]",
    )

    assert agent is None
    assert [getattr(item, "plain", item) for item in printed] == [
        "Unable to load agent '[draft]'"
    ]


@pytest.mark.asyncio
async def test_refresh_warnings_print_bracketed_text_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    monkeypatch.setattr(interactive_prompt, "rich_print", printed.append)

    state = await InteractivePrompt()._refresh_agents_if_needed(
        prompt_provider=cast("AgentApp", _WarningRefreshAgentApp()),
        state=interactive_prompt.PromptLoopAgents(
            current_agent="vertex-rag",
            available_agents=["vertex-rag"],
            available_agents_set={"vertex-rag"},
        ),
        pinned_agent=None,
    )

    assert state is not None
    assert [getattr(item, "plain", item) for item in printed] == [
        "Reload warning [card]",
        "[green]AgentCards reloaded.[/green]",
    ]


def test_startup_warning_digest_prints_bracketed_warning_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []

    monkeypatch.setattr(
        notification_tracker,
        "pop_startup_warnings",
        lambda: ["server [demo] failed"],
    )
    monkeypatch.setattr(interactive_prompt, "rich_print", printed.append)

    InteractivePrompt()._emit_startup_warning_digest_once(
        runtime_state=interactive_prompt.PromptLoopRuntimeState()
    )

    assert [getattr(item, "plain", item) for item in printed] == [
        "Startup warning:",
        "  server [demo] failed",
    ]


def test_startup_warning_digest_formats_multiple_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []

    monkeypatch.setattr(
        notification_tracker,
        "pop_startup_warnings",
        lambda: ["first", "second"],
    )
    monkeypatch.setattr(interactive_prompt, "rich_print", printed.append)

    InteractivePrompt()._emit_startup_warning_digest_once(
        runtime_state=interactive_prompt.PromptLoopRuntimeState()
    )

    assert [getattr(item, "plain", item) for item in printed] == [
        "Startup warnings (2):",
        "  • first",
        "  • second",
    ]


def test_shell_cwd_creation_result_prints_paths_and_errors_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    monkeypatch.setattr(interactive_prompt, "rich_print", printed.append)

    InteractivePrompt._print_shell_cwd_creation_result(
        ShellCwdCreationResult(
            created_paths=[Path("/tmp/[demo]/work")],
            errors=[
                ShellCwdCreationError(
                    path=Path("/tmp/[bad]/work"),
                    message="mkdir [failed]",
                )
            ],
        )
    )

    assert [getattr(item, "plain", item) for item in printed] == [
        "[green]Created missing shell cwd directories:[/green]",
        "  • /tmp/[demo]/work",
        "[red]Failed to create one or more shell cwd directories:[/red]",
        "  • /tmp/[bad]/work: mkdir [failed]",
    ]


@pytest.mark.asyncio
async def test_prompt_loop_refreshes_agent_list(monkeypatch, capsys: Any) -> None:
    inputs = iter(["STOP"])

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        return next(inputs)

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp()

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    capsys.readouterr()
    assert "sizer" in enhanced_prompt.available_agents


@pytest.mark.asyncio
async def test_prompt_loop_prunes_removed_agent(monkeypatch, capsys: Any) -> None:
    inputs = iter(["STOP"])

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        return next(inputs)

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentAppRemove()

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag", "sizer"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    capsys.readouterr()
    assert enhanced_prompt.available_agents == {"vertex-rag"}


@pytest.mark.asyncio
async def test_prompt_loop_preserves_pinned_tool_only_agent(monkeypatch, capsys: Any) -> None:
    inputs = iter(["STOP"])

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        return next(inputs)

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _FakeToolOnlyAgentApp()

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="tool-only",
        available_agents=["tool-only", "vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
        pinned_agent="tool-only",
    )

    capsys.readouterr()
    assert enhanced_prompt.available_agents == {"tool-only", "vertex-rag"}
