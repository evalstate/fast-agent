from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.acp.command_io import ACPCommandIO
from fast_agent.acp.slash.handlers import session as session_slash_handlers
from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.commands.results import CommandOutcome
from fast_agent.commands.session_summaries import FULL_SESSION_USAGE
from fast_agent.commands.shared_command_intents import parse_session_command_intent
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    acp_commands: dict[str, object] = {}


class _App:
    def _agent(self, _name: str):
        return _Agent()

    def visible_agent_names(self, *, force_include: str | None = None):
        del force_include
        return ["main"]

    def registered_agent_names(self):
        return ["main"]

    def registered_agents(self):
        return {"main": _Agent()}

    def resolve_target_agent_name(self, agent_name: str | None = None):
        return agent_name or "main"

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):
        del namespace, agent_name
        return {}


@pytest.mark.asyncio
async def test_render_session_list_uses_acp_session_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager_calls: list[Path | None] = []

    class _Manager:
        current_session = None

        def list_sessions(self) -> list[object]:
            return []

    def fake_get_session_manager(
        *,
        cwd: Path | None = None,
        environment_override=None,
        respect_env_override: bool = True,
    ) -> object:
        del environment_override, respect_env_override
        manager_calls.append(cwd)
        return _Manager()

    monkeypatch.setattr("fast_agent.session.get_session_manager", fake_get_session_manager)

    handler = SlashCommandHandler(
        session_id="s1",
        instance=instance,
        primary_agent_name="main",
    )
    handler._acp_context = cast(
        "Any",
        SimpleNamespace(
            session_cwd=str(workspace.resolve()),
            session_store_scope="workspace",
            session_store_cwd=None,
        ),
    )

    output = session_slash_handlers.render_session_list(handler)

    assert "# sessions" in output
    assert manager_calls == [workspace.resolve()]


@pytest.mark.asyncio
async def test_render_session_list_uses_app_session_store_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager_calls: list[Path | None] = []

    class _Manager:
        current_session = None

        def list_sessions(self) -> list[object]:
            return []

    def fake_get_session_manager(
        *,
        cwd: Path | None = None,
        environment_override=None,
        respect_env_override: bool = True,
    ) -> object:
        del environment_override, respect_env_override
        manager_calls.append(cwd)
        return _Manager()

    monkeypatch.setattr("fast_agent.session.get_session_manager", fake_get_session_manager)

    handler = SlashCommandHandler(
        session_id="s1",
        instance=instance,
        primary_agent_name="main",
    )
    handler._acp_context = cast(
        "Any",
        SimpleNamespace(
            session_cwd=str(workspace.resolve()),
            session_store_scope="app",
            session_store_cwd=None,
        ),
    )

    output = session_slash_handlers.render_session_list(handler)

    assert "# sessions" in output
    assert manager_calls == [None]


@pytest.mark.asyncio
async def test_handle_session_unknown_action_returns_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )

    class _Manager:
        current_session = None

        def list_sessions(self) -> list[object]:
            return []

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: _Manager())
    handler = SlashCommandHandler(
        session_id="s1",
        instance=instance,
        primary_agent_name="main",
    )

    output = await session_slash_handlers.handle_session(handler, "unknown-subcommand")

    assert "# session" in output
    assert "Unknown /session action: unknown-subcommand" in output
    assert FULL_SESSION_USAGE in output


@pytest.mark.asyncio
async def test_handle_session_blank_arguments_default_to_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    handler = SlashCommandHandler(
        session_id="s1",
        instance=instance,
        primary_agent_name="main",
    )

    class _Manager:
        current_session = None

        def list_sessions(self) -> list[object]:
            return []

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: _Manager())

    output = await session_slash_handlers.handle_session(handler, "   ")

    assert "# sessions" in output


@pytest.mark.asyncio
async def test_handle_session_new_uses_acp_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    calls: list[tuple[str, str | None, dict[str, str] | None]] = []

    class _Manager:
        def delete_session(self, name: str) -> bool:
            calls.append(("delete", name, None))
            return True

        def create_session_with_id(self, session_id: str, metadata: dict | None = None):
            calls.append(("create_with_id", session_id, metadata))
            return SimpleNamespace(
                info=SimpleNamespace(metadata=metadata or {}, name=session_id)
            )

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: _Manager())
    handler = SlashCommandHandler(
        session_id="acp-session-1",
        instance=instance,
        primary_agent_name="main",
    )

    output = await session_slash_handlers.handle_session_new(handler, "Fresh start")

    assert calls == [
        ("delete", "acp-session-1", None),
        ("create_with_id", "acp-session-1", {"title": "Fresh start"}),
    ]
    assert "Created session: Fresh start" in output


@pytest.mark.asyncio
async def test_format_outcome_includes_captured_history_overview() -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    handler = SlashCommandHandler(
        session_id="s1",
        instance=instance,
        primary_agent_name="main",
    )
    io = ACPCommandIO()
    await io.display_history_overview(
        "main",
        [
            PromptMessageExtended(role="user", content=[text_content("hello")]),
            PromptMessageExtended(role="assistant", content=[text_content("hi")]),
        ],
    )

    output = handler._format_outcome_as_markdown(
        CommandOutcome(),
        "# session resume",
        io=io,
    )

    assert "# session resume" in output
    assert "# conversation history" in output
    assert "Messages: 2 (user: 1, assistant: 1)" in output


@pytest.mark.asyncio
async def test_handle_session_export_leaves_agent_unset_for_latest_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    captured: dict[str, object | None] = {}

    async def fake_handle_session_export(
        ctx,
        *,
        target: str | None,
        agent_name: str | None,
        output_path: str | None,
        hf_dataset: str | None,
        hf_dataset_path: str | None,
        privacy_filter: bool = False,
        privacy_filter_path: str | None = None,
        download_privacy_filter: bool = False,
        privacy_filter_device: str | None = None,
        privacy_filter_variant: str | None = None,
        show_redactions: bool = False,
        current_session_id: str | None = None,
        error: str | None = None,
    ) -> CommandOutcome:
        del (
            ctx,
            privacy_filter,
            privacy_filter_path,
            download_privacy_filter,
            privacy_filter_device,
            privacy_filter_variant,
            show_redactions,
        )
        captured["target"] = target
        captured["agent_name"] = agent_name
        captured["output_path"] = output_path
        captured["hf_dataset"] = hf_dataset
        captured["hf_dataset_path"] = hf_dataset_path
        captured["current_session_id"] = current_session_id
        captured["error"] = error
        return CommandOutcome()

    monkeypatch.setattr(
        session_slash_handlers.session_export_handlers,
        "handle_session_export",
        fake_handle_session_export,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class _Manager:
        current_session = SimpleNamespace(info=SimpleNamespace(name="persisted-1"))

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: _Manager())

    handler = SlashCommandHandler(
        session_id="persisted-1",
        instance=instance,
        primary_agent_name="main",
    )
    handler._acp_context = cast(
        "Any",
        SimpleNamespace(
            session_cwd=str(workspace.resolve()),
            session_store_scope="workspace",
            session_store_cwd=None,
        ),
    )
    intent = parse_session_command_intent("export latest")

    await session_slash_handlers.handle_session_export(handler, intent)

    assert captured == {
        "target": "latest",
        "agent_name": None,
        "output_path": None,
        "hf_dataset": None,
        "hf_dataset_path": None,
        "current_session_id": "persisted-1",
        "error": None,
    }


@pytest.mark.asyncio
async def test_handle_session_export_defaults_agent_only_with_current_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    captured: dict[str, object | None] = {}

    async def fake_handle_session_export(
        ctx,
        *,
        target: str | None,
        agent_name: str | None,
        output_path: str | None,
        hf_dataset: str | None,
        hf_dataset_path: str | None,
        privacy_filter: bool = False,
        privacy_filter_path: str | None = None,
        download_privacy_filter: bool = False,
        privacy_filter_device: str | None = None,
        privacy_filter_variant: str | None = None,
        show_redactions: bool = False,
        current_session_id: str | None = None,
        error: str | None = None,
    ) -> CommandOutcome:
        del (
            ctx,
            output_path,
            hf_dataset,
            hf_dataset_path,
            privacy_filter,
            privacy_filter_path,
            download_privacy_filter,
            privacy_filter_device,
            privacy_filter_variant,
            show_redactions,
            error,
        )
        captured["target"] = target
        captured["agent_name"] = agent_name
        captured["current_session_id"] = current_session_id
        return CommandOutcome()

    monkeypatch.setattr(
        session_slash_handlers.session_export_handlers,
        "handle_session_export",
        fake_handle_session_export,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class _Manager:
        current_session = SimpleNamespace(info=SimpleNamespace(name="persisted-1"))

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: _Manager())

    handler = SlashCommandHandler(
        session_id="persisted-1",
        instance=instance,
        primary_agent_name="main",
    )
    handler._acp_context = cast(
        "Any",
        SimpleNamespace(
            session_cwd=str(workspace.resolve()),
            session_store_scope="workspace",
            session_store_cwd=None,
        ),
    )

    await session_slash_handlers.handle_session_export(handler, parse_session_command_intent("export"))

    assert captured == {
        "target": None,
        "agent_name": "main",
        "current_session_id": "persisted-1",
    }


@pytest.mark.asyncio
async def test_handle_session_export_uses_handler_session_when_manager_current_is_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    captured: dict[str, object | None] = {}

    async def fake_handle_session_export(
        ctx,
        *,
        target: str | None,
        agent_name: str | None,
        output_path: str | None,
        hf_dataset: str | None,
        hf_dataset_path: str | None,
        privacy_filter: bool = False,
        privacy_filter_path: str | None = None,
        download_privacy_filter: bool = False,
        privacy_filter_device: str | None = None,
        privacy_filter_variant: str | None = None,
        show_redactions: bool = False,
        current_session_id: str | None = None,
        error: str | None = None,
    ) -> CommandOutcome:
        del (
            ctx,
            output_path,
            hf_dataset,
            hf_dataset_path,
            privacy_filter,
            privacy_filter_path,
            download_privacy_filter,
            privacy_filter_device,
            privacy_filter_variant,
            show_redactions,
            error,
        )
        captured["target"] = target
        captured["agent_name"] = agent_name
        captured["current_session_id"] = current_session_id
        return CommandOutcome()

    monkeypatch.setattr(
        session_slash_handlers.session_export_handlers,
        "handle_session_export",
        fake_handle_session_export,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class _Manager:
        current_session = None

        def get_session(self, name: str):
            if name == "persisted-1":
                return SimpleNamespace(info=SimpleNamespace(name="persisted-1"))
            return None

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: _Manager())

    handler = SlashCommandHandler(
        session_id="persisted-1",
        instance=instance,
        primary_agent_name="main",
    )
    handler._acp_context = cast(
        "Any",
        SimpleNamespace(
            session_cwd=str(workspace.resolve()),
            session_store_scope="workspace",
            session_store_cwd=None,
        ),
    )

    await session_slash_handlers.handle_session_export(
        handler,
        parse_session_command_intent("export"),
    )

    assert captured == {
        "target": None,
        "agent_name": "main",
        "current_session_id": "persisted-1",
    }


@pytest.mark.asyncio
async def test_handle_session_export_rejects_stale_manager_current_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )

    async def fail_handle_session_export(**kwargs) -> CommandOutcome:
        raise AssertionError(f"unexpected export handler call: {kwargs}")

    monkeypatch.setattr(
        session_slash_handlers.session_export_handlers,
        "handle_session_export",
        fail_handle_session_export,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class _Manager:
        current_session = SimpleNamespace(info=SimpleNamespace(name="other-session"))

        def get_session(self, name: str):
            del name
            return None

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: _Manager())

    handler = SlashCommandHandler(
        session_id="new-session",
        instance=instance,
        primary_agent_name="main",
    )
    handler._acp_context = cast(
        "Any",
        SimpleNamespace(
            session_cwd=str(workspace.resolve()),
            session_store_scope="workspace",
            session_store_cwd=None,
        ),
    )

    output = await session_slash_handlers.handle_session_export(
        handler,
        parse_session_command_intent("export"),
    )

    assert "No active session to export." in output
