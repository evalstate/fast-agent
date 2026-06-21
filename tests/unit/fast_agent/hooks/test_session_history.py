from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, cast

import pytest
from mcp.types import TextContent

from fast_agent.hooks.hook_context import HookContext
from fast_agent.hooks.session_history import save_session_history
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.session.identity import SessionSaveIdentity


class _SavedSessionProxy(Protocol):
    name: str
    message_history: list[PromptMessageExtended]

    def list_attached_mcp_servers(self) -> list[str]: ...

    @property
    def agent_backed_tools(self) -> "Mapping[str, object]": ...


class _Session:
    def __init__(self, session_id: str, metadata: dict[str, object]) -> None:
        self.info = SimpleNamespace(
            name=session_id,
            metadata=metadata,
            last_activity=SimpleNamespace(isoformat=lambda: "2024-01-01T00:00:00"),
        )

    def _save_metadata(self) -> None:
        return None


class _Manager:
    def __init__(self, label: str) -> None:
        self.label = label
        self.workspace_dir = Path.cwd().resolve()
        self.current_session: _Session | None = None
        self.saved_agents: list[object] = []
        self.saved_identities: list[SessionSaveIdentity | None] = []
        self.saved_resolved_prompts: list[dict[str, str] | None] = []

    def get_session(self, name: str) -> object | None:
        del name
        return None

    def set_current_session(self, session: _Session) -> None:
        self.current_session = session

    def create_session_with_id(
        self,
        session_id: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.current_session = _Session(session_id, metadata or {})

    async def save_current_session(
        self,
        agent: object,
        filename: str | None = None,
        *,
        agent_registry=None,
        identity: SessionSaveIdentity | None = None,
        resolved_prompts: dict[str, str] | None = None,
    ) -> str:
        del filename, agent_registry
        self.saved_agents.append(agent)
        self.saved_identities.append(identity)
        self.saved_resolved_prompts.append(resolved_prompts)
        return "history.json"


class _Agent:
    def __init__(
        self,
        *,
        acp_context: object,
        history: list[PromptMessageExtended],
    ) -> None:
        self.name = "main"
        self.config = SimpleNamespace(tool_only=False, model="passthrough")
        self.context = SimpleNamespace(acp=acp_context, session_manager=None)
        self.message_history = history
        self.usage_accumulator = None
        self.agent_registry = None

    def load_message_history(self, messages: list[PromptMessageExtended] | None) -> None:
        self.message_history = messages or []

    def get_agent(self, name: str):
        del name
        return None


@pytest.mark.asyncio
async def test_save_session_history_uses_app_store_for_app_scoped_acp_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_manager = _Manager("workspace")
    workspace_manager.workspace_dir = workspace.resolve()
    app_manager = _Manager("app")
    session_info_updates: list[dict[str, object]] = []

    def fail_get_session_manager(**_kwargs: object) -> object:
        raise AssertionError("session history should use the manager from agent context")

    async def fake_send_session_info_update(**kwargs: object) -> None:
        session_info_updates.append(dict(kwargs))

    monkeypatch.setattr(
        "fast_agent.hooks.session_history.get_current_context",
        lambda: SimpleNamespace(config=SimpleNamespace(session_history=True)),
    )
    monkeypatch.setattr(
        "fast_agent.hooks.session_history.get_session_manager",
        fail_get_session_manager,
    )

    acp_context = SimpleNamespace(
        session_id="s-1",
        session_cwd=str(workspace.resolve()),
        session_store_scope="app",
        session_store_cwd=None,
        resolved_instructions_snapshot=lambda: {"main": "Resolved ACP prompt"},
        send_session_info_update=fake_send_session_info_update,
    )
    history = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="hello")],
        )
    ]
    agent = _Agent(
        acp_context=acp_context,
        history=history,
    )
    agent.context.session_manager = app_manager
    ctx = HookContext(
        runner=SimpleNamespace(iteration=1, request_params=None),
        agent=agent,
        message=agent.message_history[-1],
        hook_type="after_turn_complete",
    )

    await save_session_history(ctx)

    assert workspace_manager.current_session is None
    current_session = app_manager.current_session
    assert current_session is not None
    assert current_session.info.metadata["cwd"] == str(workspace.resolve())
    assert len(app_manager.saved_identities) == 1
    identity = app_manager.saved_identities[0]
    assert identity is not None
    assert identity.session_store_scope == "app"
    assert identity.session_cwd == workspace.resolve()
    assert app_manager.saved_resolved_prompts == [{"main": "Resolved ACP prompt"}]
    saved_agent = cast("_SavedSessionProxy", app_manager.saved_agents[0])
    assert saved_agent.name == "main"
    assert saved_agent.message_history == history
    assert saved_agent.list_attached_mcp_servers() == []
    assert saved_agent.agent_backed_tools == {}
    with pytest.raises(AttributeError):
        object.__getattribute__(saved_agent, "unknown_session_history_proxy_attribute")
    assert session_info_updates == [{"updated_at": "2024-01-01T00:00:00"}]
