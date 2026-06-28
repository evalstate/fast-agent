from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from fast_agent.config import get_settings, update_global_settings
from fast_agent.session import SessionManager, reset_session_manager, set_session_manager
from tests.support.command_surface import (
    CommandSurfaceAgent,
    CommandSurfaceOwner,
    CommandSurfaceProvider,
    build_acp_handler,
    dispatch_tui_command,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tui_and_acp_share_session_pin_state_effect(tmp_path: Path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    update_global_settings(old_settings.model_copy(update={"environment_dir": str(env_dir)}))
    reset_session_manager()

    try:
        provider = CommandSurfaceProvider({"main": CommandSurfaceAgent(name="main")})
        owner = CommandSurfaceOwner(agent_types=provider.agent_types())

        manager = SessionManager(environment_override=env_dir)
        set_session_manager(manager)
        provider._agent("main").context = SimpleNamespace(session_manager=manager)
        session = manager.create_session("sprint")
        session.set_pinned(False)

        await dispatch_tui_command(
            "/session pin Sprint plan",
            owner=owner,
            prompt_provider=provider,
            session_manager=manager,
        )
        assert manager.current_session is not None
        assert manager.current_session.info.metadata.get("pinned") is True
        assert manager.current_session.info.metadata.get("title") == "Sprint plan"
        assert any(
            "Pinned session:" in message for message in provider._agent("main").display.messages
        )

        manager.current_session.set_pinned(False)

        handler = build_acp_handler(provider)
        response = await handler.execute_command("session", "pin ACP plan")

        assert manager.current_session.info.metadata.get("pinned") is True
        assert manager.current_session.info.metadata.get("title") == "ACP plan"
        assert "Pinned session:" in response

        response = await handler.execute_command("session", "unpin")
        assert manager.current_session.info.metadata.get("pinned") is None
        assert "Unpinned session:" in response
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tui_and_acp_share_history_detail_error_intent() -> None:
    provider = CommandSurfaceProvider({"main": CommandSurfaceAgent(name="main")})
    owner = CommandSurfaceOwner(agent_types=provider.agent_types())

    await dispatch_tui_command("/history detail", owner=owner, prompt_provider=provider)
    emitted = "\n".join(provider._agent("main").display.messages)
    assert "Turn number required for /history detail" in emitted

    handler = build_acp_handler(provider)
    response = await handler.execute_command("history", "detail")

    assert "Turn number required for /history detail" in response
