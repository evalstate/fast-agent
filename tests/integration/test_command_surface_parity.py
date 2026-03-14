from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcp.types import TextContent

from fast_agent.config import get_settings, update_global_settings
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.session import get_session_manager, reset_session_manager
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

        manager = get_session_manager()
        session = manager.create_session("sprint")
        session.set_pinned(False)

        await dispatch_tui_command("/session pin on", owner=owner, prompt_provider=provider)
        assert manager.current_session is not None
        assert manager.current_session.info.metadata.get("pinned") is True
        assert any("Pinned session:" in message for message in provider._agent("main").display.messages)

        manager.current_session.set_pinned(False)

        handler = build_acp_handler(provider)
        response = await handler.execute_command("session", "pin on")

        assert manager.current_session.info.metadata.get("pinned") is True
        assert "Pinned session:" in response
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tui_and_acp_share_mcp_session_use_state_effect() -> None:
    provider = CommandSurfaceProvider(
        {
            "main": CommandSurfaceAgent(
                name="main",
                message_history=[
                    PromptMessageExtended(
                        role="user",
                        content=[TextContent(type="text", text="hello")],
                    )
                ],
            )
        }
    )
    owner = CommandSurfaceOwner(agent_types=provider.agent_types())
    session_client = provider._agent("main").aggregator.experimental_sessions

    await dispatch_tui_command(
        "/mcp session use demo sess-123",
        owner=owner,
        prompt_provider=provider,
    )
    assert session_client.active_session_id == "sess-123"
    assert any(
        "Selected MCP session for demo." in message
        for message in provider._agent("main").display.messages
    )

    session_client.active_session_id = "sess-initial"

    handler = build_acp_handler(provider)
    response = await handler.execute_command("mcp", "session use demo sess-123")

    assert session_client.active_session_id == "sess-123"
    assert "Selected MCP session for demo." in response


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
