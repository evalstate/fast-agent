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
    DispatchResult,
    dispatch_tui_command,
)

if TYPE_CHECKING:
    from pathlib import Path

@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_session_flow_updates_session_state(tmp_path: Path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    update_global_settings(old_settings.model_copy(update={"environment_dir": str(env_dir)}))
    reset_session_manager()

    try:
        provider = CommandSurfaceProvider({"main": CommandSurfaceAgent(name="main")})
        owner = CommandSurfaceOwner(agent_types=provider.agent_types())

        create_result = await dispatch_tui_command(
            "/session new sprint",
            owner=owner,
            prompt_provider=provider,
        )
        pin_result = await dispatch_tui_command(
            "/session pin on",
            owner=owner,
            prompt_provider=provider,
        )
        await dispatch_tui_command(
            "/session list",
            owner=owner,
            prompt_provider=provider,
        )

        assert create_result == DispatchResult(handled=True)
        assert pin_result == DispatchResult(handled=True)

        manager = get_session_manager()
        current_session = manager.current_session
        assert current_session is not None
        assert current_session.info.metadata.get("pinned") is True

        emitted = "\n".join(provider._agent("main").display.messages)
        assert "Created session:" in emitted
        assert current_session.info.name in emitted
        assert "Pinned session:" in emitted
        assert "Sessions:" in emitted
        assert "(pin)" in emitted
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_history_rewind_updates_history_and_prefills_buffer() -> None:
    agent = CommandSurfaceAgent(
        name="main",
        message_history=[
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="first question")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="first answer")],
            ),
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="second question")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="second answer")],
            ),
        ],
    )
    provider = CommandSurfaceProvider({"main": agent})
    owner = CommandSurfaceOwner(agent_types=provider.agent_types())

    result = await dispatch_tui_command(
        "/history rewind 2",
        owner=owner,
        prompt_provider=provider,
    )

    assert result.buffer_prefill == "second question"
    assert [message.role for message in agent.message_history] == ["user", "assistant"]
    assert agent.message_history[0].first_text() == "first question"
    assert agent.message_history[1].first_text() == "first answer"
    assert any("History rewound" in message for message in agent.display.messages)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_hash_agent_sets_message_handoff() -> None:
    provider = CommandSurfaceProvider(
        {
            "main": CommandSurfaceAgent(name="main"),
            "review": CommandSurfaceAgent(name="review"),
        }
    )
    owner = CommandSurfaceOwner(agent_types=provider.agent_types())

    result = await dispatch_tui_command(
        "##review please assess this change",
        owner=owner,
        prompt_provider=provider,
    )

    assert result.handled is True
    assert result.hash_send_target == "review"
    assert result.hash_send_message == "please assess this change"
    assert result.hash_send_quiet is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_attach_command_prefills_buffer_with_file_tokens(tmp_path: Path) -> None:
    attachment = tmp_path / "scan.pdf"
    attachment.write_bytes(b"%PDF-1.4")

    provider = CommandSurfaceProvider({"main": CommandSurfaceAgent(name="main")})
    owner = CommandSurfaceOwner(agent_types=provider.agent_types())

    result = await dispatch_tui_command(
        f"/attach {attachment}",
        owner=owner,
        prompt_provider=provider,
        buffer_prefill="summarize this",
    )

    assert result.buffer_prefill is not None
    assert result.buffer_prefill.startswith("summarize this ^file:")
    assert str(attachment) in result.buffer_prefill


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_attach_clear_removes_only_local_tokens() -> None:
    provider = CommandSurfaceProvider({"main": CommandSurfaceAgent(name="main")})
    owner = CommandSurfaceOwner(agent_types=provider.agent_types())

    result = await dispatch_tui_command(
        "/attach clear",
        owner=owner,
        prompt_provider=provider,
        buffer_prefill="compare ^file:/tmp/a.png with ^demo:file:///tmp/ref keep this",
    )

    assert result.buffer_prefill == "compare with ^demo:file:///tmp/ref keep this"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_attach_command_rejects_directories(tmp_path: Path) -> None:
    provider = CommandSurfaceProvider({"main": CommandSurfaceAgent(name="main")})
    owner = CommandSurfaceOwner(agent_types=provider.agent_types())

    result = await dispatch_tui_command(
        f"/attach {tmp_path}",
        owner=owner,
        prompt_provider=provider,
        buffer_prefill="draft",
    )

    assert result.buffer_prefill == "draft"
