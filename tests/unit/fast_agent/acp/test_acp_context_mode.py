from __future__ import annotations

from typing import Any

import pytest
from acp.schema import CurrentModeUpdate, SessionMode

from fast_agent.acp.acp_context import ACPContext


class _Connection:
    def __init__(self) -> None:
        self.session_updates: list[tuple[str, Any]] = []

    async def session_update(self, *, session_id: str, update: Any) -> None:
        self.session_updates.append((session_id, update))


@pytest.mark.asyncio
async def test_switch_mode_syncs_server_owned_session_state() -> None:
    connection = _Connection()
    synced_modes: list[str] = []
    context = ACPContext(
        connection=connection,
        session_id="session-1",
        set_current_mode_callback=synced_modes.append,
    )
    context.set_available_modes(
        [
            SessionMode(id="main", name="Main"),
            SessionMode(id="worker", name="Worker"),
        ]
    )
    context.set_current_mode("main")

    await context.switch_mode("worker")

    assert context.current_mode == "worker"
    assert synced_modes == ["worker"]
    assert len(connection.session_updates) == 1
    session_id, update = connection.session_updates[0]
    assert session_id == "session-1"
    assert isinstance(update, CurrentModeUpdate)
    assert update.current_mode_id == "worker"
