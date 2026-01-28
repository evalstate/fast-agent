from __future__ import annotations

from fast_agent.config import get_settings, update_global_settings
from fast_agent.session import get_session_manager, reset_session_manager


def test_prune_sessions_skips_pinned(tmp_path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    override = old_settings.model_copy(
        update={
            "environment_dir": str(env_dir),
            "session_history_window": 1,
        }
    )
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = get_session_manager()
        first = manager.create_session()
        first.set_pinned(True)

        second = manager.create_session()
        third = manager.create_session()

        sessions = manager.list_sessions()
        names = {session.name for session in sessions}

        assert first.info.name in names
        assert third.info.name in names
        assert second.info.name not in names
    finally:
        update_global_settings(old_settings)
        reset_session_manager()
