from __future__ import annotations

import os

from fast_agent.config import get_settings, update_global_settings
from fast_agent.session import apply_session_window, get_session_manager, reset_session_manager


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


def test_get_session_manager_normalizes_relative_environment_dir(tmp_path) -> None:
    original_env = os.environ.get("ENVIRONMENT_DIR")
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir(parents=True)
    second_cwd.mkdir(parents=True)

    os.environ["ENVIRONMENT_DIR"] = ".dev"
    reset_session_manager()

    try:
        manager_first = get_session_manager(cwd=first_cwd)
        normalized_env = os.environ.get("ENVIRONMENT_DIR")
        assert normalized_env is not None
        assert normalized_env == str((first_cwd / ".dev").resolve())

        manager_second = get_session_manager(cwd=second_cwd)
        assert manager_second is manager_first
        assert manager_second.base_dir == (first_cwd / ".dev" / "sessions").resolve()
    finally:
        reset_session_manager()
        if original_env is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = original_env


def test_apply_session_window_appends_pinned_overflow(tmp_path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    override = old_settings.model_copy(
        update={
            "environment_dir": str(env_dir),
            "session_history_window": 2,
        }
    )
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = get_session_manager()
        oldest = manager.create_session()
        oldest.set_pinned(True)
        middle = manager.create_session()
        newest = manager.create_session()

        visible = apply_session_window(manager.list_sessions())
        names = [session.name for session in visible]

        assert names == [newest.info.name, middle.info.name, oldest.info.name]
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


def test_resolve_session_name_ordinal_includes_pinned_overflow(tmp_path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    override = old_settings.model_copy(
        update={
            "environment_dir": str(env_dir),
            "session_history_window": 2,
        }
    )
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = get_session_manager()
        oldest = manager.create_session()
        oldest.set_pinned(True)
        manager.create_session()
        manager.create_session()

        assert manager.resolve_session_name("3") == oldest.info.name
    finally:
        update_global_settings(old_settings)
        reset_session_manager()
