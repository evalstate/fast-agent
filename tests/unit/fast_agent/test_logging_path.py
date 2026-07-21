from pathlib import Path

import pytest

from fast_agent.config import LoggerSettings, Settings, get_settings
from fast_agent.context import configure_logger
from fast_agent.core.logging.events import Event
from fast_agent.core.logging.logger import LoggingConfig
from fast_agent.core.logging.transport import AsyncEventBus
from fast_agent.paths import resolve_log_file_path


def test_default_log_path_uses_active_home(tmp_path: Path) -> None:
    home = tmp_path / "custom-home"
    settings = get_settings(home=home)

    assert resolve_log_file_path(settings) == home / "fast-agent-log.jsonl"


def test_default_log_path_uses_configured_home(tmp_path: Path) -> None:
    home = tmp_path / "configured-home"
    settings = Settings(home=str(home))

    assert resolve_log_file_path(settings) == home / "fast-agent-log.jsonl"


def test_default_log_path_stays_in_working_directory_without_home() -> None:
    settings = get_settings(no_home=True)

    assert resolve_log_file_path(settings) == Path("fast-agent-log.jsonl")


def test_explicit_relative_log_path_keeps_existing_semantics() -> None:
    settings = Settings(logger=LoggerSettings(path="logs/events.jsonl"))

    assert resolve_log_file_path(settings) == Path("logs/events.jsonl")


@pytest.mark.asyncio
async def test_logger_writes_default_log_to_active_home(tmp_path: Path) -> None:
    home = tmp_path / "active-home"
    settings = get_settings(home=home)

    try:
        await configure_logger(settings)
        await AsyncEventBus.get().emit(
            Event(type="warning", namespace="test", message="home log destination")
        )
    finally:
        await LoggingConfig.shutdown()
        AsyncEventBus.reset()

    assert (home / "fast-agent-log.jsonl").is_file()
