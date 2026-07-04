import pytest

from fast_agent.core.environment_lifecycle import open_environment_with_progress
from fast_agent.event_progress import ProgressAction
from fast_agent.tools.execution_environment import (
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionRequest,
    ShellRuntimeInfo,
)


class _Logger:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def info(self, message: str, *, data: dict[str, object]) -> None:
        del message
        self.events.append(data)


class _Environment:
    def __init__(self, *, fail_open: bool = False) -> None:
        self.fail_open = fail_open
        self.callback = None

    async def open(self) -> None:
        if self.callback is not None:
            self.callback("opening test environment")
        if self.fail_open:
            raise RuntimeError("offline")

    def set_startup_progress_callback(self, callback) -> None:
        self.callback = callback

    @property
    def cwd(self) -> str:
        return "/workspace"

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(
            name="bash",
            kind="docker",
            provider="docker",
            environment_name="ubuntu",
        )

    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        raise AssertionError("not used")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_open_environment_with_progress_emits_connecting_and_ready() -> None:
    logger = _Logger()

    await open_environment_with_progress(_Environment(), logger=logger)

    assert [event["progress_action"] for event in logger.events] == [
        ProgressAction.CONNECTING,
        ProgressAction.TOOL_PROGRESS,
        ProgressAction.READY,
    ]
    assert "server_name" not in logger.events[0]
    assert logger.events[0]["target"] == "ubuntu"
    assert logger.events[0]["details"] == "ubuntu docker | cwd: /workspace"
    assert logger.events[1]["target"] == "ubuntu"
    assert logger.events[1]["details"] == "opening test environment"


@pytest.mark.asyncio
async def test_open_environment_with_progress_emits_fatal_error() -> None:
    logger = _Logger()

    with pytest.raises(RuntimeError, match="offline"):
        await open_environment_with_progress(_Environment(fail_open=True), logger=logger)

    assert [event["progress_action"] for event in logger.events] == [
        ProgressAction.CONNECTING,
        ProgressAction.TOOL_PROGRESS,
        ProgressAction.FATAL_ERROR,
    ]
    assert "offline" in str(logger.events[2]["details"])
