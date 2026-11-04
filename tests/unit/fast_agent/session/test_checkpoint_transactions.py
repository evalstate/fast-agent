"""Committed checkpoints survive failed writes through public session APIs."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest
from mcp_types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import load_messages
from fast_agent.ptc.store import SandboxStore
from fast_agent.session.session_manager import SessionManager


def _agent(text: str) -> LlmAgent:
    agent = LlmAgent(AgentConfig(name="main", instruction="Offline", model="passthrough"))
    _history(agent, text)
    return agent


def _history(agent: LlmAgent, text: str) -> None:
    agent.load_message_history(
        [PromptMessageExtended(role="user", content=[TextContent(type="text", text=text)])]
    )


def _manager(root: Path) -> SessionManager:
    return SessionManager(cwd=root, home_override=root / ".fast-agent", respect_env_override=False)


def _texts(path: Path) -> list[str]:
    return [message.all_text() for message in load_messages(str(path))]


def _fail_metadata_once(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    original = os.replace
    failed = False

    def replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        nonlocal failed
        if Path(dst) == directory / "session.json" and not failed:
            failed = True
            raise OSError(errno.ENOSPC, "injected metadata publication failure")
        original(src, dst)

    monkeypatch.setattr(os, "replace", replace)


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", [None, "custom.json", "custom.md"])
async def test_failed_checkpoint_keeps_committed_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str | None
) -> None:
    manager = _manager(tmp_path)
    session = manager.create_session()
    agent = _agent("committed")
    await session.save_history(agent, filename=filename)
    committed_metadata = (session.directory / "session.json").read_bytes()
    _history(agent, "not committed")
    _fail_metadata_once(monkeypatch, session.directory)

    with pytest.raises(OSError, match="injected metadata"):
        await session.save_history(agent, filename=filename)

    loaded = manager.get_session(session.info.name)
    assert loaded is not None
    history = loaded.latest_history_path("main")
    assert history is not None
    assert (session.directory / "session.json").read_bytes() == committed_metadata
    assert _texts(history) == ["committed"]
    manager.close()


@pytest.mark.asyncio
async def test_fork_after_failed_save_uses_last_committed_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _manager(tmp_path)
    session = owner.create_session()
    agent = _agent("committed")
    await session.save_history(agent)
    _history(agent, "not committed")
    _fail_metadata_once(monkeypatch, session.directory)
    with pytest.raises(OSError):
        await session.save_history(agent)
    other = _manager(tmp_path)

    forked = other.fork_session(session.info.name)

    assert forked is not None
    history = forked.latest_history_path("main")
    assert history is not None
    assert _texts(history) == ["committed"]
    owner.close()
    other.close()


@pytest.mark.asyncio
async def test_failed_first_save_does_not_claim_unwritten_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    session = manager.create_session()
    agent = _agent("not committed")
    _fail_metadata_once(monkeypatch, session.directory)

    with pytest.raises(OSError):
        await session.save_history(agent)

    assert session.info.history_files == []
    assert session.latest_history_path("main") is None
    assert not (session.directory / "history_main.json").exists()
    manager.close()


class _StoredAgent(LlmAgent):
    def __init__(self, text: str) -> None:
        super().__init__(AgentConfig(name="main", instruction="Offline", model="passthrough"))
        self.store = SandboxStore()
        _history(self, text)

    @property
    def ptc_store(self) -> SandboxStore:
        return self.store


@pytest.mark.asyncio
@pytest.mark.parametrize("clear_store", [False, True])
async def test_failed_checkpoint_preserves_ptc_store_and_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clear_store: bool
) -> None:
    manager = _manager(tmp_path)
    session = manager.create_session()
    agent = _StoredAgent("committed")
    await agent.store.functions()["store"]("answer", "committed")
    await session.save_history(agent)
    store_file = session.directory / "ptc_store_main.json"
    old_store = store_file.read_bytes()
    _history(agent, "not committed")
    if clear_store:
        agent.store.clear()
    else:
        await agent.store.functions()["store"]("answer", "not committed")
    _fail_metadata_once(monkeypatch, session.directory)

    with pytest.raises(OSError):
        await session.save_history(agent)

    assert store_file.read_bytes() == old_store
    old_history = session.latest_history_path("main")
    assert old_history is not None
    assert _texts(old_history) == ["committed"]
    await session.save_history(agent)
    new_history = session.latest_history_path("main")
    assert new_history is not None
    assert _texts(new_history) == ["not committed"]
    if clear_store:
        assert not store_file.exists()
        assert session.load_snapshot().continuation.agents["main"].ptc_store_file is None
    else:
        assert "not committed" in store_file.read_text()
    manager.close()


@pytest.mark.asyncio
async def test_store_revision_is_captured_with_staged_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import threading

    from fast_agent.session.checkpoint import CheckpointTransaction

    manager = _manager(tmp_path)
    session = manager.create_session()
    agent = _StoredAgent("checkpoint")
    await agent.store.functions()["store"]("answer", "staged")
    entered = threading.Event()
    release = threading.Event()
    original = CheckpointTransaction.stage_bytes

    def stage(transaction: CheckpointTransaction, filename: str, content: bytes) -> None:
        original(transaction, filename, content)
        if filename == "ptc_store_main.json":
            entered.set()
            assert release.wait(3)

    monkeypatch.setattr(CheckpointTransaction, "stage_bytes", stage)
    save = asyncio.create_task(session.save_history(agent))
    assert await asyncio.to_thread(entered.wait, 3)
    agent.store.clear()
    release.set()
    await save

    # The completed checkpoint references what was staged, even if the live
    # agent mutated while its filesystem write was in flight.
    assert (
        session.load_snapshot().continuation.agents["main"].ptc_store_file == "ptc_store_main.json"
    )
    assert "staged" in (session.directory / "ptc_store_main.json").read_text()
    await session.save_history(agent)
    assert not (session.directory / "ptc_store_main.json").exists()
    assert session.load_snapshot().continuation.agents["main"].ptc_store_file is None
    manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["history", "prepare"])
async def test_cancelled_checkpoint_waits_for_writer_before_unlocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    import asyncio
    import threading

    import fast_agent.history.history_exporter as exporter
    from fast_agent.session.checkpoint import CheckpointTransaction
    from fast_agent.session.locking import SessionCheckpointBusyError

    manager = _manager(tmp_path)
    session = manager.create_session()
    agent = _agent("committed")
    await session.save_history(agent)
    metadata = (session.directory / "session.json").read_bytes()
    _history(agent, "cancelled")
    entered = threading.Event()
    release = threading.Event()

    if phase == "history":
        original_write = exporter._write_bytes

        def write(filename: str, content: bytes) -> None:
            entered.set()
            assert release.wait(3)
            original_write(filename, content)

        monkeypatch.setattr(exporter, "_write_bytes", write)
    else:
        original_prepare = CheckpointTransaction.prepare

        def prepare(transaction: CheckpointTransaction, snapshot: bytes) -> None:
            original_prepare(transaction, snapshot)
            entered.set()
            assert release.wait(3)

        monkeypatch.setattr(CheckpointTransaction, "prepare", prepare)
    save = asyncio.create_task(session.save_history(agent))
    assert await asyncio.to_thread(entered.wait, 3)
    save.cancel()
    save.cancel()
    # A fork must not observe files while a cancelled writer can still modify
    # them. The nonblocking checkpoint lock provides the public busy result.
    other = _manager(tmp_path)
    try:
        with pytest.raises(SessionCheckpointBusyError):
            other.fork_session(session.info.name)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await save

    assert (session.directory / "session.json").read_bytes() == metadata
    path = session.latest_history_path("main")
    assert path is not None
    assert _texts(path) == ["committed"]
    assert not (session.directory / ".checkpoint-transaction").exists()
    fork = other.fork_session(session.info.name)
    assert fork is not None
    fork_path = fork.latest_history_path("main")
    assert fork_path is not None
    assert _texts(fork_path) == ["committed"]
    manager.close()
    other.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ["load", "fork", "resume", "get", "hydrate"])
async def test_interrupted_save_is_recovered_before_public_session_read(
    tmp_path: Path, reader: str
) -> None:
    import subprocess
    import sys

    manager = _manager(tmp_path)
    session = manager.create_session()
    agent = _agent("committed")
    await session.save_history(agent)
    session_id = session.info.name
    committed_metadata = (session.directory / "session.json").read_bytes()
    manager.close()
    program = r"""
import asyncio
import os
import sys
from pathlib import Path
from mcp_types import TextContent
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.session.session_manager import SessionManager
from fast_agent.ptc.store import SandboxStore
async def save():
    root = Path(sys.argv[1])
    manager = SessionManager(cwd=root, home_override=root / ".fast-agent", respect_env_override=False)
    session = manager.load_session(sys.argv[2])
    agent = LlmAgent(AgentConfig(name="main", instruction="Offline", model="passthrough"))
    agent.load_message_history([PromptMessageExtended(role="user", content=[TextContent(type="text", text="interrupted")])])
    original = os.replace
    def replace(source, destination):
        original(source, destination)
        if Path(destination) == session.directory / "history_main.json":
            os._exit(19)
    os.replace = replace
    await session.save_history(agent)
asyncio.run(save())
"""
    process = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), session_id],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert process.returncode == 19, process.stderr
    reopened = _manager(tmp_path)
    if reader == "fork":
        recovered = reopened.fork_session(session_id)
    elif reader == "get":
        recovered = reopened.get_session(session_id)
    elif reader == "hydrate":
        from fast_agent.session.hydrator import SessionHydrator

        recovered = reopened.get_session(session_id)
        assert recovered is not None
        restored = _agent("empty")
        await SessionHydrator().hydrate_session(
            session=recovered, agents={"main": restored}, fallback_agent_name="main"
        )
        assert [message.all_text() for message in restored.message_history] == ["committed"]
    elif reader == "resume":
        restored = _agent("empty")
        result = await reopened.resume_session_agents_async({"main": restored}, session_id)
        assert result is not None
        assert [message.all_text() for message in restored.message_history] == ["committed"]
        recovered = reopened.current_session
    else:
        recovered = reopened.load_session(session_id)

    assert recovered is not None
    path = recovered.latest_history_path("main")
    assert path is not None
    assert _texts(path) == ["committed"]
    assert (session.directory / "session.json").read_bytes() == committed_metadata
    assert not (session.directory / ".checkpoint-transaction").exists()
    reopened.close()


@pytest.mark.asyncio
async def test_runtime_only_metadata_does_not_prevent_checkpoint(tmp_path: Path) -> None:
    import threading

    manager = _manager(tmp_path)
    session = manager.create_session()
    session.info.metadata["runtime_only"] = threading.Lock()
    agent = _agent("committed")

    with pytest.warns(UserWarning, match="runtime_only"):
        await session.save_history(agent)

    assert "runtime_only" not in session.load_snapshot().metadata.extras
    path = session.latest_history_path("main")
    assert path is not None
    assert _texts(path) == ["committed"]
    manager.close()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX file permission contract")
async def test_checkpoint_files_keep_private_permissions(tmp_path: Path) -> None:
    import stat

    manager = _manager(tmp_path)
    session = manager.create_session()
    agent = _StoredAgent("private transcript")
    await agent.store.functions()["store"]("answer", "private value")

    await session.save_history(agent)

    for filename in ("history_main.json", "ptc_store_main.json", "session.json"):
        assert stat.S_IMODE((session.directory / filename).stat().st_mode) == 0o600
    manager.close()
