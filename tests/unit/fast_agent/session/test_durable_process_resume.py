import json
import logging
import os
from pathlib import Path

import pytest

from fast_agent.session.durable_processes import resume_durable_processes
from fast_agent.tools.durable_processes import DurableProcessStore
from fast_agent.tools.shell_runtime import ShellRuntime

_SHELL = Path("/bin/sh")


class _Agent:
    def __init__(self, shell_runtime: ShellRuntime) -> None:
        self.shell_runtime = shell_runtime


class _NoShellAgent:
    pass


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_resume_attaches_active_process_and_reports_unavailable_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processes"
    session_id = "2608251200-AbCdEf"
    store = DurableProcessStore(root)
    active = store.create(
        command="sleep 30",
        shell=_SHELL,
        cwd=tmp_path,
        origin_session_id=session_id,
        agent_name="worker",
    )
    unavailable = store.create(
        command="sleep 60",
        shell=_SHELL,
        cwd=tmp_path,
        origin_session_id=session_id,
        agent_name="worker",
    )
    status_path = store.directory(unavailable.spec.process_id) / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "state": "running",
            "supervisor_pid": 99_999_999,
            "child_pid": 99_999_998,
        }
    )
    status_path.write_text(json.dumps(status), encoding="utf-8")
    main_runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )
    worker_runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )

    try:
        result = await resume_durable_processes(
            {
                "main": _Agent(main_runtime),
                "worker": _Agent(worker_runtime),
            },
            session_id=session_id,
            fallback_agent_name="main",
        )
        main_attached = await main_runtime.process_snapshots()
        worker_attached = await worker_runtime.process_snapshots()
    finally:
        await main_runtime.close()
        await worker_runtime.close()

    assert [snapshot.spec.process_id for snapshot in result.attached] == [active.spec.process_id]
    assert [snapshot.spec.process_id for snapshot in result.unavailable] == [
        unavailable.spec.process_id
    ]
    assert main_attached == ()
    assert [snapshot.process_id for snapshot in worker_attached] == [active.spec.process_id]


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_resume_leaves_process_unattached_without_origin_or_fallback_runtime(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processes"
    session_id = "2608251200-AbCdEf"
    store = DurableProcessStore(root)
    created = store.create(
        command="sleep 30",
        shell=_SHELL,
        cwd=tmp_path,
        origin_session_id=session_id,
        agent_name="missing",
    )
    worker_runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )

    try:
        result = await resume_durable_processes(
            {
                "main": _NoShellAgent(),
                "worker": _Agent(worker_runtime),
            },
            session_id=session_id,
            fallback_agent_name="main",
        )
        attached = await worker_runtime.process_snapshots()
    finally:
        await worker_runtime.close()

    assert result.attached == ()
    assert [snapshot.spec.process_id for snapshot in result.unattached] == [created.spec.process_id]
    assert attached == ()
    assert store.get(created.spec.process_id).status.state == "created"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_resume_retries_fallback_when_origin_runtime_has_no_durable_store(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processes"
    session_id = "2608251200-AbCdEf"
    store = DurableProcessStore(root)
    created = store.create(
        command="sleep 30",
        shell=_SHELL,
        cwd=tmp_path,
        origin_session_id=session_id,
        agent_name="origin",
    )
    origin_runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
    )
    fallback_runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )

    try:
        result = await resume_durable_processes(
            {
                "origin": _Agent(origin_runtime),
                "fallback": _Agent(fallback_runtime),
            },
            session_id=session_id,
            fallback_agent_name="fallback",
        )
        attached = await fallback_runtime.process_snapshots()
    finally:
        await origin_runtime.close()
        await fallback_runtime.close()

    assert [snapshot.spec.process_id for snapshot in result.attached] == [created.spec.process_id]
    assert [snapshot.process_id for snapshot in attached] == [created.spec.process_id]
