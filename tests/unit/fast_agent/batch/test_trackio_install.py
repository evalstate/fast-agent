"""Optional-extra smoke test against Trackio's real local storage."""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["local", "disabled"])
def test_trackio_extra_persists_batch_metrics(tmp_path: Path, mode: str) -> None:
    if mode == "local" and importlib.util.find_spec("trackio") is None:
        pytest.skip("requires the trackio extra")
    # Trackio reads its storage configuration at import time. Isolate it from
    # both other tests and any configured remote projects.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("TRACKIO_", "SPACE_"))
    }
    env.update(TRACKIO_DIR=str(tmp_path), HF_HUB_OFFLINE="1", TRACKIO_CAPTURE_CONSOLE="0")
    result = subprocess.run(
        [sys.executable, __file__, str(tmp_path), mode],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Trackio logging failed" not in result.stderr
    assert "Trackio finish failed" not in result.stderr


def _exercise_trackio(directory: Path) -> None:
    from fast_agent.batch.monitoring import BatchTrackioOptions, create_batch_monitor
    from fast_agent.batch.structured import StructuredBatchOptions
    from fast_agent.integrations.gepa import gepa_trackio_init_kwargs, safe_trackio_log

    options = StructuredBatchOptions(
        input_path=directory / "input.jsonl",
        output_path=directory / "output.jsonl",
        model="passthrough",
        trackio=BatchTrackioOptions(project="install-smoke", name="batch"),
    )
    monitor = create_batch_monitor(options)
    monitor.start(options, selected_rows=2)
    try:
        _assert_no_automatic_capture()
        monitor.complete(
            {"processed_rows": 2, "selected_rows": 2, "failed_rows": 0, "duration_ms": 100}
        )
        assert safe_trackio_log({"candidate/score": 0.75}, step=3)
    finally:
        monitor.close()

    # The SDK's storage reader verifies logging survived finish/flush, rather
    # than merely checking that the optional package imports.
    storage = importlib.import_module("trackio.sqlite_storage").SQLiteStorage
    logs = storage.get_logs("install-smoke", "batch")
    assert any(row.get("batch/processed_rows") == 2 for row in logs), logs
    assert any(row.get("candidate/score") == 0.75 for row in logs), logs

    trackio = importlib.import_module("trackio")
    trackio.init(**gepa_trackio_init_kwargs(project="gepa-smoke", name="direct"))
    try:
        _assert_no_automatic_capture()
        assert safe_trackio_log({"candidate/score": 1.0})
    finally:
        trackio.finish()
    assert any(
        row.get("candidate/score") == 1.0 for row in storage.get_logs("gepa-smoke", "direct")
    )


def _assert_no_automatic_capture() -> None:
    run = importlib.import_module("trackio.context_vars").current_run.get()
    assert run is not None
    assert run._cpu_monitor is None
    assert run._gpu_monitor is None
    assert run._console_capture is None


def _exercise_disabled(directory: Path) -> None:
    # Even ambient remote configuration must not activate monitoring.
    os.environ["TRACKIO_SPACE_ID"] = "unused/unused"
    os.environ["TRACKIO_SERVER_URL"] = "http://127.0.0.1:1"
    from fast_agent.batch.monitoring import BatchTrackioOptions, create_batch_monitor
    from fast_agent.batch.structured import StructuredBatchOptions
    from fast_agent.integrations import gepa  # noqa: F401

    for trackio_options in (None, BatchTrackioOptions()):
        options = StructuredBatchOptions(
            input_path=directory / "input.jsonl",
            output_path=directory / "output.jsonl",
            model="passthrough",
            trackio=trackio_options,
        )
        monitor = create_batch_monitor(options)
        monitor.start(options, selected_rows=0)
        monitor.complete({"processed_rows": 0})
        monitor.close()
    assert "trackio" not in sys.modules
    assert not list(directory.iterdir())


if __name__ == "__main__":
    network_attempts: list[str] = []

    def deny_network(event: str, args: tuple[object, ...]) -> None:
        if event in ("socket.connect", "socket.getaddrinfo"):
            network_attempts.append(event)
            raise RuntimeError(f"Unexpected network activity: {event}")

    # Audit hooks observe real SDK behavior without replacing socket functions.
    # Assert the record too, since optional telemetry may swallow exceptions.
    sys.addaudithook(deny_network)
    if sys.argv[2] == "local":
        _exercise_trackio(Path(sys.argv[1]))
    else:
        _exercise_disabled(Path(sys.argv[1]))
    assert not network_attempts, network_attempts
