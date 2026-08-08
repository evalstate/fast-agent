from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from fast_agent.core.logging.logger import get_logger
from fast_agent.tools.execution_environment import (
    EnvironmentTemporaryArtifacts,
)
from fast_agent.tools.local_shell_executor import LocalEnvironment
from fast_agent.tools.transient_artifacts import (
    TRANSIENT_ARTIFACT_MAX_BYTES,
    TRANSIENT_ARTIFACT_QUOTA_MARKER,
    TransientArtifactStore,
    bounded_temporary_text,
    format_retained_artifact_notice,
)


def _local_environment(tmp_path: Path) -> LocalEnvironment:
    return LocalEnvironment(
        logger=get_logger(__name__),
        working_directory=tmp_path,
    )


@pytest.mark.unit
def test_bounded_temporary_text_preserves_utf8_and_marks_quota() -> None:
    assert TRANSIENT_ARTIFACT_MAX_BYTES == 2 * 1024 * 1024
    content = "a" * 20 + "🦊" + "tail" * 20

    payload, complete = bounded_temporary_text(content, max_bytes=65)

    assert not complete
    assert len(payload) <= 65
    assert payload.decode("utf-8") == "a" * 20 + TRANSIENT_ARTIFACT_QUOTA_MARKER


@pytest.mark.unit
def test_retained_artifact_notice_treats_native_path_as_opaque() -> None:
    path = "C:\\Temp\\folder with space\\fast-agent-subagent-é.log"

    complete = format_retained_artifact_notice(
        path=path,
        retained_bytes=123,
        complete=True,
        description="subagent transcript",
    )
    partial = format_retained_artifact_notice(
        path=path,
        retained_bytes=123,
        complete=False,
        description="subagent transcript",
    )

    assert f"complete subagent transcript is available during this session at {path}" in complete
    assert "first 123 bytes of the subagent transcript" in partial
    assert "temporary-file quota was reached" in partial


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_transient_store_is_private_unique_and_parent_scoped(tmp_path: Path) -> None:
    environment = _local_environment(tmp_path)
    assert isinstance(environment, EnvironmentTemporaryArtifacts)
    store = TransientArtifactStore(environment)

    first, second = await asyncio.gather(
        store.write_text(
            producer="subagent",
            suffix=".log",
            content="first transcript",
            description="subagent transcript",
        ),
        store.write_text(
            producer="subagent",
            suffix=".log",
            content="second transcript",
            description="subagent transcript",
        ),
    )
    first_path = Path(first.artifact.path)
    second_path = Path(second.artifact.path)
    partial = await store.write_text(
        producer="subagent",
        suffix=".log",
        content="x" * 100,
        description="subagent transcript",
        max_bytes=65,
    )
    partial_path = Path(partial.artifact.path)

    assert first_path != second_path
    assert first_path.name.startswith("fast-agent-subagent-")
    assert first_path.parent == second_path.parent
    assert first_path.parent.name.startswith("fast-agent-output-")
    assert first_path.read_text(encoding="utf-8") == "first transcript"
    assert second_path.read_text(encoding="utf-8") == "second transcript"
    assert not partial.artifact.complete
    assert partial.artifact.retained_bytes <= 65
    assert partial_path.read_text(encoding="utf-8").endswith(TRANSIENT_ARTIFACT_QUOTA_MARKER)
    assert "temporary-file quota was reached" in partial.notice
    if first_path.stat().st_mode & 0o777:
        assert first_path.stat().st_mode & 0o777 == 0o600
        assert first_path.parent.stat().st_mode & 0o777 == 0o700

    await store.close()
    await store.close()

    assert not first_path.exists()
    assert not second_path.exists()
    assert not partial_path.exists()
    assert environment._temporary_artifact_directory is None
    assert not first_path.parent.exists()
    await environment.close()
    assert not first_path.parent.exists()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shared_local_stores_remove_root_after_last_artifact(tmp_path: Path) -> None:
    environment = _local_environment(tmp_path)
    first_store = TransientArtifactStore(environment)
    second_store = TransientArtifactStore(environment)
    first = await first_store.write_text(
        producer="subagent",
        suffix=".log",
        content="first transcript",
        description="subagent transcript",
    )
    second = await second_store.write_text(
        producer="subagent",
        suffix=".log",
        content="second transcript",
        description="subagent transcript",
    )
    first_path = Path(first.artifact.path)
    second_path = Path(second.artifact.path)
    directory = first_path.parent

    await first_store.close()

    assert not first_path.exists()
    assert second_path.exists()
    assert directory.exists()
    assert environment._temporary_artifact_directory == directory

    await second_store.close()

    assert not second_path.exists()
    assert not directory.exists()
    assert environment._temporary_artifact_directory is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_transient_store_cleans_resolved_path_from_symlinked_temp_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_directory = tmp_path / "real-output"
    real_directory.mkdir()
    directory_alias = tmp_path / "fast-agent-output-alias"
    try:
        directory_alias.symlink_to(real_directory, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *, prefix: str(directory_alias))
    environment = _local_environment(tmp_path)
    store = TransientArtifactStore(environment)

    result = await store.write_text(
        producer="subagent",
        suffix=".log",
        content="transcript",
        description="subagent transcript",
    )
    artifact_path = Path(result.artifact.path)

    assert artifact_path.parent == real_directory.resolve()
    assert artifact_path.exists()

    await store.close()

    assert not artifact_path.exists()
    assert not real_directory.exists()
    assert environment._temporary_artifact_directory is None
