from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from fast_agent.core.exceptions import EnvironmentStartupError
from fast_agent.tools import huggingface_sandbox_environment as hf_sandbox_environment
from fast_agent.tools.execution_environment import ShellExecutionRequest
from fast_agent.tools.huggingface_sandbox_environment import (
    FAST_AGENT_HF_SANDBOX_LABEL,
    HuggingFaceSandboxEnvironment,
    HuggingFaceVolumeMount,
    _SandboxCommandResult,
    _SandboxFiles,
)
from fast_agent.tools.huggingface_sandbox_environment import (
    _Sandbox as SandboxProtocol,
)


class _CommandResult(_SandboxCommandResult):
    def __init__(
        self,
        *,
        stdout: str = "ok",
        stderr: str = "",
        exit_code: int | None = 0,
        timed_out: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.timed_out = timed_out


class _Files(_SandboxFiles):
    def __init__(self) -> None:
        self.created: list[str] = []

    def mkdir(self, path: str) -> None:
        self.created.append(path)

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return ""

    def write(self, path: str, data: str | bytes, mode: str | None = None) -> None:
        pass

    def exists(self, path: str) -> bool:
        return True

    def delete(self, path: str, recursive: bool = False) -> None:
        pass


class _Sandbox(SandboxProtocol):
    def __init__(self) -> None:
        self.id = "sandbox-job-123"
        self.test_files = _Files()
        self.files: _Files = self.test_files
        self.cwd: str | None = None
        self.commands: list[str | list[str]] = []

    def run(
        self,
        cmd: str | list[str],
        *,
        shell: bool | None = None,
        env: dict[str, Any] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> _CommandResult:
        del shell, env, timeout, check
        self.commands.append(cmd)
        self.cwd = cwd
        if isinstance(cmd, list) and cmd[:2] == ["python3", "-c"]:
            return _CommandResult(
                stdout=(
                    "["
                    '{"path": "/workspace/skills/alpha", "name": "alpha", "kind": "directory"},'
                    '{"path": "/workspace/skills/readme.txt", "name": "readme.txt", "kind": "file"}'
                    "]"
                )
            )
        return _CommandResult()

    def kill(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_open_creates_configured_cwd() -> None:
    sandbox = _Sandbox()
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace/project")

    await environment.open()

    assert sandbox.test_files.created == ["/workspace/project"]


@pytest.mark.asyncio
async def test_open_existing_sandbox_emits_startup_stages() -> None:
    sandbox = _Sandbox()
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace/project")
    stages: list[str] = []
    environment.set_startup_progress_callback(stages.append)

    await environment.open()

    assert stages == [
        "using existing sandbox sandbox-job-123",
        "preparing cwd /workspace/project",
        "sandbox filesystem ready",
    ]


@pytest.mark.asyncio
async def test_execute_uses_created_default_cwd() -> None:
    sandbox = _Sandbox()
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    await environment.open()

    await environment.execute(ShellExecutionRequest(command="pwd"))

    assert sandbox.cwd == "/workspace"


@pytest.mark.asyncio
async def test_list_dir_returns_session_file_entries() -> None:
    sandbox = _Sandbox()
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    await environment.open()

    entries = await environment.list_dir("skills")

    assert [entry.name for entry in entries] == ["alpha", "readme.txt"]
    assert [entry.path for entry in entries] == [
        "/workspace/skills/alpha",
        "/workspace/skills/readme.txt",
    ]
    assert [entry.kind for entry in entries] == ["directory", "file"]
    assert isinstance(sandbox.commands[-1], list)
    assert sandbox.commands[-1][0] == "python3"
    assert sandbox.commands[-1][-1] == "/workspace/skills"


def test_create_sandbox_adds_fast_agent_version_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_update: dict[str, Any] | None = None

    class FakeSandbox:
        @staticmethod
        def create(
            image: str = "python:3.12",
            *,
            flavor: str = "cpu-basic",
            idle_timeout: int | float | str | None = None,
            env: dict[str, Any] | None = None,
            secrets: dict[str, Any] | None = None,
            volumes: list[Any] | None = None,
            namespace: str | None = None,
            forward_hf_token: bool = False,
            start_timeout: float = 120.0,
            token: str | None = None,
        ) -> _Sandbox:
            del (
                image,
                flavor,
                idle_timeout,
                env,
                secrets,
                volumes,
                namespace,
                forward_hf_token,
                start_timeout,
                token,
            )
            return _Sandbox()

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def update_job_labels(
            self,
            *,
            job_id: str,
            labels: dict[str, str],
            namespace: str | None = None,
            token: str | None = None,
        ) -> None:
            nonlocal captured_update
            captured_update = {
                "job_id": job_id,
                "labels": labels,
                "namespace": namespace,
                "token": token,
            }

    class FakeVolume:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeHfApi, Sandbox=FakeSandbox, Volume=FakeVolume),
    )
    monkeypatch.setattr(hf_sandbox_environment, "version", lambda package: "0.9.0")

    environment = HuggingFaceSandboxEnvironment(namespace="test-org", token="hf_test")

    environment._create_sandbox()

    assert captured_update == {
        "job_id": "sandbox-job-123",
        "labels": {FAST_AGENT_HF_SANDBOX_LABEL: "0_9_0"},
        "namespace": "test-org",
        "token": "hf_test",
    }


def test_create_sandbox_resolves_hf_token_reference_from_hub_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_token: str | None = None

    class FakeSandbox:
        @staticmethod
        def create(
            image: str = "python:3.12",
            *,
            flavor: str = "cpu-basic",
            idle_timeout: int | float | str | None = None,
            env: dict[str, Any] | None = None,
            secrets: dict[str, Any] | None = None,
            volumes: list[Any] | None = None,
            namespace: str | None = None,
            forward_hf_token: bool = False,
            start_timeout: float = 120.0,
            token: str | None = None,
        ) -> _Sandbox:
            del (
                image,
                flavor,
                idle_timeout,
                env,
                secrets,
                volumes,
                namespace,
                forward_hf_token,
                start_timeout,
            )
            nonlocal captured_token
            captured_token = token
            return _Sandbox()

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def update_job_labels(
            self,
            *,
            job_id: str,
            labels: dict[str, str],
            namespace: str | None = None,
            token: str | None = None,
        ) -> None:
            del job_id, labels, namespace, token

    class FakeVolume:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeHfApi, Sandbox=FakeSandbox, Volume=FakeVolume),
    )
    monkeypatch.setattr(
        hf_sandbox_environment,
        "get_huggingface_hub_token",
        lambda: "hf_cached_token",
    )
    environment = HuggingFaceSandboxEnvironment(token="${HF_TOKEN}")

    environment._create_sandbox()

    assert captured_token == "hf_cached_token"


def test_create_sandbox_rejects_unresolved_token_env_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_sandbox_environment, "get_huggingface_hub_token", lambda: None)
    environment = HuggingFaceSandboxEnvironment(token="${HF_TOKEN}")

    with pytest.raises(EnvironmentStartupError) as exc_info:
        environment._create_sandbox()

    assert exc_info.value.message == "Could not start Hugging Face sandbox environment."
    assert "Environment variable HF_TOKEN is not set" in exc_info.value.details


def test_create_sandbox_wraps_huggingface_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSandbox:
        @staticmethod
        def create(
            image: str = "python:3.12",
            *,
            flavor: str = "cpu-basic",
            idle_timeout: int | float | str | None = None,
            env: dict[str, Any] | None = None,
            secrets: dict[str, Any] | None = None,
            volumes: list[Any] | None = None,
            namespace: str | None = None,
            forward_hf_token: bool = False,
            start_timeout: float = 120.0,
            token: str | None = None,
        ) -> _Sandbox:
            del (
                image,
                flavor,
                idle_timeout,
                env,
                secrets,
                volumes,
                namespace,
                forward_hf_token,
                start_timeout,
                token,
            )
            raise RuntimeError("Invalid user token.")

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

    class FakeVolume:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeHfApi, Sandbox=FakeSandbox, Volume=FakeVolume),
    )
    environment = HuggingFaceSandboxEnvironment(token="bad-token")

    with pytest.raises(EnvironmentStartupError) as exc_info:
        environment._create_sandbox()

    assert exc_info.value.message == "Could not start Hugging Face sandbox environment."
    assert "Hugging Face rejected the configured token" in exc_info.value.details


def test_create_sandbox_passes_configured_hf_volume_mounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_volumes: list[Any] | None = None

    class FakeSandbox:
        @staticmethod
        def create(
            image: str = "python:3.12",
            *,
            flavor: str = "cpu-basic",
            idle_timeout: int | float | str | None = None,
            env: dict[str, Any] | None = None,
            secrets: dict[str, Any] | None = None,
            volumes: list[Any] | None = None,
            namespace: str | None = None,
            forward_hf_token: bool = False,
            start_timeout: float = 120.0,
            token: str | None = None,
        ) -> _Sandbox:
            del (
                image,
                flavor,
                idle_timeout,
                env,
                secrets,
                namespace,
                forward_hf_token,
                start_timeout,
                token,
            )
            nonlocal captured_volumes
            captured_volumes = volumes
            return _Sandbox()

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def update_job_labels(
            self,
            *,
            job_id: str,
            labels: dict[str, str],
            namespace: str | None = None,
            token: str | None = None,
        ) -> None:
            del job_id, labels, namespace, token

    class FakeVolume:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeHfApi, Sandbox=FakeSandbox, Volume=FakeVolume),
    )

    environment = HuggingFaceSandboxEnvironment(
        volume_mounts=(
            HuggingFaceVolumeMount(
                type="dataset",
                source="org/data",
                mount_path="/data",
                read_only=True,
                path="train",
                revision="main",
            ),
        )
    )

    environment._create_sandbox()

    assert captured_volumes is not None
    assert [volume.kwargs for volume in captured_volumes] == [
        {
            "type": "dataset",
            "source": "org/data",
            "mount_path": "/data",
            "read_only": True,
            "path": "train",
            "revision": "main",
        }
    ]
