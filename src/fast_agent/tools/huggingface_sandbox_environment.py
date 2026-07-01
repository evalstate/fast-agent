"""Hugging Face Sandbox-backed session environment."""

from __future__ import annotations

import asyncio
import posixpath
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol

from fast_agent.tools.session_environment import (
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionOptions,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellRuntimeInfo,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

DEFAULT_HF_SANDBOX_IDLE_TIMEOUT = 10 * 60


class _SandboxCommandResult(Protocol):
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool


class _SandboxFiles(Protocol):
    def read_text(self, path: str, encoding: str = "utf-8") -> str: ...

    def write(self, path: str, data: str | bytes, mode: str | None = None) -> None: ...

    def exists(self, path: str) -> bool: ...

    def mkdir(self, path: str) -> None: ...

    def delete(self, path: str, recursive: bool = False) -> None: ...


class _Sandbox(Protocol):
    files: _SandboxFiles

    def run(
        self,
        cmd: str | list[str],
        *,
        shell: bool | None = None,
        env: dict[str, Any] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> _SandboxCommandResult: ...

    def kill(self) -> None: ...

    def close(self) -> None: ...


class _SandboxClass(Protocol):
    @staticmethod
    def create(
        image: str = "python:3.12",
        *,
        flavor: str = "cpu-basic",
        idle_timeout: int | float | str | None = DEFAULT_HF_SANDBOX_IDLE_TIMEOUT,
        env: dict[str, Any] | None = None,
        secrets: dict[str, Any] | None = None,
        volumes: list[Any] | None = None,
        namespace: str | None = None,
        forward_hf_token: bool = False,
        start_timeout: float = 120.0,
        token: str | None = None,
    ) -> _Sandbox: ...


class _VolumeClass(Protocol):
    def __call__(
        self,
        **kwargs: Any,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class HuggingFaceBucketMount:
    """Bucket mount for a Hugging Face sandbox."""

    source: str
    mount_path: str
    read_only: bool = False
    path: str | None = None


class HuggingFaceSandboxEnvironment:
    """Run shell and filesystem tools inside a Hugging Face Sandbox."""

    def __init__(
        self,
        sandbox: _Sandbox | None = None,
        *,
        image: str = "python:3.12",
        flavor: str = "cpu-basic",
        cwd: str = "/workspace",
        bucket_mounts: tuple[HuggingFaceBucketMount, ...] = (),
        idle_timeout: int | float | str | None = None,
        env: dict[str, Any] | None = None,
        secrets: dict[str, Any] | None = None,
        namespace: str | None = None,
        forward_hf_token: bool = False,
        token: str | None = None,
        start_timeout: float = 120.0,
        owns_sandbox: bool = False,
    ) -> None:
        self._sandbox = sandbox
        self._image = image
        self._flavor = flavor
        self._cwd = _normalize_posix(cwd)
        self._bucket_mounts = bucket_mounts
        self._idle_timeout = idle_timeout
        self._env = env
        self._secrets = secrets
        self._namespace = namespace
        self._forward_hf_token = forward_hf_token
        self._token = token
        self._start_timeout = start_timeout
        self._owns_sandbox = owns_sandbox

    async def open(self) -> None:
        if self._sandbox is None:
            self._sandbox = await asyncio.to_thread(self._create_sandbox)
            self._owns_sandbox = True
        await asyncio.to_thread(self._sandbox.files.mkdir, self._cwd)

    def _create_sandbox(self) -> _Sandbox:
        try:
            from huggingface_hub import Sandbox, Volume
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face sandbox support requires huggingface_hub with Sandbox support."
            ) from exc
        sandbox_cls: _SandboxClass = Sandbox
        volume_cls: _VolumeClass = Volume

        idle_timeout = (
            DEFAULT_HF_SANDBOX_IDLE_TIMEOUT if self._idle_timeout is None else self._idle_timeout
        )
        volumes = [
            volume_cls(
                type="bucket",
                source=mount.source,
                mount_path=mount.mount_path,
                read_only=mount.read_only,
                path=mount.path,
            )
            for mount in self._bucket_mounts
        ]
        return sandbox_cls.create(
            image=self._image,
            flavor=self._flavor,
            idle_timeout=idle_timeout,
            env=self._env,
            secrets=self._secrets,
            volumes=volumes,
            namespace=self._namespace,
            forward_hf_token=self._forward_hf_token,
            start_timeout=self._start_timeout,
            token=self._token,
        )

    @property
    def cwd(self) -> str:
        return self._cwd

    def set_cwd(self, cwd: str | None) -> None:
        if cwd is not None:
            self._cwd = _normalize_posix(cwd)

    def resolve_path(self, path: str) -> str:
        if path.startswith("/"):
            return _normalize_posix(path)
        return _normalize_posix(posixpath.join(self._cwd, path))

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(name="sh", kind="remote", provider="huggingface")

    async def execute_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: "Mapping[str, str] | None" = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        execution = await self.execute(
            ShellExecutionRequest(
                command=command,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                timeout=timeout,
            )
        )
        return execution.result

    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        sandbox = self._require_sandbox()
        cwd = self.resolve_path(request.cwd) if request.cwd is not None else self._cwd

        def run_command():
            return sandbox.run(
                request.command,
                shell=True,
                env=dict(request.env or {}),
                cwd=cwd,
                timeout=request.timeout,
                check=False,
            )

        result = await asyncio.to_thread(run_command)
        if callbacks is not None:
            if result.stdout:
                await callbacks.on_stdout(result.stdout)
            if result.stderr:
                await callbacks.on_stderr(result.stderr)
            if result.timed_out:
                await callbacks.on_timeout()
        return ShellExecution(
            result=ShellExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code if result.exit_code is not None else 1,
            ),
            options=ShellExecutionOptions(timeout_seconds=request.timeout),
            timed_out=bool(result.timed_out),
        )

    async def read_text(self, path: str) -> str:
        sandbox = self._require_sandbox()
        return await asyncio.to_thread(sandbox.files.read_text, self.resolve_path(path))

    async def write_text(self, path: str, content: str) -> None:
        sandbox = self._require_sandbox()
        await asyncio.to_thread(sandbox.files.write, self.resolve_path(path), content)

    async def exists(self, path: str) -> bool:
        sandbox = self._require_sandbox()
        return await asyncio.to_thread(sandbox.files.exists, self.resolve_path(path))

    async def mkdir(self, path: str) -> None:
        sandbox = self._require_sandbox()
        await asyncio.to_thread(sandbox.files.mkdir, self.resolve_path(path))

    async def remove(self, path: str) -> None:
        sandbox = self._require_sandbox()
        await asyncio.to_thread(sandbox.files.delete, self.resolve_path(path), False)

    async def close(self) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return
        if self._owns_sandbox:
            await asyncio.to_thread(sandbox.kill)
        else:
            await asyncio.to_thread(sandbox.close)
        self._sandbox = None

    def _require_sandbox(self) -> _Sandbox:
        if self._sandbox is None:
            raise RuntimeError("Hugging Face sandbox environment is not open.")
        return self._sandbox


def _normalize_posix(path: str) -> str:
    raw = str(PurePosixPath(path))
    if not raw.startswith("/"):
        raw = f"/{raw}"
    return posixpath.normpath(raw)


__all__ = [
    "HuggingFaceBucketMount",
    "HuggingFaceSandboxEnvironment",
]
