"""Hugging Face Sandbox-backed execution environment."""

from __future__ import annotations

import asyncio
import base64
import json
import posixpath
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import PurePosixPath
from typing import Any, Callable, Literal, Protocol, cast

from fast_agent.core.exceptions import EnvironmentStartupError
from fast_agent.tools.execution_environment import (
    EnvironmentFileEntry,
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionOptions,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellRuntimeInfo,
)
from fast_agent.utils.huggingface_hub import get_huggingface_hub_token

DEFAULT_HF_SANDBOX_IDLE_TIMEOUT = 10 * 60
FAST_AGENT_HF_SANDBOX_LABEL = "fast-agent"
_ENV_REFERENCE_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_LIST_DIR_SCRIPT = """
import json
import os
import sys

root = sys.argv[1]
entries = []
for name in sorted(os.listdir(root)):
    path = os.path.join(root, name)
    if os.path.isdir(path):
        kind = "directory"
    elif os.path.isfile(path):
        kind = "file"
    else:
        kind = "other"
    entries.append({"path": path, "name": name, "kind": kind})
print(json.dumps(entries), end="")
""".strip()
_READ_BYTES_SCRIPT = """
import base64
import pathlib
import sys

sys.stdout.write(base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode("ascii"))
""".strip()


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
    id: str
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
class HuggingFaceVolumeMount:
    """Volume mount for a Hugging Face sandbox."""

    type: Literal["bucket", "model", "dataset", "space"]
    source: str
    mount_path: str
    read_only: bool | None = None
    path: str | None = None
    revision: str | None = None


@dataclass(frozen=True, slots=True)
class HuggingFaceBucketMount:
    """Bucket mount for a Hugging Face sandbox."""

    source: str
    mount_path: str
    read_only: bool = False
    path: str | None = None

    def as_volume_mount(self) -> HuggingFaceVolumeMount:
        return HuggingFaceVolumeMount(
            type="bucket",
            source=self.source,
            mount_path=self.mount_path,
            read_only=self.read_only,
            path=self.path,
        )


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
        volume_mounts: tuple[HuggingFaceVolumeMount, ...] = (),
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
        self._volume_mounts = tuple(
            [*(mount.as_volume_mount() for mount in bucket_mounts), *volume_mounts]
        )
        self._idle_timeout = idle_timeout
        self._env = env
        self._secrets = secrets
        self._namespace = namespace
        self._forward_hf_token = forward_hf_token
        self._token = token
        self._start_timeout = start_timeout
        self._owns_sandbox = owns_sandbox
        self._execution_env = dict(env or {})
        self._startup_progress_callback: Callable[[str], None] | None = None

    async def open(self) -> None:
        if self._sandbox is None:
            self._emit_startup_stage(
                f"creating sandbox image={self._image} flavor={self._flavor}"
            )
            self._sandbox = await asyncio.to_thread(self._create_sandbox)
            self._owns_sandbox = True
        else:
            self._emit_startup_stage(f"using existing sandbox {self._sandbox.id}")
        self._emit_startup_stage(f"preparing cwd {self._cwd}")
        await asyncio.to_thread(self._sandbox.files.mkdir, self._cwd)
        self._emit_startup_stage("sandbox filesystem ready")

    def set_startup_progress_callback(
        self,
        callback: Callable[[str], None] | None,
    ) -> None:
        self._startup_progress_callback = callback

    def _emit_startup_stage(self, stage: str) -> None:
        if self._startup_progress_callback is not None:
            self._startup_progress_callback(stage)

    def _create_sandbox(self) -> _Sandbox:
        token = _resolve_huggingface_token(self._token)
        try:
            from huggingface_hub import HfApi, Sandbox, Volume
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face sandbox support requires huggingface_hub with Sandbox support."
            ) from exc
        api = HfApi(token=token)
        sandbox_cls: _SandboxClass = Sandbox
        volume_cls: _VolumeClass = Volume

        idle_timeout = (
            DEFAULT_HF_SANDBOX_IDLE_TIMEOUT if self._idle_timeout is None else self._idle_timeout
        )
        self._emit_startup_stage(f"building {len(self._volume_mounts)} volume mount(s)")
        volumes = [
            volume_cls(
                type=mount.type,
                source=mount.source,
                mount_path=mount.mount_path,
                read_only=mount.read_only,
                path=mount.path,
                revision=mount.revision,
            )
            for mount in self._volume_mounts
        ]
        self._emit_startup_stage("calling Sandbox.create")
        try:
            sandbox = cast(
                "_Sandbox",
                sandbox_cls.create(
                    image=self._image,
                    flavor=self._flavor,
                    idle_timeout=idle_timeout,
                    env=self._env,
                    secrets=self._secrets,
                    volumes=volumes,
                    namespace=self._namespace,
                    forward_hf_token=self._forward_hf_token,
                    start_timeout=self._start_timeout,
                    token=token,
                ),
            )
        except Exception as exc:
            if _is_huggingface_auth_error(exc):
                raise EnvironmentStartupError(
                    "Could not start Hugging Face sandbox environment.",
                    (
                        "Hugging Face rejected the configured token. "
                        "Check the environment token setting, set HF_TOKEN, "
                        "or run `huggingface-cli login` to use the local token cache."
                    ),
                ) from exc
            raise
        self._emit_startup_stage(f"sandbox created {sandbox.id}")
        try:
            self._emit_startup_stage("applying fast-agent sandbox labels")
            api.update_job_labels(
                job_id=sandbox.id,
                labels=_fast_agent_sandbox_labels(),
                namespace=self._namespace,
                token=token,
            )
        except Exception:
            self._emit_startup_stage("label update failed; killing sandbox")
            sandbox.kill()
            raise
        return sandbox

    @property
    def cwd(self) -> str:
        return self._cwd

    def resolve_path(self, path: str) -> str:
        if path.startswith("/"):
            return _normalize_posix(path)
        return _normalize_posix(posixpath.join(self._cwd, path))

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(name="sh", kind="remote", provider="huggingface")

    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        sandbox = self._require_sandbox()
        cwd = self.resolve_path(request.cwd) if request.cwd is not None else self._cwd

        def run_command():
            effective_env = dict(self._execution_env)
            effective_env.update(request.env or {})
            return sandbox.run(
                request.command,
                shell=True,
                env=effective_env,
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

    async def read_bytes(self, path: str) -> bytes:
        sandbox = self._require_sandbox()
        resolved_path = self.resolve_path(path)

        def read_file() -> _SandboxCommandResult:
            return sandbox.run(
                ["python3", "-c", _READ_BYTES_SCRIPT, resolved_path],
                shell=False,
                check=False,
            )

        result = await asyncio.to_thread(read_file)
        if result.exit_code not in {0, None}:
            message = result.stderr.strip() or f"Unable to read file: {resolved_path}"
            raise RuntimeError(message)
        return base64.b64decode(result.stdout.encode("ascii"))

    async def write_bytes(self, path: str, content: bytes) -> None:
        sandbox = self._require_sandbox()
        await asyncio.to_thread(sandbox.files.write, self.resolve_path(path), content)

    async def exists(self, path: str) -> bool:
        sandbox = self._require_sandbox()
        return await asyncio.to_thread(sandbox.files.exists, self.resolve_path(path))

    async def list_dir(self, path: str) -> list[EnvironmentFileEntry]:
        sandbox = self._require_sandbox()
        resolved_path = self.resolve_path(path)

        def list_directory() -> _SandboxCommandResult:
            return sandbox.run(
                ["python3", "-c", _LIST_DIR_SCRIPT, resolved_path],
                shell=False,
                check=False,
            )

        result = await asyncio.to_thread(list_directory)
        if result.exit_code not in {0, None}:
            message = result.stderr.strip() or f"Unable to list directory: {resolved_path}"
            raise RuntimeError(message)
        return _parse_environment_file_entries(result.stdout)

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


def _parse_environment_file_entries(payload: str) -> list[EnvironmentFileEntry]:
    raw_entries = json.loads(payload)
    if not isinstance(raw_entries, list):
        raise RuntimeError("Sandbox directory listing returned invalid data.")

    entries: list[EnvironmentFileEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise RuntimeError("Sandbox directory listing returned invalid entry data.")
        path = raw_entry.get("path")
        name = raw_entry.get("name")
        kind = raw_entry.get("kind")
        if not isinstance(path, str) or not isinstance(name, str):
            raise RuntimeError("Sandbox directory listing entry is missing path or name.")
        entries.append(
            EnvironmentFileEntry(
                path=path,
                name=name,
                kind=_coerce_environment_file_kind(kind),
            )
        )
    return entries


def _coerce_environment_file_kind(
    value: object,
) -> Literal["file", "directory", "other", "unknown"]:
    if value == "file":
        return "file"
    if value == "directory":
        return "directory"
    if value == "other":
        return "other"
    return "unknown"


def _fast_agent_sandbox_labels() -> dict[str, str]:
    return {FAST_AGENT_HF_SANDBOX_LABEL: _label_safe_fast_agent_version()}


def _unresolved_env_reference(value: str | None) -> str | None:
    if value is None:
        return None
    match = _ENV_REFERENCE_PATTERN.fullmatch(value)
    if match is None:
        return None
    return match.group(1)


def _resolve_huggingface_token(configured_token: str | None) -> str | None:
    if unresolved_env_var := _unresolved_env_reference(configured_token):
        if unresolved_env_var == "HF_TOKEN":
            token = get_huggingface_hub_token()
            if token is not None:
                return token
            details = (
                "Environment variable HF_TOKEN is not set and no Hugging Face "
                "token was found in the local token cache. Set HF_TOKEN or run "
                "`huggingface-cli login`."
            )
        else:
            details = (
                f"Environment variable {unresolved_env_var} is not set. "
                f"Set {unresolved_env_var}, or remove "
                f"`token: ${{{unresolved_env_var}}}` to use the Hugging Face token cache."
            )
        raise EnvironmentStartupError(
            "Could not start Hugging Face sandbox environment.",
            details,
        )
    return configured_token


def _is_huggingface_auth_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "Invalid user token" in message
        or "Invalid username or password" in message
        or "401 Unauthorized" in message
    )


def _label_safe_fast_agent_version() -> str:
    try:
        raw_version = version("fast-agent-mcp")
    except PackageNotFoundError:
        raw_version = "unknown"
    label_value = re.sub(r"[^A-Za-z0-9_-]", "_", raw_version)
    return label_value[:100] or "unknown"


__all__ = [
    "FAST_AGENT_HF_SANDBOX_LABEL",
    "HuggingFaceBucketMount",
    "HuggingFaceSandboxEnvironment",
    "HuggingFaceVolumeMount",
]
