"""Factory helpers for named execution environment specs."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.core.logging.logger import get_logger
from fast_agent.paths import resolve_settings_start_path
from fast_agent.tools.environment_config import (
    CustomEnvironmentSpec,
    DockerEnvironmentSpec,
    EnvironmentSpecModel,
    HuggingFaceEnvironmentSpec,
    LocalEnvironmentSpec,
)
from fast_agent.tools.execution_environment import ShellEnvironment

if TYPE_CHECKING:
    from fast_agent.config import Settings


class EnvironmentConfigError(ValueError):
    """Raised when a configured execution environment cannot be built."""


def build_environment(
    spec: EnvironmentSpecModel,
    *,
    settings: Settings | None = None,
    workspace_root: Path | None = None,
    name: str | None = None,
) -> ShellEnvironment:
    """Build a fresh, unopened shell environment from a validated spec."""

    if workspace_root is None:
        workspace_root = resolve_settings_start_path(settings)

    environment = _build_unwrapped_environment(spec, settings=settings, workspace_root=workspace_root)
    validate_environment_type(environment, name=name)
    return environment


def validate_environment_type(environment: object, *, name: str | None = None) -> None:
    """Validate that a custom/built environment satisfies the shell protocol."""

    if isinstance(environment, ShellEnvironment):
        return
    label = f" '{name}'" if name else ""
    raise EnvironmentConfigError(f"Environment{label} does not implement ShellEnvironment.")


def _build_unwrapped_environment(
    spec: EnvironmentSpecModel,
    *,
    settings: Settings | None,
    workspace_root: Path,
) -> ShellEnvironment:
    if isinstance(spec, LocalEnvironmentSpec):
        return _build_local_environment(spec, settings=settings, workspace_root=workspace_root)
    if isinstance(spec, DockerEnvironmentSpec):
        return _build_docker_environment(spec, workspace_root=workspace_root)
    if isinstance(spec, HuggingFaceEnvironmentSpec):
        return _build_huggingface_environment(spec)
    if isinstance(spec, CustomEnvironmentSpec):
        return _build_custom_environment(spec)
    raise EnvironmentConfigError(f"Unsupported environment spec: {type(spec).__name__}")


def _build_local_environment(
    spec: LocalEnvironmentSpec,
    *,
    settings: Settings | None,
    workspace_root: Path,
) -> ShellEnvironment:
    from fast_agent.tools.local_shell_executor import LocalEnvironment

    working_directory = _resolve_workspace_path(spec.cwd, workspace_root) if spec.cwd else workspace_root
    shell_settings = settings.shell_execution if settings is not None else None
    return LocalEnvironment(
        logger=get_logger(__name__),
        timeout_seconds=shell_settings.timeout_seconds if shell_settings is not None else 90,
        warning_interval_seconds=(
            shell_settings.warning_interval_seconds if shell_settings is not None else 30
        ),
        working_directory=working_directory,
        config=settings,
        default_env=spec.env,
    )


def _build_docker_environment(
    spec: DockerEnvironmentSpec,
    *,
    workspace_root: Path,
) -> ShellEnvironment:
    from fast_agent.tools.docker_shell_environment import (
        DockerManagedShellEnvironment,
        DockerMount,
        DockerMountedEnvironment,
        DockerShellEnvironment,
    )

    if spec.container is not None:
        return DockerShellEnvironment(
            container=spec.container,
            container_cli=spec.container_cli,
            shell=spec.shell,
            cwd=spec.cwd,
            default_env=spec.env,
        )

    if spec.image is None:
        raise EnvironmentConfigError("Docker environment is missing image.")

    mounts = [
        DockerMount(
            source=_resolve_workspace_path(mount.source, workspace_root),
            target=mount.target,
            mode=mount.mode,
        )
        for mount in spec.mounts
    ]
    if (
        len(mounts) == 1
        and mounts[0].target == spec.cwd
        and mounts[0].mode == "rw"
    ):
        return DockerMountedEnvironment(
            image=spec.image,
            container_cli=spec.container_cli,
            workspace=mounts[0].source,
            target=mounts[0].target,
            shell=spec.shell,
            docker_args=spec.docker_args,
            default_env=spec.env,
        )

    return DockerManagedShellEnvironment(
        image=spec.image,
        container_cli=spec.container_cli,
        shell=spec.shell,
        cwd=spec.cwd,
        mounts=mounts,
        docker_args=spec.docker_args,
        default_env=spec.env,
    )


def _build_huggingface_environment(spec: HuggingFaceEnvironmentSpec) -> ShellEnvironment:
    from fast_agent.tools.huggingface_sandbox_environment import (
        HuggingFaceBucketMount,
        HuggingFaceSandboxEnvironment,
        HuggingFaceVolumeMount,
    )

    return HuggingFaceSandboxEnvironment(
        image=spec.image,
        flavor=spec.flavor,
        cwd=spec.cwd,
        bucket_mounts=tuple(
            HuggingFaceBucketMount(
                source=mount.source,
                mount_path=mount.mount_path,
                read_only=mount.read_only,
                path=mount.path,
            )
            for mount in spec.bucket_mounts
        ),
        volume_mounts=tuple(
            HuggingFaceVolumeMount(
                type=parsed.type,
                source=parsed.source,
                mount_path=parsed.mount_path,
                read_only=parsed.read_only,
                path=parsed.path,
                revision=parsed.revision,
            )
            for parsed in (mount.parsed() for mount in spec.volume_mounts)
        ),
        env=dict(spec.env),
        token=spec.token,
    )


def _build_custom_environment(spec: CustomEnvironmentSpec) -> ShellEnvironment:
    module_name, separator, class_name = spec.class_path.partition(":")
    if not separator or not module_name.strip() or not class_name.strip():
        raise EnvironmentConfigError(
            "Custom environment class must use 'module.path:ClassName' format."
        )

    try:
        module = importlib.import_module(module_name.strip())
    except Exception as exc:
        raise EnvironmentConfigError(
            f"Could not import custom environment module {module_name.strip()}: {exc}"
        ) from exc

    environment_class = getattr(module, class_name.strip(), None)
    if environment_class is None:
        raise EnvironmentConfigError(
            f"Custom environment class {class_name.strip()} was not found in {module_name.strip()}."
        )
    if not callable(environment_class):
        raise EnvironmentConfigError(
            f"Custom environment target {spec.class_path} is not callable."
        )
    return environment_class(**spec.params)


def _resolve_workspace_path(path: str, workspace_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_root / candidate).resolve()
__all__ = [
    "EnvironmentConfigError",
    "build_environment",
    "validate_environment_type",
]
