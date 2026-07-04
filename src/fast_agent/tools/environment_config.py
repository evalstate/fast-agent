"""Pydantic models for named execution environment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EnvironmentMountSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        description="Host workspace path to bind mount. Relative paths resolve against the workspace root.",
        examples=["."],
    )
    target: str = Field(
        description="Absolute path inside the Docker container.",
        examples=["/workspace"],
    )
    mode: Literal["ro", "rw"] = Field(
        default="rw",
        description="Docker bind mount access mode: read-only (`ro`) or read-write (`rw`).",
        examples=["rw"],
    )


class HuggingFaceBucketMountSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        description="Hugging Face bucket identifier in `namespace/name` form.",
        examples=["username/my-bucket"],
    )
    mount_path: str = Field(
        description="Absolute path where the bucket is mounted inside the sandbox.",
        examples=["/workspace"],
    )
    read_only: bool = Field(
        default=False,
        description="Whether the bucket mount is read-only.",
        examples=[False],
    )
    path: str | None = Field(
        default=None,
        description="Optional subfolder prefix inside the bucket to mount.",
        examples=["subdir"],
    )


@dataclass(frozen=True, slots=True)
class ParsedHuggingFaceVolumeMount:
    type: Literal["bucket", "model", "dataset", "space"]
    source: str
    mount_path: str
    read_only: bool | None = None
    path: str | None = None
    revision: str | None = None


class HuggingFaceVolumeMountSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str = Field(
        description=(
            "Hugging Face mount URI: `hf://[models|datasets|spaces|buckets]/"
            "namespace/name[/path]:/mount/path[:ro|:rw]`. The type defaults to models."
        ),
        examples=[
            "hf://buckets/username/my-bucket:/workspace:rw",
            "hf://datasets/username/reference-data:/data:ro",
        ],
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_uri_string(cls, value: object) -> object:
        if isinstance(value, str):
            return {"uri": value}
        return value

    @field_validator("uri")
    @classmethod
    def _validate_hf_mount_uri(cls, value: str) -> str:
        parse_huggingface_volume_mount(value)
        return value

    def parsed(self) -> ParsedHuggingFaceVolumeMount:
        return parse_huggingface_volume_mount(self.uri)


class LocalEnvironmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["local"] = Field(default="local", description="Use the host local shell.")
    cwd: str | None = Field(
        default=None,
        description="Working directory for local shell and file tools. Relative paths resolve against the workspace root.",
        examples=["."],
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables applied to shell execution.",
        examples=[{"PYTHONUNBUFFERED": "1"}],
    )


class DockerEnvironmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["docker"] = Field(description="Run shell commands in Docker or a Docker-compatible CLI.")
    image: str | None = Field(
        default=None,
        description="Container image to start. Provide exactly one of `image` or `container`.",
        examples=["ubuntu:24.04"],
    )
    container: str | None = Field(
        default=None,
        description="Existing container name or ID to execute in. Provide exactly one of `image` or `container`.",
        examples=["fast-agent-ci"],
    )
    container_cli: str = Field(
        default="docker",
        description="Executable used for container operations, for example `docker` or `wslc`.",
        examples=["docker", "wslc"],
    )
    shell: str = Field(
        default="bash",
        description="Shell executable used inside the container.",
        examples=["bash", "sh", "pwsh"],
    )
    cwd: str = Field(
        default="/workspace",
        description="Working directory inside the container.",
        examples=["/workspace"],
    )
    mounts: list[EnvironmentMountSpec] = Field(
        default_factory=list,
        description="Docker bind mounts. Use this instead of volume flags in `docker_args`.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables applied to shell execution.",
    )
    docker_args: list[str] = Field(
        default_factory=list,
        description="Extra container creation arguments. Volume and lifecycle flags are rejected; use `mounts` for bind mounts.",
        examples=[["--network=none"]],
    )

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "DockerEnvironmentSpec":
        if bool(self.image) == bool(self.container):
            raise ValueError("Provide exactly one of 'image' or 'container'.")
        return self

    @field_validator("docker_args")
    @classmethod
    def _no_volume_or_lifecycle_flags(cls, args: list[str]) -> list[str]:
        forbidden: list[str] = []
        blocked_exact = {"-v", "--volume", "--mount", "--name", "--rm"}
        blocked_prefixes = ("-v=", "--volume=", "--mount=", "--name=")
        for arg in args:
            if arg in blocked_exact or arg.startswith(blocked_prefixes):
                forbidden.append(arg)

        if forbidden:
            flags = ", ".join(forbidden)
            raise ValueError(
                "docker_args cannot include volume or lifecycle flags "
                f"({flags}); use mounts: for bind mounts."
            )
        return args


class HuggingFaceEnvironmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["huggingface"] = Field(description="Run shell commands in a Hugging Face Sandbox.")
    image: str = Field(
        default="python:3.12",
        description="Sandbox container image.",
        examples=["python:3.12"],
    )
    flavor: str = Field(
        default="cpu-basic",
        description="Hugging Face Sandbox hardware flavor.",
        examples=["cpu-basic"],
    )
    cwd: str = Field(
        default="/workspace",
        description="Working directory inside the sandbox.",
        examples=["/workspace"],
    )
    bucket_mounts: list[HuggingFaceBucketMountSpec] = Field(
        default_factory=list,
        description="Legacy bucket-only mount shorthand. Prefer `volume_mounts` for new config.",
    )
    volume_mounts: list[HuggingFaceVolumeMountSpec] = Field(
        default_factory=list,
        description="Hugging Face Sandbox volume mounts using `hf://...:/mount/path[:ro|:rw]` syntax.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables applied to shell execution.",
    )
    token: str | None = Field(
        default=None,
        description="Hugging Face token. Prefer `fast-agent.secrets.yaml` or `${HF_TOKEN}`.",
        examples=["${HF_TOKEN}"],
    )


class CustomEnvironmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: Literal["custom"] = Field(description="Load a custom ShellEnvironment class.")
    class_path: str = Field(
        alias="class",
        description="Import path in `module.path:ClassName` format.",
        examples=["mycompany.envs:KubernetesEnvironment"],
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed to the custom environment class.",
    )


EnvironmentSpec = Annotated[
    LocalEnvironmentSpec
    | DockerEnvironmentSpec
    | HuggingFaceEnvironmentSpec
    | CustomEnvironmentSpec,
    Field(discriminator="type"),
]

EnvironmentSpecModel = (
    LocalEnvironmentSpec
    | DockerEnvironmentSpec
    | HuggingFaceEnvironmentSpec
    | CustomEnvironmentSpec
)

_HF_VOLUME_TYPE_PREFIXES: dict[str, Literal["bucket", "model", "dataset", "space"]] = {
    "buckets": "bucket",
    "models": "model",
    "datasets": "dataset",
    "spaces": "space",
}
_HF_VOLUME_TYPE_SINGULARS = {
    "bucket": "buckets",
    "model": "models",
    "dataset": "datasets",
    "space": "spaces",
}


def parse_huggingface_volume_mount(uri: str) -> ParsedHuggingFaceVolumeMount:
    """Parse ``hf://[TYPE/]SOURCE[/PATH]:/MOUNT_PATH[:ro|:rw]`` volume mounts."""
    raw = uri.strip()
    try:
        from huggingface_hub.utils import parse_hf_mount
    except ImportError:
        return _parse_huggingface_volume_mount_fallback(raw)

    try:
        mount = parse_hf_mount(raw)
    except Exception as exc:
        raise ValueError(str(exc)) from exc

    mount_type = mount.source.type
    if mount_type not in {"bucket", "model", "dataset", "space"}:
        raise ValueError(f"Unsupported Hugging Face volume type '{mount_type}'.")
    return ParsedHuggingFaceVolumeMount(
        type=cast("Literal['bucket', 'model', 'dataset', 'space']", mount_type),
        source=mount.source.id,
        mount_path=mount.mount_path,
        read_only=mount.read_only,
        path=mount.source.path_in_repo or None,
        revision=mount.source.revision or None,
    )


def _parse_huggingface_volume_mount_fallback(uri: str) -> ParsedHuggingFaceVolumeMount:
    raw = uri.strip()
    if not raw.startswith("hf://"):
        raise ValueError("Hugging Face volume mounts must start with 'hf://'.")
    body = raw.removeprefix("hf://")
    if not body:
        raise ValueError("Hugging Face volume mount is missing a source.")

    location, mount_path, read_only = _split_huggingface_mount_target(body)
    mount_type, location = _split_huggingface_mount_type(location)
    source, revision, path = _split_huggingface_mount_source(location)
    return ParsedHuggingFaceVolumeMount(
        type=mount_type,
        source=source,
        mount_path=mount_path,
        read_only=read_only,
        path=path,
        revision=revision,
    )


def _split_huggingface_mount_target(body: str) -> tuple[str, str, bool | None]:
    if body.endswith(":ro"):
        read_only: bool | None = True
        body = body.removesuffix(":ro")
    elif body.endswith(":rw"):
        read_only = False
        body = body.removesuffix(":rw")
    else:
        read_only = None

    index = body.rfind(":/")
    if index == -1:
        raise ValueError(
            "Hugging Face volume mount is missing a mount path; expected "
            "'hf://...:/mount/path'."
        )

    location = body[:index]
    mount_path = body[index + 1 :]
    if not location:
        raise ValueError("Hugging Face volume mount is missing a source before the mount path.")
    if not mount_path.startswith("/"):
        raise ValueError("Hugging Face volume mount path must be an absolute path.")
    return location, mount_path, read_only


def _split_huggingface_mount_type(
    location: str,
) -> tuple[Literal["bucket", "model", "dataset", "space"], str]:
    prefix, separator, remainder = location.partition("/")
    if not separator:
        if prefix in _HF_VOLUME_TYPE_PREFIXES:
            raise ValueError(f"Hugging Face volume mount is missing an identifier after '{prefix}'.")
        if prefix in _HF_VOLUME_TYPE_SINGULARS:
            plural = _HF_VOLUME_TYPE_SINGULARS[prefix]
            raise ValueError(f"Hugging Face volume type must be plural; use 'hf://{plural}/...'.")
        return "model", location

    mount_type = _HF_VOLUME_TYPE_PREFIXES.get(prefix)
    if mount_type is not None:
        if not remainder:
            raise ValueError(f"Hugging Face volume mount is missing an identifier after '{prefix}'.")
        return mount_type, remainder
    if prefix in _HF_VOLUME_TYPE_SINGULARS:
        plural = _HF_VOLUME_TYPE_SINGULARS[prefix]
        raise ValueError(f"Hugging Face volume type must be plural; use 'hf://{plural}/...'.")
    return "model", location


def _split_huggingface_mount_source(location: str) -> tuple[str, str | None, str | None]:
    parts = [part for part in location.split("/") if part]
    if len(parts) < 2:
        raise ValueError("Hugging Face volume source must include namespace and name.")

    name, revision = _split_huggingface_revision(parts[1])
    source = f"{parts[0]}/{name}"
    path_parts = parts[2:]
    path = "/".join(path_parts) or None
    return source, revision, path


def _split_huggingface_revision(segment: str) -> tuple[str, str | None]:
    name, separator, revision = segment.partition("@")
    if not separator:
        return segment, None
    if not name or not revision:
        raise ValueError("Hugging Face volume revision must use '<name>@<revision>'.")
    return name, revision


__all__ = [
    "CustomEnvironmentSpec",
    "DockerEnvironmentSpec",
    "EnvironmentMountSpec",
    "EnvironmentSpec",
    "EnvironmentSpecModel",
    "HuggingFaceBucketMountSpec",
    "HuggingFaceEnvironmentSpec",
    "HuggingFaceVolumeMountSpec",
    "LocalEnvironmentSpec",
    "ParsedHuggingFaceVolumeMount",
    "parse_huggingface_volume_mount",
]
