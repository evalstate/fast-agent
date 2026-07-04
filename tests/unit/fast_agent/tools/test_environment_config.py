import pytest
from pydantic import TypeAdapter, ValidationError

from fast_agent.config import Settings
from fast_agent.tools.environment_config import (
    DockerEnvironmentSpec,
    EnvironmentSpec,
    HuggingFaceEnvironmentSpec,
    parse_huggingface_volume_mount,
)


def test_docker_environment_requires_image_or_container() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        DockerEnvironmentSpec(type="docker")

    with pytest.raises(ValidationError, match="exactly one"):
        DockerEnvironmentSpec(type="docker", image="ubuntu:24.04", container="ci")


def test_docker_args_reject_volume_and_lifecycle_flags() -> None:
    for docker_args in (
        ["--network=none", "--mount=type=bind,source=.,target=/workspace"],
        ["--network=none", "-v=.:/workspace"],
    ):
        with pytest.raises(ValidationError, match="use mounts"):
            DockerEnvironmentSpec(
                type="docker",
                image="ubuntu:24.04",
                docker_args=docker_args,
            )


def test_docker_args_accept_non_mount_flags() -> None:
    spec = DockerEnvironmentSpec(
        type="docker",
        image="ubuntu:24.04",
        docker_args=["--network=none"],
    )

    assert spec.docker_args == ["--network=none"]


def test_docker_environment_accepts_container_cli() -> None:
    spec = DockerEnvironmentSpec(type="docker", image="ubuntu:24.04", container_cli="wslc")

    assert spec.container_cli == "wslc"


def test_environment_spec_forbids_unknown_fields() -> None:
    adapter = TypeAdapter(EnvironmentSpec)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        adapter.validate_python(
            {
                "type": "docker",
                "image": "ubuntu:24.04",
                "moutns": [],
            }
        )


def test_settings_default_environment_must_be_configured() -> None:
    with pytest.raises(ValidationError, match="default_environment 'ubuntu' is not configured"):
        Settings(default_environment="ubuntu")


def test_settings_accepts_implicit_local_default() -> None:
    settings = Settings()

    assert settings.default_environment == "local"
    assert settings.environments == {}


def test_huggingface_volume_mount_parses_bucket_uri() -> None:
    parsed = parse_huggingface_volume_mount("hf://buckets/org/bucket/sub/dir:/workspace:rw")

    assert parsed.type == "bucket"
    assert parsed.source == "org/bucket"
    assert parsed.path == "sub/dir"
    assert parsed.mount_path == "/workspace"
    assert parsed.read_only is False
    assert parsed.revision is None


def test_huggingface_volume_mount_parses_repo_uri_with_revision() -> None:
    parsed = parse_huggingface_volume_mount("hf://datasets/org/data@main/train:/data:ro")

    assert parsed.type == "dataset"
    assert parsed.source == "org/data"
    assert parsed.revision == "main"
    assert parsed.path == "train"
    assert parsed.mount_path == "/data"
    assert parsed.read_only is True


def test_huggingface_environment_accepts_volume_mount_uri_strings() -> None:
    spec = HuggingFaceEnvironmentSpec.model_validate(
        {
            "type": "huggingface",
            "volume_mounts": [
                "hf://org/model:/models",
                {"uri": "hf://buckets/org/bucket:/workspace:rw"},
            ],
        }
    )

    assert [mount.uri for mount in spec.volume_mounts] == [
        "hf://org/model:/models",
        "hf://buckets/org/bucket:/workspace:rw",
    ]


def test_huggingface_environment_rejects_invalid_volume_mount_uri() -> None:
    with pytest.raises(ValidationError, match="Missing mount path"):
        HuggingFaceEnvironmentSpec.model_validate(
            {
                "type": "huggingface",
                "volume_mounts": ["hf://buckets/org/bucket"],
            }
        )
