import sys
from pathlib import Path

import pytest

from fast_agent.tools.docker_shell_environment import (
    DockerManagedShellEnvironment,
    DockerMountedEnvironment,
)
from fast_agent.tools.environment_config import (
    CustomEnvironmentSpec,
    DockerEnvironmentSpec,
    EnvironmentMountSpec,
    HuggingFaceEnvironmentSpec,
    LocalEnvironmentSpec,
)
from fast_agent.tools.environment_factory import EnvironmentConfigError, build_environment
from fast_agent.tools.huggingface_sandbox_environment import HuggingFaceSandboxEnvironment
from fast_agent.tools.local_shell_executor import LocalShellExecutor


def test_build_local_environment_resolves_cwd_against_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subdir = workspace / "subdir"
    subdir.mkdir()

    environment = build_environment(
        LocalEnvironmentSpec(cwd="subdir"),
        workspace_root=workspace,
    )

    assert isinstance(environment, LocalShellExecutor)
    assert environment.working_directory() == subdir


def test_build_docker_environment_resolves_mount_sources_against_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = build_environment(
        DockerEnvironmentSpec(
            type="docker",
            image="ubuntu:24.04",
            mounts=[EnvironmentMountSpec(source=".", target="/workspace", mode="ro")],
            docker_args=["--network=none"],
        ),
        workspace_root=workspace,
    )

    assert isinstance(environment, DockerManagedShellEnvironment)
    assert environment._mounts[0].source == workspace  # noqa: SLF001
    assert environment._docker_args == ("--network=none",)  # noqa: SLF001


def test_build_docker_mounted_environment_keeps_file_tools_with_docker_args(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = build_environment(
        DockerEnvironmentSpec(
            type="docker",
            image="ubuntu:24.04",
            cwd="/workspace",
            mounts=[EnvironmentMountSpec(source=".", target="/workspace", mode="rw")],
            docker_args=["--network=none"],
        ),
        workspace_root=workspace,
    )

    assert isinstance(environment, DockerMountedEnvironment)
    assert environment._mounts[0].source == workspace  # noqa: SLF001
    assert environment._docker_args == ("--network=none",)  # noqa: SLF001


def test_build_docker_environment_applies_container_cli(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = build_environment(
        DockerEnvironmentSpec(
            type="docker",
            image="ubuntu:24.04",
            container_cli="wslc",
            mounts=[EnvironmentMountSpec(source=".", target="/workspace", mode="rw")],
        ),
        workspace_root=workspace,
    )

    assert environment.runtime_info().provider == "wslc"


def test_build_huggingface_environment_applies_hf_volume_mounts(tmp_path: Path) -> None:
    environment = build_environment(
        HuggingFaceEnvironmentSpec.model_validate(
            {
                "type": "huggingface",
                "volume_mounts": [
                    "hf://datasets/org/data@main/train:/data:ro",
                    "hf://buckets/org/bucket/output:/workspace:rw",
                ],
            }
        ),
        workspace_root=tmp_path,
    )

    assert isinstance(environment, HuggingFaceSandboxEnvironment)
    assert [mount.type for mount in environment._volume_mounts] == ["dataset", "bucket"]  # noqa: SLF001
    assert [mount.source for mount in environment._volume_mounts] == ["org/data", "org/bucket"]  # noqa: SLF001
    assert [mount.path for mount in environment._volume_mounts] == ["train", "output"]  # noqa: SLF001
    assert [mount.revision for mount in environment._volume_mounts] == ["main", None]  # noqa: SLF001
    assert [mount.read_only for mount in environment._volume_mounts] == [True, False]  # noqa: SLF001


def test_build_custom_environment_imports_module_path(tmp_path: Path) -> None:
    module_path = tmp_path / "custom_env.py"
    module_path.write_text(
        "\n".join(
            [
                "from fast_agent.tools.local_shell_executor import LocalShellExecutor",
                "from fast_agent.core.logging.logger import get_logger",
                "class CustomEnv(LocalShellExecutor):",
                "    def __init__(self, cwd):",
                "        super().__init__(logger=get_logger(__name__), working_directory=cwd)",
            ]
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        environment = build_environment(
            CustomEnvironmentSpec.model_validate(
                {
                    "type": "custom",
                    "class": "custom_env:CustomEnv",
                    "params": {"cwd": tmp_path},
                }
            ),
            workspace_root=tmp_path,
        )
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("custom_env", None)

    assert isinstance(environment, LocalShellExecutor)
    assert environment.working_directory() == tmp_path


def test_build_custom_environment_rejects_non_shell_type(tmp_path: Path) -> None:
    module_path = tmp_path / "bad_env.py"
    module_path.write_text("class NotAnEnvironment:\n    pass\n", encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(EnvironmentConfigError, match="does not implement ShellEnvironment"):
            build_environment(
                CustomEnvironmentSpec.model_validate(
                    {"type": "custom", "class": "bad_env:NotAnEnvironment"}
                ),
                workspace_root=tmp_path,
                name="bad",
            )
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("bad_env", None)
