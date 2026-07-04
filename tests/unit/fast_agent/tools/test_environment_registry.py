from pathlib import Path

import pytest

from fast_agent.config import Settings
from fast_agent.core.logging.logger import get_logger
from fast_agent.tools.environment_config import LocalEnvironmentSpec
from fast_agent.tools.environment_registry import (
    EnvironmentRegistry,
    UnknownEnvironmentError,
    environment_name,
)
from fast_agent.tools.local_shell_executor import LocalShellExecutor


def test_environment_registry_lists_implicit_local() -> None:
    registry = EnvironmentRegistry(Settings())

    assert registry.names() == ("local",)
    assert registry.default_name == "local"


def test_environment_registry_builds_fresh_instances(tmp_path: Path) -> None:
    settings = Settings(
        environments={"workspace": LocalEnvironmentSpec(cwd=".")},
        default_environment="workspace",
    )
    registry = EnvironmentRegistry(settings, workspace_root=tmp_path)

    first = registry.build("workspace")
    second = registry.build("workspace")

    assert isinstance(first, LocalShellExecutor)
    assert isinstance(second, LocalShellExecutor)
    assert first is not second
    assert first.working_directory() == tmp_path
    assert environment_name(first) == "workspace"


def test_environment_registry_unknown_name_lists_valid_names() -> None:
    registry = EnvironmentRegistry(
        Settings(environments={"workspace": LocalEnvironmentSpec()}),
    )

    with pytest.raises(
        UnknownEnvironmentError,
        match="Unknown environment 'missing'. Valid environments: local, workspace",
    ):
        registry.build("missing")


def test_environment_registry_resolve_passes_instances_through() -> None:
    environment = LocalShellExecutor(logger=get_logger(__name__))
    registry = EnvironmentRegistry(Settings())

    assert registry.resolve(environment) is environment
    assert environment_name(environment) is None
