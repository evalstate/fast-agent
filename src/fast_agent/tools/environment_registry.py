"""Registry for resolving named execution environments from settings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from fast_agent.tools.environment_config import EnvironmentSpecModel, LocalEnvironmentSpec
from fast_agent.tools.environment_factory import build_environment, validate_environment_type
from fast_agent.tools.execution_environment import ShellEnvironment

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fast_agent.config import Settings


EnvironmentSelection = ShellEnvironment | str | None
_ENVIRONMENT_NAMES: WeakKeyDictionary[object, str] = WeakKeyDictionary()


class EnvironmentRegistry:
    """Build fresh shell environment instances from named settings entries."""

    def __init__(self, settings: "Settings", *, workspace_root: Path | None = None) -> None:
        self._settings = settings
        self._workspace_root = workspace_root or Path.cwd().resolve()

    @property
    def default_name(self) -> str:
        return self._settings.default_environment

    def names(self) -> tuple[str, ...]:
        return tuple(sorted({"local", *self._settings.environments}))

    def spec(self, name: str) -> EnvironmentSpecModel:
        if name == "local":
            return self._settings.environments.get("local", LocalEnvironmentSpec())
        spec = self._settings.environments.get(name)
        if spec is None:
            raise UnknownEnvironmentError(name, self.names())
        return spec

    def build(self, name: str | None = None) -> ShellEnvironment:
        environment_name = self.default_name if name is None else name
        environment = build_environment(
            self.spec(environment_name),
            settings=self._settings,
            workspace_root=self._workspace_root,
            name=environment_name,
        )
        register_environment_name(environment, environment_name)
        return environment

    def resolve(self, selection: EnvironmentSelection) -> ShellEnvironment:
        if selection is None:
            return self.build()
        if isinstance(selection, str):
            return self.build(selection)
        validate_environment_type(selection)
        return selection


class UnknownEnvironmentError(ValueError):
    """Raised when a requested named environment is not configured."""

    def __init__(self, name: str, valid_names: "Iterable[str]") -> None:
        choices = ", ".join(valid_names)
        super().__init__(f"Unknown environment '{name}'. Valid environments: {choices}")


def register_environment_name(environment: object, name: str) -> None:
    try:
        _ENVIRONMENT_NAMES[environment] = name
    except TypeError:
        return


def environment_name(environment: object) -> str | None:
    try:
        return _ENVIRONMENT_NAMES.get(environment)
    except TypeError:
        return None


__all__ = [
    "EnvironmentRegistry",
    "EnvironmentSelection",
    "UnknownEnvironmentError",
    "environment_name",
    "register_environment_name",
]
