"""Minimal custom environment used by .fast-agent/fast-agent.yaml."""

from __future__ import annotations

from pathlib import Path

from fast_agent.core.logging.logger import get_logger
from fast_agent.tools.environment_factory import validate_environment_type
from fast_agent.tools.execution_environment import ShellRuntimeInfo
from fast_agent.tools.local_shell_executor import LocalEnvironment


class ExampleEnvironment(LocalEnvironment):
    def __init__(self, *, label: str = "custom-local") -> None:
        self._label = label
        super().__init__(
            logger=get_logger(__name__),
            working_directory=Path(__file__).parent,
        )

    def runtime_info(self) -> ShellRuntimeInfo:
        info = super().runtime_info()
        return ShellRuntimeInfo(
            name=info.name,
            path=info.path,
            kind="local",
            provider=self._label,
            environment_name=info.environment_name,
        )


validate_environment_type(ExampleEnvironment())
