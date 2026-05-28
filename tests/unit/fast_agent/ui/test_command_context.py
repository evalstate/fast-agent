from __future__ import annotations

from typing import Any, cast

from fast_agent.commands.context import StaticAgentProvider
from fast_agent.ui.interactive.command_context import build_command_context


class _LegacyProvider(StaticAgentProvider):
    def __init__(self) -> None:
        super().__init__({"main": object()})
        self._noenv_mode = True


def test_build_command_context_reads_legacy_noenv_storage() -> None:
    context = build_command_context(cast("Any", _LegacyProvider()), "main")

    assert context.noenv is True
    assert context.current_agent_name == "main"
