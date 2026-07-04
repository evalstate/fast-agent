from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from fast_agent.commands.model_capabilities import resolve_reasoning_effort
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec


@dataclass
class _LlmWithDefaultReasoning:
    reasoning_effort: ReasoningEffortSetting | None
    reasoning_effort_spec: ReasoningEffortSpec


def test_resolve_reasoning_effort_uses_spec_default_when_unset() -> None:
    default = ReasoningEffortSetting(kind="effort", value="medium")
    llm = _LlmWithDefaultReasoning(
        reasoning_effort=None,
        reasoning_effort_spec=ReasoningEffortSpec(
            kind="effort",
            allowed_efforts=["none", "low", "medium", "high", "xhigh"],
            default=default,
        ),
    )

    assert resolve_reasoning_effort(cast("Any", llm)) == default
