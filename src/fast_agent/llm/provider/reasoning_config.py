from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fast_agent.llm.reasoning_effort import ReasoningEffortSetting

if TYPE_CHECKING:
    from pydantic import BaseModel

ProviderReasoningConfigValue = ReasoningEffortSetting | str | int | bool | None


def reasoning_setting_from_config(
    config: "BaseModel",
) -> tuple[ProviderReasoningConfigValue, bool]:
    fields = type(config).model_fields
    values = config.__dict__

    if "reasoning" in fields:
        raw_reasoning = cast("ProviderReasoningConfigValue", values.get("reasoning"))
        if raw_reasoning is not None:
            return raw_reasoning, False

    if "reasoning_effort" not in fields:
        return None, False

    raw_reasoning_effort = cast(
        "ProviderReasoningConfigValue",
        values.get("reasoning_effort"),
    )
    should_warn = (
        "reasoning_effort" in config.model_fields_set
        and raw_reasoning_effort != fields["reasoning_effort"].default
    )
    return raw_reasoning_effort, should_warn
