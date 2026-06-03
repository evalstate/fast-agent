from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec
from fast_agent.llm.text_verbosity import TextVerbositySpec
from fast_agent.ui.prompt.input import _build_cycle_callbacks

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp


class _ProviderStub:
    def __init__(self, agent: object) -> None:
        self._agent_obj = agent

    def _agent(self, _name: str) -> object:
        return self._agent_obj


class _AgentStub:
    def __init__(self, llm: object) -> None:
        self.llm = llm


class _MissingEnabledWebToolLlm:
    web_search_supported = True
    web_fetch_supported = True

    def __init__(self) -> None:
        self.web_search_values: list[bool | None] = []
        self.web_fetch_values: list[bool | None] = []

    def set_web_search_enabled(self, value: bool | None) -> None:
        self.web_search_values.append(value)

    def set_web_fetch_enabled(self, value: bool | None) -> None:
        self.web_fetch_values.append(value)


class _MissingServiceTierValueLlm:
    service_tier_supported = True
    available_service_tiers = ("fast",)

    def __init__(self) -> None:
        self.service_tier_values: list[str | None] = []

    def set_service_tier(self, value: str | None) -> None:
        self.service_tier_values.append(value)


class _RejectingServiceTierLlm(_MissingServiceTierValueLlm):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def set_service_tier(self, value: str | None) -> None:
        self.attempts += 1
        raise ValueError(f"unsupported tier: {value}")


class _MissingReasoningValueLlm:
    reasoning_effort_spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high"],
        default=ReasoningEffortSetting(kind="effort", value="medium"),
    )

    def __init__(self) -> None:
        self.reasoning_values: list[ReasoningEffortSetting | None] = []

    def set_reasoning_effort(self, value: ReasoningEffortSetting | None) -> None:
        self.reasoning_values.append(value)


class _MissingTextVerbosityValueLlm:
    text_verbosity_spec = TextVerbositySpec(allowed=("low", "medium", "high"), default="medium")

    def __init__(self) -> None:
        self.verbosity_values: list[str | None] = []

    def set_text_verbosity(self, value: str | None) -> None:
        self.verbosity_values.append(value)


def test_cycle_web_tool_callbacks_use_safe_enabled_resolvers() -> None:
    llm = _MissingEnabledWebToolLlm()
    callbacks = _build_cycle_callbacks(
        agent_name="agent",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub(llm))),
    )

    callbacks.on_cycle_web_search()
    callbacks.on_cycle_web_fetch()

    assert llm.web_search_values == [True]
    assert llm.web_fetch_values == [True]


def test_cycle_service_tier_callback_uses_safe_capability_resolvers() -> None:
    llm = _MissingServiceTierValueLlm()
    callbacks = _build_cycle_callbacks(
        agent_name="agent",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub(llm))),
    )

    callbacks.on_cycle_service_tier()

    assert llm.service_tier_values == ["fast"]


def test_cycle_service_tier_callback_ignores_rejected_setting() -> None:
    llm = _RejectingServiceTierLlm()
    callbacks = _build_cycle_callbacks(
        agent_name="agent",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub(llm))),
    )

    callbacks.on_cycle_service_tier()

    assert llm.attempts == 1


def test_cycle_reasoning_callback_uses_safe_capability_resolvers() -> None:
    llm = _MissingReasoningValueLlm()
    callbacks = _build_cycle_callbacks(
        agent_name="agent",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub(llm))),
    )

    callbacks.on_cycle_reasoning()

    assert llm.reasoning_values == [ReasoningEffortSetting(kind="effort", value="high")]


def test_cycle_verbosity_callback_uses_safe_capability_resolvers() -> None:
    llm = _MissingTextVerbosityValueLlm()
    callbacks = _build_cycle_callbacks(
        agent_name="agent",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub(llm))),
    )

    callbacks.on_cycle_verbosity()

    assert llm.verbosity_values == ["high"]
