"""Protocols shared by command handlers and renderers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.skills.registry import SkillRegistry
    from fast_agent.types import PromptMessageExtended


@runtime_checkable
class WarningAwareAgent(Protocol):
    @property
    def warnings(self) -> list[str]: ...

    @property
    def skill_registry(self) -> "SkillRegistry | None": ...


@runtime_checkable
class InstructionAwareAgent(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def instruction(self) -> str | None: ...


@runtime_checkable
class ACPCommandAllowlistProvider(Protocol):
    @property
    def acp_session_commands_allowlist(self) -> set[str] | None: ...


@runtime_checkable
class ParallelAgentProtocol(Protocol):
    @property
    def fan_out_agents(self) -> list["AgentProtocol"] | None: ...

    @property
    def fan_in_agent(self) -> "AgentProtocol | None": ...


@runtime_checkable
class HfDisplayInfoProvider(Protocol):
    def get_hf_display_info(self) -> dict[str, Any]: ...


@runtime_checkable
class HistoryEditableAgent(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def message_history(self) -> list["PromptMessageExtended"]: ...

    @property
    def usage_accumulator(self): ...

    def load_message_history(self, messages: list["PromptMessageExtended"] | None) -> None: ...

    def pop_last_message(self) -> "PromptMessageExtended | None": ...

    def clear(self, *, clear_prompts: bool = False) -> None: ...


@runtime_checkable
class TemplateMessageProvider(Protocol):
    @property
    def template_messages(self) -> list["PromptMessageExtended"]: ...
