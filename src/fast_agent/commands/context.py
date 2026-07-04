"""Context and IO abstraction for shared command handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from fast_agent.config import Settings, get_settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable, Mapping, Sequence
    from pathlib import Path

    from fast_agent.commands.results import CommandMessage
    from fast_agent.commands.session_summaries import SessionListSummary
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.session import ResumeSessionAgentsResult, Session, SessionInfo, SessionManager
    from fast_agent.session.identity import SessionStoreScope
    from fast_agent.types import PromptMessageExtended


class CommandIO(Protocol):
    """UI/transport specific IO operations used by shared command handlers."""

    async def emit(self, message: CommandMessage) -> None:
        """Display a message in the current UI."""

    async def prompt_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        """Prompt for free-form text input."""

    async def prompt_selection(
        self,
        prompt: str,
        *,
        options: Sequence[str],
        allow_cancel: bool = False,
        default: str | None = None,
    ) -> str | None:
        """Prompt for a selection from a list of options."""

    async def prompt_model_selection(
        self,
        *,
        initial_provider: str | None = None,
        default_model: str | None = None,
    ) -> str | None:
        """Prompt for a model selection and return the selected model token/spec."""

    async def prompt_argument(
        self,
        arg_name: str,
        *,
        description: str | None = None,
        required: bool = True,
    ) -> str | None:
        """Prompt for a prompt argument value."""

    async def display_history_turn(
        self,
        agent_name: str,
        turn: list[PromptMessageExtended],
        *,
        turn_index: int | None = None,
        total_turns: int | None = None,
    ) -> None:
        """Display a history turn with rich formatting."""

    async def display_history_overview(
        self,
        agent_name: str,
        history: list[PromptMessageExtended],
        usage: "UsageAccumulator" | None = None,
    ) -> None:
        """Display a conversation history overview."""

    async def display_usage_report(self, agents: dict[str, object]) -> None:
        """Display a usage report for the provided agents."""

    async def display_system_prompt(
        self,
        agent_name: str,
        system_prompt: str,
        *,
        server_count: int = 0,
    ) -> None:
        """Display the system prompt for the active agent."""


class AgentProvider(Protocol):
    """Minimum provider surface for shared command handlers (expand as needed)."""

    def _agent(self, name: str): ...

    def resolve_target_agent_name(self, agent_name: str | None = None) -> str | None: ...

    def visible_agent_names(self, *, force_include: str | None = None) -> Iterable[str]: ...

    def registered_agent_names(self) -> Iterable[str]: ...

    def registered_agents(self) -> dict[str, object]: ...

    def list_prompts(
        self,
        namespace: str | None,
        agent_name: str | None = None,
    ) -> Awaitable[object]: ...


class StaticAgentProvider:
    """Minimal mapping-backed agent provider for shared command contexts."""

    def __init__(self, agents: Mapping[str, object] | None = None) -> None:
        self._agents = dict(agents or {})

    def _agent(self, name: str) -> object:
        return self._agents[name]

    def resolve_target_agent_name(self, agent_name: str | None = None) -> str | None:
        return agent_name

    def visible_agent_names(self, *, force_include: str | None = None) -> Iterable[str]:
        del force_include
        return list(self._agents.keys())

    def registered_agent_names(self) -> Iterable[str]:
        return list(self._agents.keys())

    def registered_agents(self) -> dict[str, object]:
        return dict(self._agents)

    async def list_prompts(
        self,
        namespace: str | None,
        agent_name: str | None = None,
    ) -> object:
        del namespace, agent_name
        return {}


class SessionCommandRuntime(Protocol):
    """Explicit session capability supplied by an interactive/runtime boundary."""

    def resolve_manager(self) -> "SessionManager": ...

    def current_session_id(self) -> str | None: ...

    def active_session_id(self, *, fallback_session_id: str | None = None) -> str | None: ...

    def build_list_summary(self, *, show_help: bool = False) -> "SessionListSummary": ...

    def create_session(
        self,
        *,
        session_name: str | None,
        session_id: str | None = None,
        replace_existing: bool = False,
        metadata: dict[str, str] | None = None,
    ) -> "Session": ...

    def list_sessions(self) -> list["SessionInfo"]: ...

    def delete_session(self, session_id: str) -> bool: ...

    def resolve_session_name(self, name: str | None) -> str | None: ...

    def get_session(self, session_id: str) -> "Session | None": ...

    async def resume_agents(
        self,
        agents: Mapping[str, "AgentProtocol"],
        session_id: str | None,
        *,
        fallback_agent_name: str | None,
    ) -> "ResumeSessionAgentsResult | None": ...

    def title_session(self, title: str, *, session_id: str | None = None) -> "Session | None": ...

    def fork_current_session(self, *, title: str | None = None) -> "Session | None": ...


async def noninteractive_prompt_selection(
    prompt: str,
    *,
    options: Sequence[str],
    allow_cancel: bool = False,
    default: str | None = None,
) -> str | None:
    """Default no-op selection prompt for non-interactive command IO."""
    del prompt, options, allow_cancel, default
    return None


async def noninteractive_prompt_model_selection(
    *,
    initial_provider: str | None = None,
    default_model: str | None = None,
) -> str | None:
    """Default no-op model picker for non-interactive command IO."""
    del initial_provider, default_model
    return None


async def noninteractive_prompt_argument(
    arg_name: str,
    *,
    description: str | None = None,
    required: bool = True,
) -> str | None:
    """Default no-op prompt-argument handler for non-interactive command IO."""
    del arg_name, description, required
    return None


class NonInteractiveCommandIOBase(CommandIO):
    """Shared no-op prompt/display behavior for non-interactive command IO."""

    async def prompt_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        del prompt
        return default if allow_empty else None

    async def prompt_selection(
        self,
        prompt: str,
        *,
        options: Sequence[str],
        allow_cancel: bool = False,
        default: str | None = None,
    ) -> str | None:
        return await noninteractive_prompt_selection(
            prompt,
            options=options,
            allow_cancel=allow_cancel,
            default=default,
        )

    async def prompt_model_selection(
        self,
        *,
        initial_provider: str | None = None,
        default_model: str | None = None,
    ) -> str | None:
        return await noninteractive_prompt_model_selection(
            initial_provider=initial_provider,
            default_model=default_model,
        )

    async def prompt_argument(
        self,
        arg_name: str,
        *,
        description: str | None = None,
        required: bool = True,
    ) -> str | None:
        return await noninteractive_prompt_argument(
            arg_name,
            description=description,
            required=required,
        )

    async def display_history_turn(
        self,
        agent_name: str,
        turn: list[PromptMessageExtended],
        *,
        turn_index: int | None = None,
        total_turns: int | None = None,
    ) -> None:
        del agent_name, turn, turn_index, total_turns

    async def display_history_overview(
        self,
        agent_name: str,
        history: list[PromptMessageExtended],
        usage: "UsageAccumulator" | None = None,
    ) -> None:
        del agent_name, history, usage

    async def display_usage_report(self, agents: dict[str, object]) -> None:
        del agents

    async def display_system_prompt(
        self,
        agent_name: str,
        system_prompt: str,
        *,
        server_count: int = 0,
    ) -> None:
        del agent_name, system_prompt, server_count


_SESSION_SKILL_SOURCE_OVERRIDES: dict[tuple[str, str, str], str] = {}


@dataclass(slots=True)
class CommandContext:
    """Context passed to shared command handlers."""

    agent_provider: AgentProvider
    current_agent_name: str
    io: CommandIO
    settings: Settings | None = None
    no_home: bool = False
    acp_session_id: str | None = None
    session_cwd: Path | None = None
    session_store_scope: SessionStoreScope = "workspace"
    session_store_cwd: Path | None = None
    session_manager: "SessionManager | None" = None
    session_runtime: SessionCommandRuntime | None = None
    skill_source_overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.no_home:
            if self.session_manager is not None or self.session_runtime is not None:
                raise ValueError("no_home command contexts cannot enable sessions.")
            return
        if self.session_runtime is not None:
            return
        if self.session_manager is None:
            return
        from fast_agent.commands.session_runtime import SessionManagerCommandRuntime

        self.session_runtime = SessionManagerCommandRuntime(
            explicit_manager=self.session_manager,
            session_cwd=self.session_cwd,
            session_store_scope=self.session_store_scope,
            session_store_cwd=self.session_store_cwd,
            settings=self.settings,
        )

    @property
    def sessions_enabled(self) -> bool:
        """Return True when this context carries an explicit session capability."""
        return self.session_runtime is not None

    def resolve_settings(self) -> Settings:
        return self.settings or get_settings()

    def active_skill_source(self, agent_name: str) -> str | None:
        return self.skill_source_overrides.get(agent_name) or _SESSION_SKILL_SOURCE_OVERRIDES.get(
            self._skill_source_override_key(agent_name)
        )

    def set_active_skill_source(self, agent_name: str, source: str) -> None:
        self.skill_source_overrides[agent_name] = source
        _SESSION_SKILL_SOURCE_OVERRIDES[self._skill_source_override_key(agent_name)] = source

    def clear_active_skill_source(self, agent_name: str) -> None:
        self.skill_source_overrides.pop(agent_name, None)
        _SESSION_SKILL_SOURCE_OVERRIDES.pop(self._skill_source_override_key(agent_name), None)

    def _skill_source_override_key(self, agent_name: str) -> tuple[str, str, str]:
        if self.acp_session_id is not None:
            return ("acp", self.acp_session_id, agent_name)
        return ("provider", str(id(self.agent_provider)), agent_name)

    def resolve_session_manager(self) -> "SessionManager":
        if self.session_runtime is not None:
            return self.session_runtime.resolve_manager()
        raise RuntimeError("Sessions are not enabled for this command context.")

    def _session_manager_cwd(self) -> "Path | None":
        if self.session_store_scope == "app":
            return None
        if self.session_store_cwd is not None:
            return self.session_store_cwd
        return self.session_cwd
