"""Shared FastAgent run lifecycle setup and teardown."""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING

from opentelemetry import trace

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from fast_agent.core.fastagent import FastAgent, ManagedRunState, RunRuntime, RunSettings
    from fast_agent.interfaces import AgentProtocol


@dataclass(slots=True)
class FastAgentRunLifecycleState:
    settings: RunSettings
    runtime: RunRuntime
    exit_stack: AsyncExitStack


class FastAgentRunLifecycle:
    """Shared setup/teardown path for ``FastAgent.run`` and Harness."""

    def __init__(self, fast_agent: FastAgent) -> None:
        self._fast_agent = fast_agent

    async def enter(
        self,
        *,
        model_override: str | None = None,
        force_headless: bool = False,
        before_apply_skills: Callable[[], None] | None = None,
    ) -> FastAgentRunLifecycleState:
        exit_stack = AsyncExitStack()
        try:
            await self._fast_agent.app.initialize()
            settings = self._fast_agent._prepare_run_settings(
                model_override=model_override,
                force_headless=force_headless,
            )
            span_context = trace.get_tracer(__name__).start_as_current_span(self._fast_agent.name)
            exit_stack.enter_context(span_context)
            await exit_stack.enter_async_context(self._fast_agent.app.run())
            default_skills = self._fast_agent._load_default_skills_for_run()
            if before_apply_skills is not None:
                before_apply_skills()
            self._fast_agent._apply_skills_to_agent_configs(default_skills)

            if settings.quiet_mode:
                self._fast_agent._configure_quiet_mode_for_run()

            self._fast_agent._validate_run_preconditions()
            runtime = self._fast_agent._create_run_runtime(settings)
            return FastAgentRunLifecycleState(
                settings=settings,
                runtime=runtime,
                exit_stack=exit_stack,
            )
        except BaseException:
            exc_type, exc, traceback = sys.exc_info()
            await exit_stack.__aexit__(exc_type, exc, traceback)
            raise

    async def exit(
        self,
        state: FastAgentRunLifecycleState,
        run_state: ManagedRunState | None,
        active_agents: dict[str, AgentProtocol],
        *,
        had_error: bool,
        shutdown_timeout: float | None = None,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        try:
            await self._fast_agent._finalize_run(
                run_state,
                active_agents,
                had_error=had_error,
                settings=state.settings,
                shutdown_timeout=shutdown_timeout,
            )
        finally:
            await state.exit_stack.__aexit__(exc_type, exc, traceback)


__all__ = [
    "FastAgentRunLifecycle",
    "FastAgentRunLifecycleState",
]
