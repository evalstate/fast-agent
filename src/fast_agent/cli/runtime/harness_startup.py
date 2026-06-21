"""Harness-backed startup helpers for local interactive CLI sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from fast_agent.core.exceptions import (
    AgentConfigError,
    CircularDependencyError,
    ModelConfigError,
    PromptExitError,
    ProviderKeyError,
    ServerConfigError,
    ServerInitializationError,
)

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from fast_agent.cli.runtime.run_request import AgentRunRequest
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.core.harness import AgentHarness, HarnessSession
    from fast_agent.session.session_manager import SessionManager


class HarnessProvider(Protocol):
    def harness(self) -> AbstractAsyncContextManager[AgentHarness]: ...

    def _handle_error(self, e: Exception, error_type: str | None = None) -> None: ...


class CliRuntimeProvider(HarnessProvider, Protocol):
    def run(self) -> AbstractAsyncContextManager[AgentApp]: ...


class CliFlow(Protocol):
    async def __call__(
        self,
        agent_app: AgentApp,
        request: AgentRunRequest,
        *,
        session_manager: SessionManager | None = None,
        harness_session: HarnessSession | None = None,
    ) -> None: ...


class ParallelCliFlow(Protocol):
    async def __call__(
        self,
        agent_app: AgentApp,
        request: AgentRunRequest,
        fan_out_agent_names: list[str],
        *,
        session_manager: SessionManager | None = None,
        harness_session: HarnessSession | None = None,
    ) -> None: ...


def should_use_harness_startup(request: AgentRunRequest) -> bool:
    return (
        request.mode == "interactive"
        and request.is_repl
        and request.allow_sessions
        and request.resume is None
    )


def new_harness_session_id(request: AgentRunRequest) -> str:
    from fast_agent.session.session_manager import SessionManager

    manager = SessionManager(environment_override=request.environment_dir)
    return manager.generate_session_id()


async def run_harness_cli_flow(
    fast: HarnessProvider,
    request: AgentRunRequest,
    *,
    flow: CliFlow,
) -> None:
    session_id = new_harness_session_id(request)
    try:
        async with fast.harness() as harness:
            session = await harness.session(session_id, agent_name=request.target_agent_name)
            await flow(
                session.agent_app,
                request,
                session_manager=session.session_manager,
                harness_session=session,
            )
    except PromptExitError as exc:
        fast._handle_error(exc)
        raise SystemExit(0) from exc
    except (
        ServerConfigError,
        ProviderKeyError,
        AgentConfigError,
        ServerInitializationError,
        ModelConfigError,
        CircularDependencyError,
    ) as exc:
        fast._handle_error(exc)
        raise SystemExit(1) from exc


async def run_cli_flow(
    fast: CliRuntimeProvider,
    request: AgentRunRequest,
    *,
    flow: CliFlow,
) -> None:
    if should_use_harness_startup(request):
        await run_harness_cli_flow(fast, request, flow=flow)
        return

    async with fast.run() as agent_app:
        await flow(agent_app, request)


async def run_harness_parallel_cli_flow(
    fast: HarnessProvider,
    request: AgentRunRequest,
    fan_out_agent_names: list[str],
    *,
    flow: ParallelCliFlow,
) -> None:
    session_id = new_harness_session_id(request)
    try:
        async with fast.harness() as harness:
            session = await harness.session(session_id, agent_name=request.target_agent_name)
            await flow(
                session.agent_app,
                request,
                fan_out_agent_names,
                session_manager=session.session_manager,
                harness_session=session,
            )
    except PromptExitError as exc:
        fast._handle_error(exc)
        raise SystemExit(0) from exc
    except (
        ServerConfigError,
        ProviderKeyError,
        AgentConfigError,
        ServerInitializationError,
        ModelConfigError,
        CircularDependencyError,
    ) as exc:
        fast._handle_error(exc)
        raise SystemExit(1) from exc


async def run_parallel_cli_flow(
    fast: CliRuntimeProvider,
    request: AgentRunRequest,
    fan_out_agent_names: list[str],
    *,
    flow: ParallelCliFlow,
) -> None:
    if should_use_harness_startup(request):
        await run_harness_parallel_cli_flow(
            fast,
            request,
            fan_out_agent_names,
            flow=flow,
        )
        return

    async with fast.run() as agent_app:
        await flow(agent_app, request, fan_out_agent_names)
