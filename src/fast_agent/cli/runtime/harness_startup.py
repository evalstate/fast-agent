"""Harness-backed startup helpers for local CLI sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from fast_agent.core.exceptions import (
    AgentConfigError,
    CircularDependencyError,
    EnvironmentStartupError,
    ModelConfigError,
    PromptExitError,
    ProviderKeyError,
    ServerConfigError,
    ServerInitializationError,
)
from fast_agent.core.harness_app import AppOpenRequest
from fast_agent.tools.environment_registry import UnknownEnvironmentError

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from fast_agent.cli.runtime.run_request import AgentRunRequest
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.core.harness import AgentHarness, HarnessSession
    from fast_agent.session.session_manager import SessionManager
    from fast_agent.tools.environment_registry import EnvironmentSelection


class HarnessProvider(Protocol):
    def harness(
        self,
        *,
        environment: EnvironmentSelection = None,
    ) -> AbstractAsyncContextManager[AgentHarness]: ...

    def _handle_error(self, e: Exception, error_type: str | None = None) -> None: ...


class CliRuntimeProvider(HarnessProvider, Protocol):
    def run(
        self,
        *,
        environment: EnvironmentSelection = None,
    ) -> AbstractAsyncContextManager[AgentApp]: ...


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
    return request.mode == "interactive" and request.allow_sessions


def initial_harness_session_id(request: AgentRunRequest) -> str:
    from fast_agent.session.session_manager import SessionManager

    manager = SessionManager(home_override=request.home)
    if request.resume is not None:
        if request.resume:
            resolved = manager.resolve_session_name(request.resume)
            if resolved is not None:
                return resolved
        latest = manager.load_latest_session(require_content=True)
        if latest is not None:
            return latest.info.name
    return manager.generate_session_id()


def _disable_core_resume_for_harness_startup(fast: HarnessProvider) -> tuple[bool, object | None]:
    """Let harness persistence own initial session hydration for interactive runs."""
    args = getattr(fast, "args", None)
    if args is None:
        return False, None
    original = getattr(args, "resume_requested", None)
    if original is not True:
        return False, original
    args.resume_requested = False
    return True, original


def _restore_core_resume_flag(fast: HarnessProvider, disabled: bool, original: object | None) -> None:
    if not disabled:
        return
    args = getattr(fast, "args", None)
    if args is not None:
        args.resume_requested = original


async def run_harness_cli_flow(
    fast: HarnessProvider,
    request: AgentRunRequest,
    *,
    flow: CliFlow,
    prepare: Callable[[], None] | None = None,
) -> None:
    session_id = initial_harness_session_id(request)
    disabled_resume, original_resume = _disable_core_resume_for_harness_startup(fast)
    try:
        async with fast.harness(environment=request.environment) as harness:
            if prepare is not None:
                prepare()
            app = harness.app()
            async with app.open(
                AppOpenRequest(session_id=session_id, agent=request.target_agent_name)
            ) as session:
                await flow(
                    session.agent_app,
                    request,
                    session_manager=session.env.session_manager,
                    harness_session=session.env.harness_session,
                )
    except PromptExitError as exc:
        fast._handle_error(exc)
        raise SystemExit(0) from exc
    except (
        ServerConfigError,
        ProviderKeyError,
        AgentConfigError,
        ServerInitializationError,
        EnvironmentStartupError,
        ModelConfigError,
        CircularDependencyError,
        UnknownEnvironmentError,
    ) as exc:
        fast._handle_error(exc)
        raise SystemExit(1) from exc
    finally:
        _restore_core_resume_flag(fast, disabled_resume, original_resume)


async def run_cli_flow(
    fast: CliRuntimeProvider,
    request: AgentRunRequest,
    *,
    flow: CliFlow,
    prepare: Callable[[], None] | None = None,
) -> None:
    if should_use_harness_startup(request):
        await run_harness_cli_flow(fast, request, flow=flow, prepare=prepare)
        return

    if prepare is not None:
        prepare()
    async with fast.run(environment=request.environment) as agent_app:
        await flow(agent_app, request)


async def run_harness_parallel_cli_flow(
    fast: HarnessProvider,
    request: AgentRunRequest,
    fan_out_agent_names: list[str],
    *,
    flow: ParallelCliFlow,
) -> None:
    session_id = initial_harness_session_id(request)
    disabled_resume, original_resume = _disable_core_resume_for_harness_startup(fast)
    try:
        async with fast.harness(environment=request.environment) as harness:
            app = harness.app()
            async with app.open(
                AppOpenRequest(session_id=session_id, agent=request.target_agent_name)
            ) as session:
                await flow(
                    session.agent_app,
                    request,
                    fan_out_agent_names,
                    session_manager=session.env.session_manager,
                    harness_session=session.env.harness_session,
                )
    except PromptExitError as exc:
        fast._handle_error(exc)
        raise SystemExit(0) from exc
    except (
        ServerConfigError,
        ProviderKeyError,
        AgentConfigError,
        ServerInitializationError,
        EnvironmentStartupError,
        ModelConfigError,
        CircularDependencyError,
        UnknownEnvironmentError,
    ) as exc:
        fast._handle_error(exc)
        raise SystemExit(1) from exc
    finally:
        _restore_core_resume_flag(fast, disabled_resume, original_resume)


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

    async with fast.run(environment=request.environment) as agent_app:
        await flow(agent_app, request, fan_out_agent_names)
