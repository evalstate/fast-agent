"""Run and server orchestration helpers for FastAgent."""

from __future__ import annotations

import asyncio
import sys
from argparse import Namespace
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any, cast

from fast_agent.core.exceptions import (
    AgentConfigError,
    CircularDependencyError,
    ModelConfigError,
    PromptExitError,
    ProviderKeyError,
    ServerConfigError,
    ServerInitializationError,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.run_lifecycle import FastAgentRunLifecycle
from fast_agent.core.server_runtime import (
    ServerRuntimeContext,
    resolve_server_instance_scope,
    run_server_mode,
)
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.ui.usage_display import display_usage_report
from fast_agent.utils.transports import uses_protocol_stdio

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from fast_agent.core.agent_app import AgentApp
    from fast_agent.core.fastagent import (
        FastAgent,
        ManagedRunState,
        RunRuntime,
        RunSettings,
        RuntimeCallbacks,
    )
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.types import PromptMessageExtended

logger = get_logger(__name__)
_PROMPT_EXIT_SHUTDOWN_TIMEOUT_SECONDS = 1.0


class FastAgentRunMixin:
    name: str
    args: Any
    context: Any
    _agent_card_watch_reload: Callable[[], Awaitable[Any]] | None
    _agent_card_watch_task: asyncio.Task[None] | None
    _server_instance_dispose: Callable[[Any], Awaitable[None]] | None
    _server_managed_instances: list[Any] | None
    _skills_directory_override: Any

    def _get_acp_server_class(self) -> type[Any]:
        raise NotImplementedError

    @staticmethod
    def _resolve_server_instance_scope(
        *,
        transport: str,
        instance_scope: str | None,
    ) -> str:
        return resolve_server_instance_scope(transport=transport, instance_scope=instance_scope)

    async def _initialize_managed_run_state(self, runtime: "RunRuntime") -> "ManagedRunState":
        raise NotImplementedError

    def _build_runtime_callbacks(
        self,
        state: "ManagedRunState",
        settings: "RunSettings",
    ) -> "RuntimeCallbacks":
        raise NotImplementedError

    def _configure_wrapper_callbacks(
        self,
        state: "ManagedRunState",
        callbacks: "RuntimeCallbacks",
        settings: "RunSettings",
    ) -> None:
        raise NotImplementedError

    def _configure_streaming_for_run(self, active_agents: dict[str, "AgentProtocol"]) -> None:
        raise NotImplementedError

    async def _apply_card_tool_cli_option(
        self,
        state: "ManagedRunState",
        refresh_shared_instance: Callable[[], Awaitable[Any]],
    ) -> None:
        raise NotImplementedError

    def _handle_error(self, e: Exception, error_type: str | None = None) -> None:
        raise NotImplementedError

    async def _handle_server_mode(
        self,
        state: "ManagedRunState",
        callbacks: "RuntimeCallbacks",
        settings: "RunSettings",
    ) -> None:
        await run_server_mode(
            ServerRuntimeContext(
                app_name=self.name,
                args=self.args,
                config=self.context.config,
                skills_directory_override=self._skills_directory_override,
                state=state,
                callbacks=callbacks,
                settings=settings,
                acp_server_factory=self._get_acp_server_class,
            )
        )

    def _get_selected_agent(
        self,
        wrapper: "AgentApp",
        active_agents: dict[str, "AgentProtocol"],
        agent_name: str | None,
    ) -> "AgentProtocol":
        if agent_name and agent_name not in active_agents:
            available_agents = ", ".join(active_agents.keys())
            print(
                f"\n\nError: Agent '{agent_name}' not found. Available agents: {available_agents}"
            )
            raise SystemExit(1)
        return wrapper._agent(agent_name)

    async def _handle_message_mode(
        self,
        state: "ManagedRunState",
        settings: "RunSettings",
    ) -> None:
        message = getattr(self.args, "message", None)
        if not message:
            return

        agent_name = getattr(self.args, "agent", None)
        try:
            agent = self._get_selected_agent(state.wrapper, state.active_agents, agent_name)
            response = await agent.send(message)
            if settings.quiet_mode:
                print(f"{response}")
            raise SystemExit(0)
        except SystemExit:
            raise
        except Exception as exc:
            display_agent = agent_name or "<default>"
            print(f"\n\nError sending message to agent '{display_agent}': {exc!s}")
            raise SystemExit(1) from exc

    async def _handle_prompt_file_mode(
        self,
        state: "ManagedRunState",
        settings: "RunSettings",
    ) -> None:
        prompt_file = getattr(self.args, "prompt_file", None)
        if not prompt_file:
            return

        agent_name = getattr(self.args, "agent", None)
        prompt: list[PromptMessageExtended] = load_prompt(prompt_file)
        try:
            agent = self._get_selected_agent(state.wrapper, state.active_agents, agent_name)
            prompt_result = await agent.generate(prompt)
            if settings.quiet_mode:
                print(f"{prompt_result.last_text()}")
            raise SystemExit(0)
        except SystemExit:
            raise
        except Exception as exc:
            display_agent = agent_name or "<default>"
            print(f"\n\nError sending message to agent '{display_agent}': {exc!s}")
            raise SystemExit(1) from exc

    async def _stop_watch_task(self) -> None:
        if self._agent_card_watch_task is None:
            return
        self._agent_card_watch_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._agent_card_watch_task
        self._agent_card_watch_task = None
        self._agent_card_watch_reload = None

    def _print_usage_summary_for_run(
        self,
        state: "ManagedRunState | None",
        active_agents: dict[str, "AgentProtocol"],
        *,
        had_error: bool,
        settings: "RunSettings",
    ) -> None:
        if had_error or settings.quiet_mode:
            return

        managed_instances = state.runtime.managed_instances if state is not None else []
        if managed_instances and not settings.server_mode:
            self._print_usage_report(managed_instances[0].agents)
            return
        if active_agents:
            self._print_usage_report(active_agents)

    async def _dispose_managed_instances(
        self,
        active_agents: dict[str, "AgentProtocol"],
    ) -> None:
        if self._server_managed_instances and self._server_instance_dispose is not None:
            remaining_instances = list(self._server_managed_instances)
            for instance in remaining_instances:
                with suppress(Exception):
                    await self._server_instance_dispose(instance)
            self._server_managed_instances.clear()
            return

        for agent in active_agents.values():
            with suppress(Exception):
                await agent.shutdown()

    async def _finalize_run(
        self,
        state: "ManagedRunState | None",
        active_agents: dict[str, "AgentProtocol"],
        *,
        had_error: bool,
        settings: "RunSettings",
        shutdown_timeout: float | None = None,
    ) -> None:
        try:
            from fast_agent.ui.progress_display import progress_display

            progress_display.stop()
        except Exception:
            pass

        await self._stop_watch_task()
        self._print_usage_summary_for_run(
            state,
            active_agents,
            had_error=had_error,
            settings=settings,
        )
        if shutdown_timeout is None:
            await self._dispose_managed_instances(active_agents)
            return

        try:
            await asyncio.wait_for(
                self._dispose_managed_instances(active_agents),
                timeout=shutdown_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out while shutting down agents after exit request",
                timeout_seconds=shutdown_timeout,
            )

    def _print_usage_report(self, active_agents: dict[str, Any]) -> None:
        """Print a formatted table of token usage for all agents."""
        display_usage_report(active_agents, show_if_progress_disabled=False, subdued_colors=True)

    @asynccontextmanager
    async def run(self) -> AsyncIterator["AgentApp"]:
        """
        Context manager for running the application.
        Initializes all registered agents.
        """
        active_agents: dict[str, AgentProtocol] = {}
        had_error = False
        run_state: ManagedRunState | None = None
        shutdown_timeout: float | None = None
        lifecycle = FastAgentRunLifecycle(cast("FastAgent", self))
        lifecycle_state = None
        try:
            lifecycle_state = await lifecycle.enter()
            settings = lifecycle_state.settings
            run_state = await self._initialize_managed_run_state(lifecycle_state.runtime)
            active_agents = run_state.active_agents

            callbacks = self._build_runtime_callbacks(run_state, settings)
            self._configure_wrapper_callbacks(run_state, callbacks, settings)
            self._configure_streaming_for_run(run_state.active_agents)
            await self._apply_card_tool_cli_option(
                run_state,
                callbacks.refresh_shared_instance,
            )
            await self._handle_server_mode(run_state, callbacks, settings)
            await self._handle_message_mode(run_state, settings)
            await self._handle_prompt_file_mode(run_state, settings)

            yield run_state.wrapper

        except PromptExitError as e:
            shutdown_timeout = _PROMPT_EXIT_SHUTDOWN_TIMEOUT_SECONDS
            self._handle_error(e)
            raise SystemExit(0) from e
        except (
            ServerConfigError,
            ProviderKeyError,
            AgentConfigError,
            ServerInitializationError,
            ModelConfigError,
            CircularDependencyError,
        ) as e:
            had_error = True
            self._handle_error(e)
            raise SystemExit(1) from e

        finally:
            if lifecycle_state is not None:
                exc_type, exc, traceback = sys.exc_info()
                await lifecycle.exit(
                    lifecycle_state,
                    run_state,
                    active_agents,
                    had_error=had_error,
                    shutdown_timeout=shutdown_timeout,
                    exc_type=exc_type,
                    exc=exc,
                    traceback=traceback,
                )

    async def start_server(
        self,
        transport: str = "http",
        host: str = "127.0.0.1",
        port: int = 8000,
        server_name: str | None = None,
        server_description: str | None = None,
        tool_description: str | None = None,
        instance_scope: str | None = None,
        permissions_enabled: bool = True,
        tool_name_template: str | None = None,
    ) -> None:
        """
        Start the application as an MCP, ACP, or A2A server.
        This method initializes agents and exposes them through the selected server transport.
        It is a blocking method that runs until the server is stopped.
        """
        original_args = getattr(self, "args", None)

        self.args = Namespace()
        self.args.server = True
        self.args.transport = transport
        self.args.host = host
        self.args.port = port
        self.args.tool_description = tool_description
        self.args.tool_name_template = tool_name_template
        self.args.server_description = server_description
        self.args.server_name = server_name
        self.args.instance_scope = self._resolve_server_instance_scope(
            transport=transport,
            instance_scope=instance_scope,
        )
        self.args.permissions_enabled = permissions_enabled
        self.args.quiet = bool(getattr(original_args, "quiet", False))
        if uses_protocol_stdio(transport):
            self.args.quiet = True
        self.args.model = None
        missing_arg = object()
        for arg_name in ("model", "agent", "reload", "watch", "card_tools"):
            arg_value = getattr(original_args, arg_name, missing_arg)
            if arg_value is not missing_arg:
                setattr(self.args, arg_name, arg_value)

        try:
            async with self.run():
                pass
        finally:
            if original_args:
                self.args = original_args

    async def main(self) -> bool:
        """Return True when server mode was requested for this app."""
        return bool(getattr(getattr(self, "args", None), "server", False))


__all__ = ["FastAgentRunMixin"]
