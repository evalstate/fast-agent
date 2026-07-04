"""Managed AgentInstance lifecycle for FastAgent."""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fast_agent import config
from fast_agent.core.agent_app import AgentApp, AgentCardLoadResult, AgentRefreshResult
from fast_agent.core.card_tool_attachment import load_and_attach_card_tool_agents
from fast_agent.core.default_agent import agent_is_default, resolve_default_agent_name
from fast_agent.core.direct_factory import (
    active_agents_in_dependency_group,
    create_agents_in_dependency_order,
)
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.instruction_refresh import rebuild_agent_instruction
from fast_agent.core.instruction_utils import apply_instruction_context
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.runtime_finalization import (
    SessionRestoreRequest,
    hydrate_current_session_for_refresh,
    hydration_warnings,
    restore_requested_session,
    session_restore_warnings,
    validate_final_provider_state,
)
from fast_agent.core.validation import get_agent_dependencies, get_dependencies_groups
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.tools.local_shell_executor import LocalEnvironment

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from fastmcp.tools import FunctionTool

    from fast_agent.config import MCPServerSettings
    from fast_agent.context import Context
    from fast_agent.core.agent_card_types import AgentCardData
    from fast_agent.core.fastagent import (
        AgentInstance,
        ManagedRunState,
        RunRuntime,
        RunSettings,
        RuntimeCallbacks,
    )
    from fast_agent.interfaces import AgentProtocol, ModelFactoryFunctionProtocol
    from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult
    from fast_agent.mcp.types import McpAgentProtocol
    from fast_agent.types import PromptMessageExtended

logger = get_logger(__name__)
_DEFAULT_CLI_AGENT_PLACEHOLDER = "agent"


class ManagedRuntimeMixin:
    name: str
    args: Any
    app: Any
    agents: dict[str, "AgentCardData"]
    context: "Context"
    _agent_card_histories: dict[str, list[Path]]
    _agent_card_history_len: dict[str, int]
    _agent_card_history_mtime: dict[str, float]
    _agent_card_last_changed: set[str]
    _agent_card_last_dependents: set[str]
    _agent_card_last_removed: set[str]
    _agent_card_roots: dict[Path, set[str]]
    _agent_card_watch_reload: "Callable[[], Awaitable[AgentRefreshResult]] | None"
    _agent_card_watch_task: asyncio.Task[None] | None
    _agent_registry_version: int
    _card_collision_warnings: list[str]
    _registered_tools: list["FunctionTool"]
    _server_instance_dispose: "Callable[[AgentInstance], Awaitable[None]] | None"
    _server_managed_instances: list["AgentInstance"] | None
    _skills_directory_override: Any

    def _build_model_factory_func(
        self,
        cli_model_override: str | None,
    ) -> "ModelFactoryFunctionProtocol":
        raise NotImplementedError

    def _build_global_prompt_context(
        self,
        *,
        apply_global_prompt_context: bool,
        no_home_mode: bool,
    ) -> dict[str, str] | None:
        raise NotImplementedError

    def _apply_agent_card_histories(self, agents: dict[str, "AgentProtocol"]) -> None:
        raise NotImplementedError

    async def _apply_instruction_context(
        self,
        instance: "AgentInstance",
        context_vars: dict[str, str],
    ) -> None:
        raise NotImplementedError

    def _record_history_snapshot(
        self,
        name: str,
        history_len: int,
        mtime: float | None,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def _get_history_files_mtime(history_files: "Sequence[Path]") -> float | None:
        raise NotImplementedError

    def _sync_agent_card_mcp_servers(self) -> None:
        raise NotImplementedError

    async def _watch_agent_cards(self) -> None:
        raise NotImplementedError

    def _handle_error(self, e: Exception, error_type: str | None = None) -> None:
        raise NotImplementedError

    def load_agents(self, path: str | Path) -> list[str]:
        raise NotImplementedError

    def attach_agent_tools(self, parent_name: str, child_names: "Sequence[str]") -> list[str]:
        raise NotImplementedError

    def detach_agent_tools(self, parent_name: str, child_names: "Sequence[str]") -> list[str]:
        raise NotImplementedError

    async def reload_agents(self) -> bool:
        raise NotImplementedError

    def dump_agent_card_text(self, name: str, *, as_yaml: bool = False) -> str:
        raise NotImplementedError

    def _create_run_runtime(self, settings: "RunSettings") -> "RunRuntime":
        """Create the immutable/shared runtime resources for a run."""
        from fast_agent.core.fastagent import RunRuntime

        active_settings = config.get_settings()
        shell_settings = active_settings.shell_execution
        return RunRuntime(
            model_factory_func=self._build_model_factory_func(settings.cli_model_override),
            global_prompt_context=self._build_global_prompt_context(
                apply_global_prompt_context=not settings.is_acp_server_mode,
                no_home_mode=settings.no_home_mode,
            ),
            resume_requested=settings.resume_requested,
            resume_session_id=settings.resume_session_id,
            target_agent_name=settings.target_agent_name,
            is_acp_server_mode=settings.is_acp_server_mode,
            no_home_mode=settings.no_home_mode,
            managed_instances=[],
            instance_lock=asyncio.Lock(),
            shell_environment=LocalEnvironment(
                logger=logger,
                timeout_seconds=shell_settings.timeout_seconds,
                warning_interval_seconds=shell_settings.warning_interval_seconds,
                config=active_settings,
            ),
        )

    async def _instantiate_agent_instance(
        self,
        runtime: "RunRuntime",
        app_override: AgentApp | None = None,
    ) -> "AgentInstance":
        from fast_agent.core.fastagent import AgentInstance

        async with runtime.instance_lock:
            agents_map = await create_agents_in_dependency_order(
                self.app,
                self.agents,
                runtime.model_factory_func,
                global_function_tools=self._registered_tools,
                shell_environment=runtime.shell_environment,
            )

            tool_only_agents = {
                name for name, data in self.agents.items() if data.get("tool_only", False)
            }
            settings = config.get_settings()
            plugin_command_base_path = (
                Path(settings._config_file).parent if settings._config_file is not None else None
            )
            if app_override is None:
                app = AgentApp(
                    agents_map,
                    tool_only_agents=tool_only_agents,
                    card_collision_warnings=self._card_collision_warnings,
                    no_home_mode=runtime.no_home_mode,
                    plugin_commands=settings.commands,
                    plugin_command_base_path=plugin_command_base_path,
                )
            else:
                app_override.set_agents(
                    agents_map,
                    tool_only_agents=tool_only_agents,
                    card_collision_warnings=self._card_collision_warnings,
                )
                app_override.set_plugin_commands(
                    settings.commands,
                    base_path=plugin_command_base_path,
                )
                app_override.no_home_mode = runtime.no_home_mode
                app = app_override

            instance = AgentInstance(
                app,
                agents_map,
                registry_version=self._agent_registry_version,
            )
            runtime.managed_instances.append(instance)
            self._apply_agent_card_histories(instance.agents)
            return instance

    async def _dispose_agent_instance(
        self,
        runtime: "RunRuntime",
        instance: "AgentInstance",
    ) -> None:
        async with runtime.instance_lock:
            if instance in runtime.managed_instances:
                runtime.managed_instances.remove(instance)
        await instance.shutdown()

    def _session_restore_request(
        self,
        app: AgentApp,
        runtime: "RunRuntime",
    ) -> SessionRestoreRequest:
        target_agent_name = runtime.target_agent_name
        if (
            target_agent_name == _DEFAULT_CLI_AGENT_PLACEHOLDER
            and app.get_agent(target_agent_name) is None
        ):
            target_agent_name = None
        return SessionRestoreRequest(
            session_id=runtime.resume_session_id,
            fallback_agent_name=app.resolve_target_agent_name(target_agent_name),
        )

    async def _finalize_initial_agent_instance(
        self,
        runtime: "RunRuntime",
        instance: "AgentInstance",
    ) -> AgentRefreshResult:
        if runtime.global_prompt_context:
            await self._apply_instruction_context(instance, runtime.global_prompt_context)

        restore_result = None
        if runtime.resume_requested:
            restore_result = await restore_requested_session(
                instance.agents,
                self._session_restore_request(instance.app, runtime),
            )
            instance.app.set_session_restore_result(restore_result)

        if not runtime.is_acp_server_mode:
            validate_final_provider_state(instance.agents)

        return AgentRefreshResult(
            changed=restore_result is not None,
            active_agent=restore_result.active_agent if restore_result is not None else None,
            warnings=session_restore_warnings(restore_result),
        )

    async def _initialize_managed_run_state(self, runtime: "RunRuntime") -> "ManagedRunState":
        """Create the primary shared app instance for this run."""
        from fast_agent.core.fastagent import ManagedRunState

        primary_instance = await self._instantiate_agent_instance(runtime)
        refresh_result = await self._finalize_initial_agent_instance(runtime, primary_instance)
        primary_instance.app.set_refresh_result(refresh_result)
        return ManagedRunState(
            runtime=runtime,
            primary_instance=primary_instance,
            wrapper=primary_instance.app,
            active_agents=primary_instance.agents,
        )

    def _expand_impacted_agents(
        self,
        impacted: set[str],
        removed_names: set[str],
    ) -> set[str]:
        if not impacted:
            return impacted

        expanded = set(impacted)
        reverse_deps: dict[str, set[str]] = {}
        for name, agent_data in self.agents.items():
            for dep in get_agent_dependencies(agent_data):
                reverse_deps.setdefault(dep, set()).add(name)

        queue = list(expanded)
        while queue:
            current = queue.pop()
            for parent in reverse_deps.get(current, set()):
                if parent in removed_names or parent in expanded:
                    continue
                expanded.add(parent)
                queue.append(parent)
        return expanded

    async def _rebuild_impacted_agents(
        self,
        active_agents: dict[str, "AgentProtocol"],
        impacted: set[str],
        model_factory_func: "ModelFactoryFunctionProtocol",
    ) -> None:
        if not impacted:
            return

        dependencies = get_dependencies_groups(self.agents)
        for group in dependencies:
            group_targets = [name for name in group if name in impacted]
            if not group_targets:
                continue
            await active_agents_in_dependency_group(
                self.app,
                self.agents,
                model_factory_func,
                self._registered_tools,
                group_targets,
                active_agents,
            )

    def _reload_updated_agent_file_histories(
        self,
        updated_agents: dict[str, "AgentProtocol"],
        *,
        allow_unchanged_empty: bool = False,
    ) -> None:
        for name, new_agent in updated_agents.items():
            history_files = self._agent_card_histories.get(name)
            if not history_files:
                continue

            files_mtime = self._get_history_files_mtime(history_files)
            if files_mtime is None:
                continue

            last_mtime = self._agent_card_history_mtime.get(name)
            last_len = self._agent_card_history_len.get(name)
            current_len = len(new_agent.message_history)
            if last_mtime is None:
                if current_len != 0:
                    continue
            elif not (
                allow_unchanged_empty and current_len == 0 and files_mtime == last_mtime
            ) and (files_mtime <= last_mtime or (last_len is not None and current_len != last_len)):
                continue

            messages: list[PromptMessageExtended] = []
            for history_file in history_files:
                messages.extend(load_prompt(history_file))
            if not messages:
                continue

            new_agent.message_history.clear()
            new_agent.message_history.extend(messages)
            self._record_history_snapshot(name, len(new_agent.message_history), files_mtime)

    async def _finalize_updated_agents(
        self,
        updated_agents: dict[str, "AgentProtocol"],
        runtime: "RunRuntime",
    ) -> None:
        if not updated_agents:
            return

        if runtime.global_prompt_context:
            await apply_instruction_context(updated_agents.values(), runtime.global_prompt_context)

        if not runtime.is_acp_server_mode:
            validate_final_provider_state(updated_agents)

    @staticmethod
    def _log_local_hydration_messages(warnings: list[str]) -> None:
        for warning in warnings:
            logger.warning(
                "Shared runtime reload hydration warning",
                name="shared_reload_hydration_warning",
                warning=warning,
            )

    async def _refresh_result_from_session_restore(
        self,
        agents: dict[str, "AgentProtocol"],
        updated_agents: dict[str, "AgentProtocol"] | None = None,
    ) -> AgentRefreshResult:
        agents_to_hydrate = updated_agents if updated_agents is not None else agents
        hydration = await hydrate_current_session_for_refresh(
            agents_to_hydrate,
            fallback_agent_name=resolve_default_agent_name(
                agents_to_hydrate,
                is_default=lambda _name, agent: agent_is_default(agent),
            ),
        )
        self._log_local_hydration_messages(hydration_warnings(hydration))
        if hydration is None:
            if updated_agents:
                self._reload_updated_agent_file_histories(
                    updated_agents,
                    allow_unchanged_empty=True,
                )
            return AgentRefreshResult(changed=True)

        snapshot_agent_names = set(hydration.snapshot.continuation.agents)
        if updated_agents:
            unpersisted_agents = {
                name: agent
                for name, agent in updated_agents.items()
                if name not in snapshot_agent_names
            }
            if unpersisted_agents:
                self._reload_updated_agent_file_histories(
                    unpersisted_agents,
                    allow_unchanged_empty=True,
                )

        active_agent = hydration.active_agent
        if updated_agents is not None and active_agent not in updated_agents:
            active_agent = None
        return AgentRefreshResult(
            changed=True,
            active_agent=active_agent,
            warnings=hydration_warnings(hydration),
        )

    async def _refresh_shared_instance(self, state: "ManagedRunState") -> AgentRefreshResult:
        if self._agent_registry_version <= state.primary_instance.registry_version:
            return AgentRefreshResult(changed=False)

        self._sync_agent_card_mcp_servers()
        changed_names = set(self._agent_card_last_changed)
        removed_names = set(self._agent_card_last_removed)
        dependent_names = set(self._agent_card_last_dependents)
        active_agents_local = state.active_agents

        if not (changed_names or removed_names or dependent_names):
            new_instance = await self._instantiate_agent_instance(
                state.runtime,
                app_override=state.wrapper,
            )
            await self._finalize_updated_agents(new_instance.agents, state.runtime)
            refresh_result = await self._refresh_result_from_session_restore(new_instance.agents)
            old_instance = state.primary_instance
            state.primary_instance = new_instance
            state.active_agents = new_instance.agents
            await self._dispose_agent_instance(state.runtime, old_instance)
            return refresh_result

        async with state.runtime.instance_lock:
            impacted = set(changed_names)
            impacted.update(dependent_names)
            impacted.difference_update(removed_names)
            impacted = self._expand_impacted_agents(impacted, removed_names)

            removed_instances = [active_agents_local.pop(name, None) for name in removed_names]
            for agent in removed_instances:
                if agent is None:
                    continue
                await agent.shutdown()

            old_agents = {
                name: active_agents_local.get(name)
                for name in impacted
                if name in active_agents_local
            }

            await self._rebuild_impacted_agents(
                active_agents_local,
                impacted,
                state.runtime.model_factory_func,
            )

            for name, old_agent in old_agents.items():
                new_agent = active_agents_local.get(name)
                if old_agent is None or new_agent is None:
                    continue
                if old_agent is new_agent:
                    continue
                await old_agent.shutdown()

            if impacted:
                updated_agents = {
                    name: active_agents_local[name]
                    for name in impacted
                    if name in active_agents_local
                }
                await self._finalize_updated_agents(updated_agents, state.runtime)
                refresh_result = await self._refresh_result_from_session_restore(
                    active_agents_local,
                    updated_agents,
                )
            else:
                refresh_result = AgentRefreshResult(changed=True)

            state.primary_instance.registry_version = self._agent_registry_version
            state.active_agents = active_agents_local
            self._agent_card_last_changed.clear()
            self._agent_card_last_removed.clear()
            self._agent_card_last_dependents.clear()
            return refresh_result

    async def _reload_and_refresh(self, state: "ManagedRunState") -> AgentRefreshResult:
        changed = await self.reload_agents()
        if not changed:
            return AgentRefreshResult(changed=False)
        return await self._refresh_shared_instance(state)

    async def _load_card_core(
        self,
        state: "ManagedRunState",
        source: str,
        parent_name: str | None,
        *,
        should_refresh: bool,
    ) -> AgentCardLoadResult:
        loaded_names = self.load_agents(source)

        attached_names: list[str] = []
        if parent_name:
            target_name = parent_name
            if target_name not in self.agents:
                target_name = next(iter(self.agents.keys()), None)
            if target_name and loaded_names:
                attached_names = self.attach_agent_tools(target_name, loaded_names)

        if should_refresh:
            await self._refresh_shared_instance(state)
        return AgentCardLoadResult(loaded_names=loaded_names, attached_names=attached_names)

    async def _attach_agent_tools_and_refresh(
        self,
        state: "ManagedRunState",
        parent_name: str,
        child_names: "Sequence[str]",
    ) -> list[str]:
        added = self.attach_agent_tools(parent_name, child_names)
        if added:
            await self._refresh_shared_instance(state)
        return added

    async def _detach_agent_tools_and_refresh(
        self,
        state: "ManagedRunState",
        parent_name: str,
        child_names: "Sequence[str]",
    ) -> list[str]:
        removed = self.detach_agent_tools(parent_name, child_names)
        if removed:
            await self._refresh_shared_instance(state)
        return removed

    @staticmethod
    def _resolve_runtime_mcp_agent(
        active_agents: dict[str, "AgentProtocol"],
        agent_name: str,
    ) -> "McpAgentProtocol":
        from fast_agent.mcp.types import McpAgentProtocol

        target_agent = active_agents.get(agent_name)
        if target_agent is None:
            raise RuntimeError(f"Agent '{agent_name}' was not found")
        if not isinstance(target_agent, McpAgentProtocol):
            raise RuntimeError(f"Agent '{agent_name}' does not support MCP server management")
        return target_agent

    async def _attach_mcp_server_and_refresh(
        self,
        active_agents: dict[str, "AgentProtocol"],
        agent_name: str,
        server_name: str,
        server_config: "MCPServerSettings | None" = None,
        options: "MCPAttachOptions | None" = None,
    ) -> "MCPAttachResult":
        target_agent = self._resolve_runtime_mcp_agent(active_agents, agent_name)
        result = await target_agent.attach_mcp_server(
            server_name=server_name,
            server_config=server_config,
            options=options,
        )
        await rebuild_agent_instruction(target_agent)
        return result

    async def _detach_mcp_server_and_refresh(
        self,
        active_agents: dict[str, "AgentProtocol"],
        agent_name: str,
        server_name: str,
    ) -> "MCPDetachResult":
        target_agent = self._resolve_runtime_mcp_agent(active_agents, agent_name)
        result = await target_agent.detach_mcp_server(server_name)
        await rebuild_agent_instruction(target_agent)
        return result

    async def _list_attached_mcp_servers(
        self,
        active_agents: dict[str, "AgentProtocol"],
        agent_name: str,
    ) -> list[str]:
        target_agent = self._resolve_runtime_mcp_agent(active_agents, agent_name)
        return target_agent.list_attached_mcp_servers()

    async def _list_configured_detached_mcp_servers(
        self,
        active_agents: dict[str, "AgentProtocol"],
        agent_name: str,
    ) -> list[str]:
        target_agent = self._resolve_runtime_mcp_agent(active_agents, agent_name)
        return target_agent.aggregator.list_configured_detached_servers()

    async def _dump_agent_card_callback(self, name: str) -> str:
        return self.dump_agent_card_text(name)

    async def _create_runtime_agent_instance(self, state: "ManagedRunState") -> "AgentInstance":
        instance = await self._instantiate_agent_instance(state.runtime)
        self._configure_runtime_mcp_callbacks(instance.app)
        return instance

    def _build_runtime_callbacks(
        self,
        state: "ManagedRunState",
        settings: "RunSettings",
    ) -> "RuntimeCallbacks":
        from fast_agent.core.fastagent import RuntimeCallbacks

        return RuntimeCallbacks(
            create_instance=partial(self._create_runtime_agent_instance, state),
            dispose_instance=partial(self._dispose_agent_instance, state.runtime),
            refresh_shared_instance=partial(self._refresh_shared_instance, state),
            reload_and_refresh=partial(self._reload_and_refresh, state),
            reload_source=self.reload_agents if settings.reload_enabled else None,
            load_card_and_refresh=partial(self._load_card_core, state, should_refresh=True),
            load_card_source=partial(self._load_card_core, state, should_refresh=False),
            attach_agent_tools_and_refresh=partial(
                self._attach_agent_tools_and_refresh,
                state,
            ),
            detach_agent_tools_and_refresh=partial(
                self._detach_agent_tools_and_refresh,
                state,
            ),
            attach_agent_tools_source=self._attach_agent_tools_source,
            detach_agent_tools_source=self._detach_agent_tools_source,
            attach_mcp_server=partial(
                self._attach_mcp_server_and_refresh,
                state.active_agents,
            ),
            detach_mcp_server=partial(
                self._detach_mcp_server_and_refresh,
                state.active_agents,
            ),
            list_attached_mcp_servers=partial(
                self._list_attached_mcp_servers,
                state.active_agents,
            ),
            list_configured_detached_mcp_servers=partial(
                self._list_configured_detached_mcp_servers,
                state.active_agents,
            ),
            dump_agent_card=self._dump_agent_card_callback,
        )

    async def _attach_agent_tools_source(
        self,
        parent_name: str,
        child_names: "Sequence[str]",
    ) -> list[str]:
        return self.attach_agent_tools(parent_name, child_names)

    async def _detach_agent_tools_source(
        self,
        parent_name: str,
        child_names: "Sequence[str]",
    ) -> list[str]:
        return self.detach_agent_tools(parent_name, child_names)

    def _configure_runtime_mcp_callbacks(self, app: AgentApp) -> None:
        def active_agents() -> dict[str, AgentProtocol]:
            return cast("dict[str, AgentProtocol]", app.registered_agents())

        async def attach_mcp_server(
            agent_name: str,
            server_name: str,
            server_config: "MCPServerSettings | None" = None,
            options: "MCPAttachOptions | None" = None,
        ) -> MCPAttachResult:
            return await self._attach_mcp_server_and_refresh(
                active_agents(),
                agent_name,
                server_name,
                server_config,
                options,
            )

        async def detach_mcp_server(
            agent_name: str,
            server_name: str,
        ) -> MCPDetachResult:
            return await self._detach_mcp_server_and_refresh(
                active_agents(),
                agent_name,
                server_name,
            )

        async def list_attached_mcp_servers(agent_name: str) -> list[str]:
            return await self._list_attached_mcp_servers(active_agents(), agent_name)

        async def list_configured_detached_mcp_servers(agent_name: str) -> list[str]:
            return await self._list_configured_detached_mcp_servers(
                active_agents(),
                agent_name,
            )

        app.set_attach_mcp_server_callback(attach_mcp_server)
        app.set_detach_mcp_server_callback(detach_mcp_server)
        app.set_list_attached_mcp_servers_callback(list_attached_mcp_servers)
        app.set_list_configured_detached_mcp_servers_callback(list_configured_detached_mcp_servers)

    def _configure_wrapper_callbacks(
        self,
        state: "ManagedRunState",
        callbacks: "RuntimeCallbacks",
        settings: "RunSettings",
    ) -> None:
        wrapper = state.wrapper
        wrapper.set_reload_callback(
            callbacks.reload_and_refresh if settings.reload_enabled else None
        )
        wrapper.set_refresh_callback(
            callbacks.refresh_shared_instance if settings.reload_enabled else None
        )
        wrapper.set_load_card_callback(callbacks.load_card_and_refresh)
        wrapper.set_attach_agent_tools_callback(callbacks.attach_agent_tools_and_refresh)
        wrapper.set_detach_agent_tools_callback(callbacks.detach_agent_tools_and_refresh)
        wrapper.set_dump_agent_callback(callbacks.dump_agent_card)
        self._configure_runtime_mcp_callbacks(wrapper)

        self._agent_card_watch_reload = (
            callbacks.reload_and_refresh if settings.reload_enabled else None
        )
        if getattr(self.args, "watch", False) and self._agent_card_roots:
            self._agent_card_watch_task = asyncio.create_task(self._watch_agent_cards())

        self._server_instance_dispose = callbacks.dispose_instance
        self._server_managed_instances = state.runtime.managed_instances

    async def _apply_card_tool_cli_option(
        self,
        state: "ManagedRunState",
        refresh_callback: "Callable[[], Awaitable[AgentRefreshResult]]",
    ) -> None:
        card_tools = getattr(self.args, "card_tools", None)
        if not card_tools:
            return

        try:
            load_and_attach_card_tool_agents(
                self,
                card_tools,
                preferred_agent_names=[getattr(self.args, "agent", None)],
            )
        except AgentConfigError as exc:
            self._handle_error(exc)
            raise SystemExit(1) from exc

        await refresh_callback()


__all__ = ["ManagedRuntimeMixin"]
