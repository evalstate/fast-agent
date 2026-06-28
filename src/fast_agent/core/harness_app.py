"""Harness application boundary for local UI and protocol adapters."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from fast_agent.skills import SkillManifest, SkillRegistry
from fast_agent.types import AgentRequest, AgentResponse, PromptMessageExtended, RequestParams
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from contextlib import AbstractAsyncContextManager

    from mcp.types import PromptMessage

    from fast_agent.config import Settings
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.core.harness import HarnessSession, HarnessSessions
    from fast_agent.session.session_manager import SessionManager
    from fast_agent.tools.session_environment import ShellExecutionResult

RuntimeSkillSource: TypeAlias = SkillManifest | SkillRegistry | Path | str
RuntimeSkillConfig: TypeAlias = RuntimeSkillSource | Sequence[RuntimeSkillSource | None] | None


@dataclass(frozen=True, slots=True)
class AppOpenRequest:
    """Request to open a harness-backed application session."""

    session_id: str | None = None
    agent: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class HarnessApp(Protocol):
    """Application boundary shared by local UI and protocol adapters."""

    def open(
        self,
        request: AppOpenRequest | None = None,
    ) -> "AbstractAsyncContextManager[HarnessAppSession]": ...


class HarnessAppSession(Protocol):
    """Live application session opened from a harness app."""

    @property
    def agent_app(self) -> "AgentApp": ...

    @property
    def env(self) -> "AgentRuntimeEnvironment": ...

    async def invoke(self, request: AgentRequest) -> AgentResponse: ...


class HarnessSessionProvider(Protocol):
    """Minimal provider used by the default harness app."""

    async def session(
        self,
        session_id: str | None = None,
        *,
        agent_name: str | None = None,
    ) -> "HarnessSession": ...


@dataclass(frozen=True, slots=True)
class HarnessAppContext:
    """Context passed to custom harness app factories."""

    default_app: HarnessApp
    session_provider: HarnessSessionProvider
    settings: "Settings | None" = None


class HarnessAppFactory(Protocol):
    """Factory loaded from ``harness_app.entrypoint``."""

    def __call__(self, context: HarnessAppContext) -> HarnessApp: ...


@runtime_checkable
class RuntimeSkillTarget(Protocol):
    """Agent interface required for runtime skill updates."""

    @property
    def skill_manifests(self) -> list[SkillManifest]: ...

    def set_skill_manifests(self, manifests: Sequence[SkillManifest]) -> None: ...


class RuntimeAgent:
    """Agent helper scoped to a runtime environment."""

    def __init__(self, env: AgentRuntimeEnvironment, name: str | None) -> None:
        self._env = env
        self._name = name

    async def send(
        self,
        message: "str | PromptMessage | PromptMessageExtended | Sequence[str | PromptMessage | PromptMessageExtended]",
        *,
        request_params: RequestParams | None = None,
    ) -> str:
        return await self._env.harness_session.send(
            message,
            agent_name=self._name,
            request_params=request_params,
        )

    async def generate(
        self,
        message: "str | PromptMessage | PromptMessageExtended | Sequence[str | PromptMessage | PromptMessageExtended]",
        *,
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended:
        return await self._env.harness_session.generate(
            message,
            agent_name=self._name,
            request_params=request_params,
        )

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        if request.agent is not None:
            return await self._env.harness_session.invoke(request)
        return await self._env.harness_session.invoke(
            AgentRequest(
                message=request.message,
                agent=self._name,
                session_id=request.session_id,
                auth=request.auth,
                params=request.params,
                metadata=dict(request.metadata),
                state=dict(request.state),
                progress=request.progress,
            )
        )


class RuntimeTools:
    """Programmatic tools available to a harness app session."""

    def __init__(self, harness_session: "HarnessSession") -> None:
        self._harness_session = harness_session

    async def bash(
        self,
        command: str,
        *,
        cwd: "str | Path | None" = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> "ShellExecutionResult":
        return await self._harness_session.shell(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )

    async def execute(
        self,
        command: str,
        *,
        args: Sequence[str] = (),
        cwd: "str | Path | None" = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> "ShellExecutionResult":
        shell_command = shlex.join([command, *args])
        return await self.bash(shell_command, cwd=cwd, env=env, timeout=timeout)


class RuntimeSkills:
    """Skill helpers scoped to one harness app environment."""

    def __init__(self, env: AgentRuntimeEnvironment) -> None:
        self._env = env

    def add(
        self,
        skills: RuntimeSkillConfig,
        *,
        agent: str | None = None,
    ) -> list[SkillManifest]:
        """Add skills to the target agent and return the resulting manifests."""
        target = self._skill_target(agent)
        manifests = _deduplicate_skills([*target.skill_manifests, *self._resolve(skills)])
        target.set_skill_manifests(manifests)
        return manifests

    def replace(
        self,
        skills: RuntimeSkillConfig,
        *,
        agent: str | None = None,
    ) -> list[SkillManifest]:
        """Replace the target agent's skills and return the resulting manifests."""
        target = self._skill_target(agent)
        manifests = _deduplicate_skills(self._resolve(skills))
        target.set_skill_manifests(manifests)
        return manifests

    def _skill_target(self, agent: str | None) -> RuntimeSkillTarget:
        target = self._env.agent_app.resolve_agent(agent or self._env.harness_session.default_agent_name)
        if not isinstance(target, RuntimeSkillTarget):
            target_name = agent or self._env.harness_session.default_agent_name or "<default>"
            raise TypeError(f"Agent {target_name!r} does not support runtime skills.")
        return target

    def _resolve(self, skills: RuntimeSkillConfig) -> list[SkillManifest]:
        if skills is None:
            return []
        if isinstance(skills, SkillManifest):
            return [skills]
        if isinstance(skills, SkillRegistry):
            return skills.load_manifests()
        if isinstance(skills, (Path, str)):
            return SkillRegistry(base_dir=Path.cwd(), directories=[skills]).load_manifests()
        if isinstance(skills, Sequence):
            manifests: list[SkillManifest] = []
            directory_entries: list[Path | str] = []
            for item in skills:
                if item is None:
                    continue
                if isinstance(item, SkillManifest):
                    manifests.append(item)
                elif isinstance(item, SkillRegistry):
                    manifests.extend(item.load_manifests())
                elif isinstance(item, (Path, str)):
                    directory_entries.append(item)
            if directory_entries:
                manifests.extend(
                    SkillRegistry(base_dir=Path.cwd(), directories=directory_entries).load_manifests()
                )
            return manifests
        return []


@dataclass(slots=True)
class AgentRuntimeEnvironment:
    """Runtime environment exposed to harness app implementations."""

    harness_session: "HarnessSession"

    @property
    def agent_app(self) -> "AgentApp":
        return self.harness_session.agent_app

    @property
    def session_manager(self) -> "SessionManager | None":
        return self.harness_session.session_manager

    @property
    def tools(self) -> RuntimeTools:
        return RuntimeTools(self.harness_session)

    @property
    def skills(self) -> RuntimeSkills:
        return RuntimeSkills(self)

    def agent(self, name: str | None = None) -> RuntimeAgent:
        return RuntimeAgent(self, name)


def _deduplicate_skills(manifests: Sequence[SkillManifest]) -> list[SkillManifest]:
    unique: dict[str, SkillManifest] = {}
    for manifest in manifests:
        key = strip_casefold(manifest.name)
        if key not in unique:
            unique[key] = manifest
    return list(unique.values())


class DefaultHarnessAppSession:
    """Default harness app session over one ``HarnessSession``."""

    def __init__(self, harness_session: "HarnessSession") -> None:
        self._env = AgentRuntimeEnvironment(harness_session)

    @property
    def agent_app(self) -> "AgentApp":
        return self._env.agent_app

    @property
    def env(self) -> AgentRuntimeEnvironment:
        return self._env

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        return await self._env.harness_session.invoke(request)


class HarnessSessionsAppProvider:
    """Adapt ``HarnessSessions`` to the default app's provider protocol."""

    def __init__(self, sessions: "HarnessSessions") -> None:
        self._sessions = sessions

    async def session(
        self,
        session_id: str | None = None,
        *,
        agent_name: str | None = None,
    ) -> "HarnessSession":
        return await self._sessions.get_or_create(session_id, agent_name=agent_name)


class DefaultHarnessApp:
    """Default app: open a harness session and expose AgentApp plus invoke()."""

    def __init__(self, session_provider: HarnessSessionProvider) -> None:
        self._session_provider = session_provider

    @asynccontextmanager
    async def open(
        self,
        request: AppOpenRequest | None = None,
    ) -> AsyncIterator[DefaultHarnessAppSession]:
        resolved = request or AppOpenRequest()
        session = await self._session_provider.session(
            resolved.session_id,
            agent_name=resolved.agent,
        )
        yield DefaultHarnessAppSession(session)


def load_harness_app(
    *,
    session_provider: HarnessSessionProvider,
    settings: "Settings | None" = None,
    entrypoint: str | None = None,
) -> HarnessApp:
    """Load the configured harness app or return the default app."""
    default_app = DefaultHarnessApp(session_provider)
    resolved_entrypoint = _resolve_harness_app_entrypoint(settings, entrypoint)
    if resolved_entrypoint is None:
        return default_app

    factory = _load_harness_app_factory(resolved_entrypoint)
    return factory(
        HarnessAppContext(
            default_app=default_app,
            session_provider=session_provider,
            settings=settings,
        )
    )


def _resolve_harness_app_entrypoint(
    settings: "Settings | None",
    entrypoint: str | None,
) -> str | None:
    configured = entrypoint
    if configured is None and settings is not None:
        configured = settings.harness_app.entrypoint
    if configured is None:
        return None
    normalized = configured.strip()
    return normalized or None


def _load_harness_app_factory(entrypoint: str) -> HarnessAppFactory:
    module_name, separator, attribute_name = entrypoint.partition(":")
    if not separator or not module_name.strip() or not attribute_name.strip():
        raise ValueError(
            "harness_app.entrypoint must use 'module:function' format, "
            f"got {entrypoint!r}."
        )

    module = import_module(module_name.strip())
    factory = getattr(module, attribute_name.strip())
    if not callable(factory):
        raise TypeError(f"harness_app.entrypoint {entrypoint!r} is not callable.")
    return factory


__all__ = [
    "AgentRuntimeEnvironment",
    "AppOpenRequest",
    "DefaultHarnessApp",
    "DefaultHarnessAppSession",
    "HarnessApp",
    "HarnessAppContext",
    "HarnessAppFactory",
    "HarnessAppSession",
    "HarnessSessionProvider",
    "HarnessSessionsAppProvider",
    "load_harness_app",
    "RuntimeAgent",
    "RuntimeSkillConfig",
    "RuntimeSkills",
    "RuntimeSkillSource",
    "RuntimeSkillTarget",
    "RuntimeTools",
]
