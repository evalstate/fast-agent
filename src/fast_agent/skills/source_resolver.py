"""Resolve configured and live skill install sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from fast_agent.skills.configuration import get_marketplace_url
from fast_agent.skills.marketplace_source import MarketplaceSkillSource
from fast_agent.skills.mcp_source import McpSkillSource, UnavailableMcpSkillSource

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext
    from fast_agent.skills.mcp_registry import McpSkillInstallClient, McpSkillRegistry
    from fast_agent.skills.models import SkillUpdateInfo
    from fast_agent.skills.sources import SkillInstallSource

MCP_REGISTRY_PREFIX = "mcp://"


@runtime_checkable
class McpSkillRegistryAggregator(Protocol):
    async def list_mcp_skill_registries(self) -> list[McpSkillRegistry]: ...


@runtime_checkable
class McpSkillRegistryAgent(Protocol):
    @property
    def aggregator(self) -> McpSkillRegistryAggregator: ...


@dataclass(frozen=True, slots=True)
class SkillSourceResolution:
    source: SkillInstallSource | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SkillUpdateSourceGroup:
    source: SkillInstallSource
    updates: list[SkillUpdateInfo]


def mcp_registry_source(server_name: str) -> str:
    return f"{MCP_REGISTRY_PREFIX}{server_name}"


def mcp_registry_server_name(source: str) -> str | None:
    if not source.startswith(MCP_REGISTRY_PREFIX):
        return None
    server_name = source[len(MCP_REGISTRY_PREFIX) :].strip()
    return server_name or None


def find_mcp_registry(
    registries: list[McpSkillRegistry],
    server_name: str,
) -> McpSkillRegistry | None:
    for registry in registries:
        if registry.server_name == server_name:
            return registry
    return None


class SkillSourceResolver:
    def __init__(self, ctx: "CommandContext", *, agent_name: str) -> None:
        self._ctx = ctx
        self._agent_name = agent_name
        self._mcp_registries: list[McpSkillRegistry] | None = None

    async def mcp_registries(self) -> list[McpSkillRegistry]:
        if self._mcp_registries is not None:
            return self._mcp_registries
        try:
            agent = self._ctx.agent_provider._agent(self._agent_name)
        except KeyError:
            self._mcp_registries = []
            return self._mcp_registries
        if not isinstance(agent, McpSkillRegistryAgent):
            self._mcp_registries = []
            return self._mcp_registries
        aggregator = agent.aggregator
        if not isinstance(aggregator, McpSkillRegistryAggregator):
            self._mcp_registries = []
            return self._mcp_registries
        self._mcp_registries = await aggregator.list_mcp_skill_registries()
        return self._mcp_registries

    async def active_source(self, *, override: str | None = None) -> SkillSourceResolution:
        settings = self._ctx.resolve_settings()
        source_url = (
            override
            or self._ctx.active_skill_source(self._agent_name)
            or get_marketplace_url(settings)
        )
        mcp_server_name = mcp_registry_server_name(source_url)
        if mcp_server_name is None:
            return SkillSourceResolution(source=MarketplaceSkillSource(source_url))

        registries = await self.mcp_registries()
        registry = find_mcp_registry(registries, mcp_server_name)
        if registry is None:
            return SkillSourceResolution(
                source=None,
                error=f"MCP skill registry is not available: {mcp_server_name}",
            )

        agent = self._ctx.agent_provider._agent(self._agent_name)
        if not isinstance(agent, McpSkillRegistryAgent):
            return SkillSourceResolution(
                source=None,
                error="This agent does not expose MCP skill registries.",
            )
        registry_client = cast("McpSkillInstallClient", agent.aggregator)
        return SkillSourceResolution(
            source=McpSkillSource(aggregator=registry_client, registry=registry)
        )

    async def update_sources(
        self,
        updates: list[SkillUpdateInfo],
    ) -> list[SkillUpdateSourceGroup]:
        if not updates:
            return []

        settings = self._ctx.resolve_settings()
        marketplace_updates: list[SkillUpdateInfo] = []
        mcp_updates_by_server: dict[str, list[SkillUpdateInfo]] = {}
        unavailable_mcp_updates: list[SkillUpdateInfo] = []

        for update in updates:
            managed_source = update.managed_source
            if managed_source is None or managed_source.source_origin != "mcp":
                marketplace_updates.append(update)
                continue
            if managed_source.mcp_server_name is None:
                unavailable_mcp_updates.append(update)
                continue
            mcp_updates_by_server.setdefault(managed_source.mcp_server_name, []).append(update)

        groups: list[SkillUpdateSourceGroup] = []
        if marketplace_updates:
            groups.append(
                SkillUpdateSourceGroup(
                    source=MarketplaceSkillSource(get_marketplace_url(settings)),
                    updates=marketplace_updates,
                )
            )

        if unavailable_mcp_updates:
            groups.append(
                SkillUpdateSourceGroup(
                    source=UnavailableMcpSkillSource(
                        server_name="unknown",
                        detail="MCP source metadata is missing the server name.",
                    ),
                    updates=unavailable_mcp_updates,
                )
            )

        if not mcp_updates_by_server:
            return groups

        registries = await self.mcp_registries()
        try:
            agent = self._ctx.agent_provider._agent(self._agent_name)
        except KeyError:
            agent = None
        registry_client = (
            cast("McpSkillInstallClient", agent.aggregator)
            if isinstance(agent, McpSkillRegistryAgent)
            else None
        )

        for server_name, source_updates in mcp_updates_by_server.items():
            registry = find_mcp_registry(registries, server_name)
            if registry is None:
                source: SkillInstallSource = UnavailableMcpSkillSource(
                    server_name=server_name,
                    detail=f"MCP skill registry is not available: {server_name}",
                )
            elif registry_client is None:
                source = UnavailableMcpSkillSource(
                    server_name=server_name,
                    detail="This agent does not expose MCP skill registries.",
                )
            else:
                source = McpSkillSource(aggregator=registry_client, registry=registry)
            groups.append(SkillUpdateSourceGroup(source=source, updates=source_updates))

        return groups
