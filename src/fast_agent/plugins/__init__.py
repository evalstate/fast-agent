"""First-class fast-agent plugin support."""

from fast_agent.plugins.models import (
    LocalPlugin,
    MarketplacePlugin,
    PluginManifest,
    PluginPostUserTurnContext,
    PluginPostUserTurnFunction,
    PluginPostUserTurnOutput,
    PluginPostUserTurnResult,
    PluginUpdateInfo,
)

__all__ = [
    "LocalPlugin",
    "MarketplacePlugin",
    "PluginManifest",
    "PluginPostUserTurnContext",
    "PluginPostUserTurnFunction",
    "PluginPostUserTurnOutput",
    "PluginPostUserTurnResult",
    "PluginUpdateInfo",
]
