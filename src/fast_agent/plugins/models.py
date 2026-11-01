"""Data models for first-class fast-agent plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, Protocol

from fast_agent.marketplace.provenance_io import repo_subdir_for_manifest_path
from fast_agent.marketplace.update_status import CommonMarketplaceUpdateStatus
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping

    from mcp_types import Tool

    from fast_agent.command_actions.models import PluginCommandActionSpec
    from fast_agent.llm.usage_tracking import TurnUsage
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

DEFAULT_PLUGIN_REGISTRIES = [
    "https://github.com/fast-agent-ai/card-packs",
]

DEFAULT_PLUGIN_MARKETPLACE_URL = (
    "https://github.com/fast-agent-ai/card-packs/blob/main/marketplace.json"
)

PLUGIN_MANIFEST_FILENAME = "plugin.yaml"
PLUGIN_SOURCE_FILENAME = ".plugin-source.json"
PLUGIN_SOURCE_SCHEMA_VERSION = 1
LOCAL_REVISION = "local"

PluginSourceOrigin = Literal["remote", "local"]
PluginUpdateStatus = CommonMarketplaceUpdateStatus | Literal["invalid_local_plugin"]


@dataclass(frozen=True)
class PluginManifest:
    schema_version: int
    name: str
    version: str | None
    description: str | None
    commands: dict[str, PluginCommandActionSpec]
    post_user_turn: PluginPostUserTurnSpec | None
    path: Path
    tool_call_review: PluginToolCallReviewSpec | None = None


@dataclass(frozen=True)
class PluginPostUserTurnSpec:
    plugin_name: str
    handler: str


@dataclass(frozen=True)
class PluginToolCallReviewSpec:
    plugin_name: str
    handler: str


@dataclass(frozen=True, slots=True)
class PluginPostUserTurnOutput:
    """Optional terminal and external-status projections from a post-turn plugin."""

    display: str | None = None
    session_usage: str | None = None


type PluginPostUserTurnResult = str | PluginPostUserTurnOutput | None
type PluginPostUserTurnReturn = PluginPostUserTurnResult | Awaitable[PluginPostUserTurnResult]


@dataclass(frozen=True, slots=True)
class PluginPostUserTurnContext:
    """Canonical usage exposed after one successful top-level interactive turn."""

    plugin_name: str
    agent_name: str
    turn_usage: tuple[TurnUsage, ...]
    session_usage: tuple[TurnUsage, ...]
    config: Mapping[str, object]
    turn_messages: tuple[PromptMessageExtended, ...] = ()
    """The completed turn in the selected agent's history, as stored (process polls
    may be folded; see ``restore_process_poll_history``)."""


class PluginPostUserTurnFunction(Protocol):
    __name__: str

    def __call__(self, ctx: PluginPostUserTurnContext) -> PluginPostUserTurnReturn: ...


@dataclass(frozen=True, slots=True)
class PluginToolCall:
    """One tool call awaiting review, after fast-agent resolved its tool name."""

    id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PluginToolCallReviewContext:
    """What a reviewer sees: the conversation so far and the calls about to run."""

    plugin_name: str
    agent_name: str
    model: str | None
    instruction: str
    """The agent's system prompt."""
    tools: tuple[Tool, ...]
    """Tools offered to the model for this request."""
    history: tuple[PromptMessageExtended, ...]
    """Conversation before ``request``."""
    request: PromptMessageExtended
    """The assistant message making the calls."""
    tool_calls: tuple[PluginToolCall, ...]
    config: Mapping[str, object]
    """The plugin's mapping from ``plugins.config.<plugin-name>``."""


type PluginToolCallReviewResult = Mapping[str, str] | None
"""Denied call ids mapped to the error text returned to the model; ``None`` allows all."""


class PluginToolCallReviewFunction(Protocol):
    __name__: str

    def __call__(
        self, ctx: PluginToolCallReviewContext
    ) -> PluginToolCallReviewResult | Awaitable[PluginToolCallReviewResult]: ...


@dataclass(frozen=True)
class PluginContributions:
    commands: dict[str, PluginCommandActionSpec]
    post_user_turn: dict[str, PluginPostUserTurnSpec]
    tool_call_review: dict[str, PluginToolCallReviewSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketplacePlugin:
    name: str
    description: str | None
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None = None
    bundle_name: str | None = None
    version: str | None = None
    path_oid: str | None = None
    """Git tree id of ``repo_path`` at the published revision."""
    requires_fast_agent: str | None = None
    bundles: tuple[str, ...] = ()
    """Names of marketplace ``plugin_bundles`` that include this plugin."""

    @property
    def repo_subdir(self) -> str:
        return repo_subdir_for_manifest_path(self.repo_path, PLUGIN_MANIFEST_FILENAME)

    @property
    def install_dir_name(self) -> str:
        path = PurePosixPath(self.repo_path)
        if strip_casefold(path.name) == PLUGIN_MANIFEST_FILENAME:
            return path.parent.name or self.name
        return path.name or self.name


@dataclass(frozen=True)
class InstalledPluginSource:
    schema_version: int
    installed_via: str
    source_origin: PluginSourceOrigin
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None
    installed_commit: str | None
    installed_path_oid: str | None
    installed_revision: str
    installed_at: str
    content_fingerprint: str


@dataclass(frozen=True)
class LocalPlugin:
    index: int
    name: str
    plugin_dir: Path
    manifest: PluginManifest | None
    source: InstalledPluginSource | None
    metadata_error: str | None = None
    manifest_error: str | None = None


@dataclass(frozen=True)
class PluginUpdateInfo:
    index: int
    name: str
    plugin_dir: Path
    status: PluginUpdateStatus
    detail: str | None = None
    current_revision: str | None = None
    available_revision: str | None = None
    managed_source: InstalledPluginSource | None = None
