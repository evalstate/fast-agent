"""AgentCard load, reload, and watch helpers for FastAgent."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fast_agent import config
from fast_agent.agents.agent_types import AgentConfig, MCPConnectTarget
from fast_agent.core.agent_app import AgentRefreshResult
from fast_agent.core.agent_card_paths import is_agent_card_path, is_markdown_agent_card_path
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.connect_targets import resolve_target_entry
from fast_agent.tools.function_tool_config import function_tool_entrypoint
from fast_agent.tools.python_file_loader import parse_callable_file_spec
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from fast_agent.config import MCPServerSettings
    from fast_agent.core.agent_card_loader import LoadedAgentCard
    from fast_agent.core.agent_card_types import AgentCardData
    from fast_agent.skills import SkillManifest

FileSignature = tuple[int, int]
logger = get_logger(__name__)


@dataclass(frozen=True)
class CardFileSnapshot:
    current_files: set[Path]
    removed_files: set[Path]
    current_stats: dict[Path, FileSignature]
    changed_files: set[Path]


@dataclass(frozen=True)
class WatchFileSnapshot:
    watch_files: set[Path]
    removed_files: set[Path]
    current_stats: dict[Path, FileSignature]
    changed_files: set[Path]
    current_tool_files: set[Path]


@dataclass(frozen=True)
class ResolvedMCPConnectEntry:
    name: str
    settings: "MCPServerSettings"
    target_label: str


class AgentCardRuntimeMixin:
    agents: dict[str, "AgentCardData"]
    app: Any
    _agent_card_roots: dict[Path, set[str]]
    _agent_card_reload_lock: asyncio.Lock | None
    _agent_card_last_changed: set[str]
    _agent_card_last_removed: set[str]
    _agent_card_last_dependents: set[str]
    _agent_registry_version: int
    _agent_card_file_cache: dict[Path, FileSignature]
    _agent_card_root_files: dict[Path, set[Path]]
    _agent_card_root_watch_files: dict[Path, set[Path]]
    _agent_card_tool_files: dict[Path, set[Path]]
    _agent_card_histories: dict[str, list[Path]]
    _agent_card_history_mtime: dict[str, float]
    _agent_card_history_len: dict[str, int]
    _agent_card_sources: dict[str, Path]
    _agent_card_name_by_path: dict[Path, str]
    _agent_card_watch_reload: "Callable[[], Awaitable[AgentRefreshResult]] | None"
    _agent_declared_servers: dict[str, list[str]]
    _base_mcp_servers: dict[str, "MCPServerSettings"] | None
    _card_collision_warnings: list[str]
    _card_mcp_owned_servers: dict[str, set[str]]
    _default_skill_manifests: list["SkillManifest"]
    _dynamic_mcp_server_names: set[str]

    def _apply_skills_to_agent_configs(self, default_skills: list["SkillManifest"]) -> None: ...

    async def reload_agents(self) -> bool:
        """Reload all previously registered AgentCard roots."""
        if not self._agent_card_roots:
            return False

        if self._agent_card_reload_lock is None:
            self._agent_card_reload_lock = asyncio.Lock()

        async with self._agent_card_reload_lock:
            self._agent_card_last_changed = set()
            self._agent_card_last_removed = set()
            self._agent_card_last_dependents = set()
            changed = False
            for root in sorted(self._agent_card_roots.keys()):
                if self._load_agent_cards_from_root(root, incremental=True):
                    changed = True

            if changed:
                self._agent_registry_version += 1
            return changed

    def _resolve_current_card_files(self, root: Path, *, incremental: bool) -> set[Path]:
        if root.exists():
            return self._collect_agent_card_files(root)
        if incremental:
            return set()
        raise AgentConfigError(f"AgentCard path not found: {root}")

    @staticmethod
    def _collect_existing_file_signatures(
        paths: set[Path],
    ) -> tuple[set[Path], dict[Path, FileSignature]]:
        existing_paths = set(paths)
        current_stats: dict[Path, FileSignature] = {}
        for path_entry in list(existing_paths):
            try:
                stat = path_entry.stat()
            except FileNotFoundError:
                existing_paths.discard(path_entry)
                continue
            current_stats[path_entry] = (stat.st_mtime_ns, stat.st_size)
        return existing_paths, current_stats

    def _select_changed_files(
        self,
        current_stats: dict[Path, FileSignature],
        *,
        incremental: bool,
    ) -> set[Path]:
        if not incremental:
            return set(current_stats.keys())
        return {
            path_entry
            for path_entry, signature in current_stats.items()
            if self._agent_card_file_cache.get(path_entry) != signature
        }

    def _build_card_file_snapshot(self, root: Path, *, incremental: bool) -> CardFileSnapshot:
        current_files = self._resolve_current_card_files(root, incremental=incremental)
        previous_files = self._agent_card_root_files.get(root, set())
        removed_files = previous_files - current_files
        current_files, current_stats = self._collect_existing_file_signatures(current_files)
        changed_files = self._select_changed_files(current_stats, incremental=incremental)
        return CardFileSnapshot(
            current_files=current_files,
            removed_files=removed_files,
            current_stats=current_stats,
            changed_files=changed_files,
        )

    def _load_agent_cards_from_file(
        self,
        path_entry: Path,
        *,
        incremental: bool,
    ) -> list["LoadedAgentCard"]:
        from fast_agent.core.agent_card_loader import load_agent_cards

        try:
            return load_agent_cards(path_entry)
        except AgentConfigError as exc:
            if not incremental:
                raise
            if "Instruction is required" in exc.message:
                logger.warning(
                    "Skipping incomplete AgentCard during reload; waiting for write to complete",
                    path=str(path_entry),
                )
                return []
            logger.warning(
                "Skipping invalid AgentCard during reload",
                path=str(path_entry),
                error=str(exc),
            )
            return []
        except Exception as exc:
            if not incremental:
                raise
            logger.warning(
                "Skipping invalid AgentCard during reload",
                path=str(path_entry),
                error=str(exc),
            )
            return []

    def _record_loaded_card_tool_files(self, cards: "Sequence[LoadedAgentCard]") -> None:
        for card in cards:
            config = card.agent_data.get("config")
            function_tools = getattr(config, "function_tools", None)
            self._agent_card_tool_files[card.path] = self._resolve_function_tool_paths(
                card.path,
                function_tools,
            )

    def _load_cards_for_paths(
        self,
        path_entries: set[Path],
        *,
        incremental: bool,
        cards: list["LoadedAgentCard"],
        loaded_card_files: set[Path],
    ) -> None:
        for path_entry in sorted(path_entries):
            loaded_cards = self._load_agent_cards_from_file(path_entry, incremental=incremental)
            cards.extend(loaded_cards)
            loaded_card_files.add(path_entry)
            self._record_loaded_card_tool_files(loaded_cards)

    def _collect_current_tool_files(self, current_card_files: set[Path]) -> set[Path]:
        current_tool_files: set[Path] = set()
        for card_path in current_card_files:
            current_tool_files.update(self._agent_card_tool_files.get(card_path, set()))
        return current_tool_files

    def _collect_current_history_files_for_root(
        self,
        root: Path,
        cards: "Sequence[LoadedAgentCard]",
    ) -> set[Path]:
        current_history_files: set[Path] = set()
        for history_files in self._agent_card_histories.values():
            for history_file in history_files:
                try:
                    if history_file.is_relative_to(root):
                        current_history_files.add(history_file)
                except ValueError:
                    continue
        for card in cards:
            for history_file in card.message_files or []:
                try:
                    if history_file.is_relative_to(root):
                        current_history_files.add(history_file)
                except ValueError:
                    continue
        return current_history_files

    def _build_watch_file_snapshot(
        self,
        root: Path,
        *,
        current_card_files: set[Path],
        cards: "Sequence[LoadedAgentCard]",
        incremental: bool,
    ) -> WatchFileSnapshot:
        current_tool_files = self._collect_current_tool_files(current_card_files)
        current_history_files = self._collect_current_history_files_for_root(root, cards)
        watch_files = set(current_card_files) | current_tool_files | current_history_files
        previous_watch_files = self._agent_card_root_watch_files.get(root, set())
        removed_files = previous_watch_files - watch_files
        watch_files, current_stats = self._collect_existing_file_signatures(watch_files)
        changed_files = self._select_changed_files(current_stats, incremental=incremental)
        return WatchFileSnapshot(
            watch_files=watch_files,
            removed_files=removed_files,
            current_stats=current_stats,
            changed_files=changed_files,
            current_tool_files=current_tool_files,
        )

    @staticmethod
    def _get_changed_tool_files(
        current_card_files: set[Path],
        watch_snapshot: WatchFileSnapshot,
    ) -> set[Path]:
        changed_tool_files = {
            path_entry
            for path_entry in watch_snapshot.changed_files
            if path_entry in watch_snapshot.current_tool_files
        }
        removed_tool_files = {
            path_entry
            for path_entry in watch_snapshot.removed_files
            if path_entry not in current_card_files
        }
        return changed_tool_files | removed_tool_files

    def _reload_cards_affected_by_tool_changes(
        self,
        current_card_files: set[Path],
        changed_tool_files: set[Path],
        *,
        incremental: bool,
        cards: list["LoadedAgentCard"],
        loaded_card_files: set[Path],
    ) -> None:
        if not changed_tool_files:
            return

        affected_card_files = {
            card_path
            for card_path in current_card_files
            if self._agent_card_tool_files.get(card_path, set()) & changed_tool_files
        }
        self._load_cards_for_paths(
            affected_card_files - loaded_card_files,
            incremental=incremental,
            cards=cards,
            loaded_card_files=loaded_card_files,
        )

    def _load_agent_cards_from_root(self, root: Path, *, incremental: bool) -> bool:
        card_snapshot = self._build_card_file_snapshot(root, incremental=incremental)
        for removed_path in card_snapshot.removed_files:
            self._agent_card_tool_files.pop(removed_path, None)

        cards: list[LoadedAgentCard] = []
        loaded_card_files: set[Path] = set()
        self._load_cards_for_paths(
            card_snapshot.changed_files,
            incremental=incremental,
            cards=cards,
            loaded_card_files=loaded_card_files,
        )

        missing_tool_cards = {
            path_entry
            for path_entry in card_snapshot.current_files
            if path_entry not in self._agent_card_tool_files
        }
        self._load_cards_for_paths(
            missing_tool_cards,
            incremental=incremental,
            cards=cards,
            loaded_card_files=loaded_card_files,
        )

        watch_snapshot = self._build_watch_file_snapshot(
            root,
            current_card_files=card_snapshot.current_files,
            cards=cards,
            incremental=incremental,
        )
        changed_tool_files = self._get_changed_tool_files(
            card_snapshot.current_files,
            watch_snapshot,
        )
        self._reload_cards_affected_by_tool_changes(
            card_snapshot.current_files,
            changed_tool_files,
            incremental=incremental,
            cards=cards,
            loaded_card_files=loaded_card_files,
        )

        self._apply_agent_card_updates(
            root,
            current_files=card_snapshot.current_files,
            removed_files=card_snapshot.removed_files,
            changed_cards=cards,
            current_stats=watch_snapshot.current_stats,
        )

        for path_entry in watch_snapshot.removed_files:
            self._agent_card_file_cache.pop(path_entry, None)

        self._agent_card_root_watch_files[root] = set(watch_snapshot.watch_files)

        return bool(card_snapshot.removed_files or watch_snapshot.changed_files)

    def _collect_agent_card_files(self, root: Path) -> set[Path]:
        def _has_frontmatter(path: Path) -> bool:
            try:
                raw_text = path.read_text(encoding="utf-8")
            except Exception:
                return False
            if raw_text.startswith("\ufeff"):
                raw_text = raw_text.lstrip("\ufeff")
            for line in raw_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                return stripped in ("---", "+++")
            return False

        if root.is_dir():
            return {
                entry
                for entry in root.iterdir()
                if entry.is_file()
                and is_agent_card_path(entry)
                and (not is_markdown_agent_card_path(entry) or _has_frontmatter(entry))
            }

        if not is_agent_card_path(root):
            raise AgentConfigError(f"Unsupported AgentCard file extension: {root}")
        if is_markdown_agent_card_path(root) and not _has_frontmatter(root):
            raise AgentConfigError(
                "AgentCard markdown files must include frontmatter",
                f"Missing frontmatter in {root}",
            )
        return {root}

    @staticmethod
    def _resolve_function_tool_paths(
        card_path: Path,
        function_tools: "Sequence[object] | None",
    ) -> set[Path]:
        tool_paths: set[Path] = set()
        if not function_tools:
            return tool_paths
        for spec in function_tools:
            entrypoint = function_tool_entrypoint(spec)
            if not entrypoint:
                continue
            try:
                parsed = parse_callable_file_spec(
                    entrypoint,
                    invalid_message="Invalid function tool spec '{spec}'",
                )
            except AgentConfigError:
                continue

            module_path = Path(parsed.module_path_text)
            if not module_path.is_absolute():
                module_path = (card_path.parent / module_path).resolve()
            if strip_casefold(module_path.suffix) == ".py":
                tool_paths.add(module_path)
        return tool_paths

    def _removed_agent_card_names(self, removed_files: set[Path]) -> set[str]:
        return {
            self._agent_card_name_by_path[path]
            for path in removed_files
            if path in self._agent_card_name_by_path
        }

    def _agent_card_dependents(self, agent_names: set[str]) -> set[str]:
        if not agent_names:
            return set()

        from fast_agent.core.validation import get_agent_dependencies

        return {
            name
            for name, agent_data in self.agents.items()
            if get_agent_dependencies(agent_data) & agent_names
        }

    @staticmethod
    def _is_tool_card_path(path: Path) -> bool:
        return path.parent.name == "tool-cards"

    def _handle_agent_card_source_collision(
        self,
        card: "LoadedAgentCard",
        existing_source: Path,
        removed_files: set[Path],
        removed_names: set[str],
    ) -> bool:
        if existing_source in removed_files:
            return False

        existing_is_tool = self._is_tool_card_path(existing_source)
        new_is_tool = self._is_tool_card_path(card.path)
        if existing_is_tool == new_is_tool:
            raise AgentConfigError(
                f"Agent '{card.name}' already loaded from {existing_source}",
                f"Path: {card.path}",
            )

        if new_is_tool:
            warning_msg = (
                f"Agent '{card.name}' defined in both tool-cards and agent-cards. "
                f"Using tool-card version from {card.path}. "
                f"Skipping agent-card at {existing_source}."
            )
            print(f"Warning: {warning_msg}", file=sys.stderr)
            self._card_collision_warnings.append(warning_msg)
            removed_names.add(card.name)
            return False

        warning_msg = (
            f"Agent '{card.name}' defined in both tool-cards and agent-cards. "
            f"Using tool-card version from {existing_source}. "
            f"Skipping agent-card at {card.path}."
        )
        print(f"Warning: {warning_msg}", file=sys.stderr)
        self._card_collision_warnings.append(warning_msg)
        return True

    def _validate_changed_agent_cards(
        self,
        root: Path,
        *,
        removed_files: set[Path],
        changed_cards: list["LoadedAgentCard"],
        removed_names: set[str],
    ) -> set[str]:
        seen_names: dict[str, Path] = {}
        changed_names: set[str] = set()
        for card in changed_cards:
            changed_names.add(card.name)
            if card.name in seen_names:
                raise AgentConfigError(
                    f"Duplicate agent name '{card.name}' during reload",
                    f"Conflicts: {seen_names[card.name]} and {card.path}",
                )
            seen_names[card.name] = card.path

            existing_source = self._agent_card_sources.get(card.name)
            if card.name in self.agents and existing_source is None:
                raise AgentConfigError(
                    f"Agent '{card.name}' already exists and is not loaded from AgentCard",
                    f"Path: {root}",
                )
            if existing_source is not None and existing_source != card.path:
                should_skip = self._handle_agent_card_source_collision(
                    card,
                    existing_source,
                    removed_files,
                    removed_names,
                )
                if should_skip:
                    continue

            previous_name = self._agent_card_name_by_path.get(card.path)
            if previous_name and previous_name != card.name:
                removed_names.add(previous_name)

        return changed_names

    def _remove_agent_cards(self, removed_names: set[str], removed_files: set[Path]) -> None:
        for name in sorted(removed_names):
            self.agents.pop(name, None)
            self._agent_card_sources.pop(name, None)
            self._agent_card_histories.pop(name, None)
            self._agent_card_history_mtime.pop(name, None)
            self._agent_card_history_len.pop(name, None)

        for path_entry in removed_files:
            self._agent_card_name_by_path.pop(path_entry, None)
            self._agent_card_file_cache.pop(path_entry, None)

    def _register_changed_agent_cards(self, changed_cards: list["LoadedAgentCard"]) -> None:
        for card in changed_cards:
            self.agents[card.name] = card.agent_data
            self._agent_card_sources[card.name] = card.path
            self._agent_card_name_by_path[card.path] = card.name

            if card.message_files:
                self._agent_card_histories[card.name] = card.message_files
            else:
                self._agent_card_histories.pop(card.name, None)
                self._agent_card_history_mtime.pop(card.name, None)
                self._agent_card_history_len.pop(card.name, None)

    def _prune_removed_child_agents(self, removed_names: set[str]) -> None:
        if not removed_names:
            return

        for agent_data in self.agents.values():
            child_agents = agent_data.get("child_agents")
            if not child_agents:
                continue
            pruned = [name for name in child_agents if name not in removed_names]
            if pruned != child_agents:
                agent_data["child_agents"] = pruned

    def _cache_agent_card_files(
        self,
        root: Path,
        *,
        current_files: set[Path],
        current_stats: dict[Path, tuple[int, int]],
    ) -> None:
        for path_entry, signature in current_stats.items():
            self._agent_card_file_cache[path_entry] = signature

        self._agent_card_root_files[root] = set(current_files)
        self._agent_card_roots[root] = {
            self._agent_card_name_by_path[path_entry]
            for path_entry in current_files
            if path_entry in self._agent_card_name_by_path
        }

    def _record_agent_card_reload_changes(
        self,
        *,
        changed_names: set[str],
        removed_names: set[str],
        removed_dependents: set[str],
    ) -> None:
        self._apply_skills_to_agent_configs(self._default_skill_manifests)
        self._agent_card_last_changed.update(changed_names)
        self._agent_card_last_removed.update(removed_names)
        self._agent_card_last_dependents.update(removed_dependents)
        self._agent_card_last_dependents.update(
            self._agent_card_dependents(changed_names) - changed_names
        )

    def _apply_agent_card_updates(
        self,
        root: Path,
        *,
        current_files: set[Path],
        removed_files: set[Path],
        changed_cards: list["LoadedAgentCard"],
        current_stats: dict[Path, tuple[int, int]],
    ) -> None:
        removed_names = self._removed_agent_card_names(removed_files)
        removed_dependents = self._agent_card_dependents(removed_names)
        changed_names = self._validate_changed_agent_cards(
            root,
            removed_files=removed_files,
            changed_cards=changed_cards,
            removed_names=removed_names,
        )

        self._remove_agent_cards(removed_names, removed_files)
        self._register_changed_agent_cards(changed_cards)
        self._prune_removed_child_agents(removed_names)
        self._cache_agent_card_files(root, current_files=current_files, current_stats=current_stats)

        if changed_cards or removed_files:
            self._record_agent_card_reload_changes(
                changed_names=changed_names,
                removed_names=removed_names,
                removed_dependents=removed_dependents,
            )

    @staticmethod
    def _settings_signature(settings: "MCPServerSettings") -> dict[str, Any]:
        return settings.model_dump(exclude_none=True)

    @staticmethod
    def _settings_equivalent(left: "MCPServerSettings", right: "MCPServerSettings") -> bool:
        return AgentCardRuntimeMixin._settings_signature(left) == (
            AgentCardRuntimeMixin._settings_signature(right)
        )

    @staticmethod
    def _copy_server_settings(settings: "MCPServerSettings", *, name: str) -> "MCPServerSettings":
        copied = settings.model_copy(deep=True)
        copied.name = name
        return copied

    def _ensure_app_mcp_settings(self):
        context = getattr(self.app, "context", None)
        if context is None:
            return None

        app_config = getattr(context, "config", None)
        if app_config is None:
            return None

        if app_config.mcp is None:
            app_config.mcp = config.MCPSettings()
        return context, app_config

    def _remember_base_mcp_servers(
        self,
        configured_servers: dict[str, "MCPServerSettings"],
    ) -> None:
        if self._base_mcp_servers is not None:
            return
        self._base_mcp_servers = {
            name: self._copy_server_settings(server, name=name)
            for name, server in configured_servers.items()
        }

    def _preserved_runtime_mcp_servers(
        self,
        existing_registry: object,
    ) -> dict[str, "MCPServerSettings"]:
        if not isinstance(existing_registry, dict):
            return {}

        preserved: dict[str, MCPServerSettings] = {}
        for name, server in existing_registry.items():
            if not isinstance(name, str):
                continue
            if name in self._dynamic_mcp_server_names:
                continue
            if not isinstance(server, config.MCPServerSettings):
                continue
            preserved[name] = self._copy_server_settings(server, name=name)
        return preserved

    def _effective_mcp_servers(
        self,
        preserved_runtime_servers: dict[str, "MCPServerSettings"],
    ) -> dict[str, "MCPServerSettings"]:
        effective_servers: dict[str, MCPServerSettings] = {
            name: self._copy_server_settings(server, name=name)
            for name, server in (self._base_mcp_servers or {}).items()
        }
        for name, server in preserved_runtime_servers.items():
            effective_servers.setdefault(name, server)
        return effective_servers

    @staticmethod
    def _connector_mcp_payload(entry: MCPConnectTarget, explicit_name: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": explicit_name,
            "description": entry.description,
            "management": entry.management,
            "connector_id": entry.connector_id,
            "headers": dict(entry.headers) if entry.headers is not None else None,
            "access_token": entry.access_token,
            "auth": dict(entry.auth) if entry.auth is not None else None,
        }
        if entry.defer_loading is not None:
            payload["defer_loading"] = entry.defer_loading
        return payload

    @staticmethod
    def _target_mcp_overrides(entry: MCPConnectTarget) -> dict[str, Any]:
        optional_fields = {
            "description": entry.description,
            "management": entry.management,
            "headers": dict(entry.headers) if entry.headers is not None else None,
            "access_token": entry.access_token,
            "defer_loading": entry.defer_loading,
            "auth": dict(entry.auth) if entry.auth is not None else None,
        }
        return {name: value for name, value in optional_fields.items() if value is not None}

    def _resolve_connector_mcp_entry(
        self,
        agent_name: str,
        index: int,
        entry: MCPConnectTarget,
        explicit_name: str | None,
    ) -> ResolvedMCPConnectEntry:
        if entry.target is not None:
            raise AgentConfigError(
                f"Invalid mcp_connect entry for agent '{agent_name}'",
                f"Entry {index}: target must be omitted when connector_id is set",
            )
        if explicit_name is None:
            raise AgentConfigError(
                f"Invalid mcp_connect entry for agent '{agent_name}'",
                f"Entry {index}: name is required when connector_id is set",
            )

        try:
            settings = config.MCPServerSettings.model_validate(
                self._connector_mcp_payload(entry, explicit_name)
            )
        except Exception as exc:
            raise AgentConfigError(
                f"Invalid mcp_connect entry for agent '{agent_name}'",
                f"Entry {index} connector_id '{entry.connector_id}': {exc}",
            ) from exc

        return ResolvedMCPConnectEntry(
            name=explicit_name,
            settings=settings,
            target_label=str(entry.connector_id),
        )

    def _resolve_target_mcp_entry(
        self,
        agent_name: str,
        index: int,
        entry: MCPConnectTarget,
        explicit_name: str | None,
    ) -> ResolvedMCPConnectEntry:
        target = entry.target
        if target is None:
            raise AgentConfigError(
                f"Invalid mcp_connect entry for agent '{agent_name}'",
                f"Entry {index}: target must be a non-empty string",
            )

        try:
            resolved_entry = resolve_target_entry(
                target=target,
                default_name=explicit_name,
                overrides=self._target_mcp_overrides(entry),
                source_path=f"mcp_connect[{index}].target",
            )
        except Exception as exc:
            raise AgentConfigError(
                f"Invalid mcp_connect entry for agent '{agent_name}'",
                f"Entry {index} target '{target}': {exc}",
            ) from exc

        return ResolvedMCPConnectEntry(
            name=resolved_entry.server_name,
            settings=resolved_entry.settings,
            target_label=target,
        )

    def _resolve_mcp_connect_entry(
        self,
        agent_name: str,
        index: int,
        entry: MCPConnectTarget,
    ) -> ResolvedMCPConnectEntry:
        if entry.connector_id is not None:
            return self._resolve_connector_mcp_entry(agent_name, index, entry, entry.name)
        return self._resolve_target_mcp_entry(agent_name, index, entry, entry.name)

    def _resolve_agent_mcp_connect_servers(
        self,
        effective_servers: dict[str, "MCPServerSettings"],
    ) -> tuple[dict[str, list[str]], dict[str, set[str]], set[str]]:
        resolved_servers_by_agent: dict[str, list[str]] = {}
        card_owners: dict[str, set[str]] = {}
        all_dynamic_server_names: set[str] = set()

        for agent_name in sorted(self.agents.keys()):
            agent_data = self.agents[agent_name]
            config_obj = agent_data.get("config")
            if not isinstance(config_obj, AgentConfig):
                continue

            entries = config_obj.mcp_connect
            if not entries:
                resolved_servers_by_agent[agent_name] = []
                continue

            owner = str(config_obj.source_path or f"agent:{agent_name}")
            for index, entry in enumerate(entries):
                resolved = self._resolve_mcp_connect_entry(agent_name, index, entry)
                existing = effective_servers.get(resolved.name)
                if existing is not None and not self._settings_equivalent(
                    existing,
                    resolved.settings,
                ):
                    raise AgentConfigError(
                        (
                            f"Server name collision for '{resolved.name}' from mcp_connect "
                            f"target '{resolved.target_label}'."
                        ),
                        "Set an explicit unique `name` or change target.",
                    )

                if existing is None:
                    effective_servers[resolved.name] = self._copy_server_settings(
                        resolved.settings,
                        name=resolved.name,
                    )

                resolved_servers_by_agent.setdefault(agent_name, []).append(resolved.name)
                card_owners.setdefault(owner, set()).add(resolved.name)
                all_dynamic_server_names.add(resolved.name)

        return resolved_servers_by_agent, card_owners, all_dynamic_server_names

    def _merge_agent_mcp_servers(
        self,
        resolved_servers_by_agent: dict[str, list[str]],
    ) -> None:
        active_agent_names = set(self.agents.keys())
        for name, agent_data in self.agents.items():
            config_obj = agent_data.get("config")
            if not isinstance(config_obj, AgentConfig):
                continue

            current_declared = list(config_obj.servers)
            if name not in self._agent_declared_servers or name in self._agent_card_last_changed:
                self._agent_declared_servers[name] = current_declared

            base_servers = list(self._agent_declared_servers.get(name, []))
            self._agent_declared_servers[name] = base_servers
            config_obj.servers = list(
                dict.fromkeys(base_servers + resolved_servers_by_agent.get(name, []))
            )

        for name in list(self._agent_declared_servers.keys()):
            if name not in active_agent_names:
                self._agent_declared_servers.pop(name, None)

    def _publish_effective_mcp_servers(
        self,
        app_config: Any,
        registry: Any,
        effective_servers: dict[str, "MCPServerSettings"],
    ) -> None:
        app_config.mcp.servers = {
            name: self._copy_server_settings(server, name=name)
            for name, server in effective_servers.items()
        }
        if registry is not None:
            registry.registry = {
                name: self._copy_server_settings(server, name=name)
                for name, server in effective_servers.items()
            }

    def _sync_agent_card_mcp_servers(self) -> None:
        sync_context = self._ensure_app_mcp_settings()
        if sync_context is None:
            return
        context, app_config = sync_context

        self._remember_base_mcp_servers(app_config.mcp.servers or {})

        registry = getattr(context, "server_registry", None)
        existing_registry = getattr(registry, "registry", {}) if registry is not None else {}
        effective_servers = self._effective_mcp_servers(
            self._preserved_runtime_mcp_servers(existing_registry)
        )
        resolved_servers_by_agent, card_owners, all_dynamic_server_names = (
            self._resolve_agent_mcp_connect_servers(effective_servers)
        )

        self._merge_agent_mcp_servers(resolved_servers_by_agent)
        self._publish_effective_mcp_servers(app_config, registry, effective_servers)
        self._card_mcp_owned_servers = card_owners
        self._dynamic_mcp_server_names = all_dynamic_server_names

    async def _watch_agent_cards(self) -> None:
        roots = sorted(self._agent_card_roots.keys())
        if not roots:
            return

        try:
            from watchfiles import awatch

            async for _changes in awatch(*roots):
                await self._reload_agent_cards_from_watch()
        except ImportError:
            logger.info("watchfiles not available; falling back to polling for AgentCard reloads")
            try:
                while True:
                    await asyncio.sleep(1.0)
                    await self._reload_agent_cards_from_watch()
            except asyncio.CancelledError:
                return
        except asyncio.CancelledError:
            return

    async def _reload_agent_cards_from_watch(self) -> bool:
        reload_callback = self._agent_card_watch_reload
        if reload_callback is None:
            return await self.reload_agents()
        result = await reload_callback()
        if isinstance(result, AgentRefreshResult):
            return result.changed
        return result


__all__ = [
    "AgentCardRuntimeMixin",
    "CardFileSnapshot",
    "FileSignature",
    "ResolvedMCPConnectEntry",
    "WatchFileSnapshot",
]
