"""Private, versioned raw MCP tool snapshots (never connection credentials)."""

import hashlib
import json
import os
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Literal

from mcp_types import Tool
from pydantic import BaseModel, Field, SecretBytes, SecretStr, ValidationError

from fast_agent.config import MCPServerSettings, Settings
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.definition_digests import Digests, covers
from fast_agent.paths import resolve_home_dir

logger = get_logger(__name__)


class ToolCacheInfo(BaseModel):
    source: Literal["live", "disk"]
    fetched_at: float
    tool_count: int
    digest: bool = False
    """Server definition digests are checked on every call; no TTL applies."""
    expires_at: float | None = None
    """Local reuse deadline for catalogs without digests."""
    persisted: bool = False
    """A snapshot of this catalog is on disk for later sessions."""


class ToolSnapshot(BaseModel):
    version: Literal[3] = 3
    key: str
    fetched_at: float
    tools: list[Tool] = Field(default_factory=list)
    instructions: str | None = None
    digests: Digests = Field(default_factory=dict)
    """Server definition digests keyed by the method that produced them."""

    def digest_mode(self, *, include_instructions: bool) -> bool:
        """Server digests cover everything cached: ignore the TTL, detect change on call."""
        return covers(self.digests, instructions=include_instructions)


def _identity_value(value: object) -> object:
    # Python-mode dumps preserve secrets; JSON-mode dumps mask them and collide.
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, SecretBytes):
        return value.get_secret_value().hex()
    if isinstance(value, dict):
        return {key: _identity_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    return value


class ToolCatalogCache:
    def __init__(self, config: MCPServerSettings, *, settings: Settings | None = None) -> None:
        self.config = config
        # --no-home disables the default cache location; an explicit directory still applies.
        self._no_home = (
            config.tool_cache.directory is None
            and settings is not None
            and settings._fast_agent_no_home
        )
        # Hash explicit resolved configuration, never volatile inherited environment.
        payload = json.dumps(
            [
                _identity_value(
                    config.model_dump(
                        mode="python", exclude={"tool_cache", "connection_policy", "load_on_start"}
                    )
                ),
                config.tool_cache.auth_identity,
                str(Path(config.cwd or ".").resolve()) if config.transport == "stdio" else None,
            ],
            sort_keys=True,
            default=str,
        ).encode()
        self.key = hashlib.sha256(payload).hexdigest()
        self._settings = settings

    @property
    def path(self) -> Path:
        # Resolved lazily: the default home location is unavailable under --no-home.
        directory = (
            Path(self.config.tool_cache.directory).expanduser()
            if self.config.tool_cache.directory is not None
            else resolve_home_dir(self._settings if self._settings is not None else Settings())
            / "cache"
            / "mcp-tools"
        )
        return directory / f"{self.key}.json"

    @property
    def enabled(self) -> bool:
        return (
            self.config.tool_cache.enabled
            and not self._no_home
            and not (self.config.auth and self.config.auth.forward)
        )

    @property
    def partitioned(self) -> bool:
        """OAuth/session credentials are not necessarily represented in server settings,
        so TTL snapshots of network servers need an explicit account partition."""
        return self.config.transport == "stdio" or bool(
            (self.config.tool_cache.auth_identity or "").strip()
        )

    def reusable(self, snapshot: ToolSnapshot) -> bool:
        """Digest snapshots are checked by the server on every call, so they need no
        partition; another account's snapshot is rejected before any tool executes."""
        return self.digest_mode(snapshot) or self.partitioned

    def load(self) -> ToolSnapshot | None:
        if not self.enabled:
            return None
        try:
            snapshot = ToolSnapshot.model_validate_json(self.path.read_bytes())
            if snapshot.key == self.key and (
                self.digest_mode(snapshot) or (self.partitioned and self._fresh(snapshot))
            ):
                return snapshot
        except (OSError, ValidationError):
            pass
        return None

    def digest_mode(self, snapshot: ToolSnapshot) -> bool:
        return snapshot.digest_mode(include_instructions=self.config.include_instructions)

    def _fresh(self, snapshot: ToolSnapshot) -> bool:
        age = time.time() - snapshot.fetched_at
        return 0 <= age < self.config.tool_cache.ttl_seconds

    def clear(self) -> None:
        if self.enabled:
            with suppress(OSError):
                self.path.unlink(missing_ok=True)

    def save(self, snapshot: ToolSnapshot) -> bool:
        """Write the snapshot if policy allows; returns whether it is now on disk."""
        if not self.enabled or not self.reusable(snapshot):
            return False
        # Persistence is optional: filesystem failures must not break discovery.
        temporary: str | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, temporary = tempfile.mkstemp(dir=self.path.parent)
            with os.fdopen(fd, "w") as stream:
                stream.write(snapshot.model_dump_json())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            return True
        except OSError as exc:
            logger.warning(f"Unable to persist MCP tool catalog: {exc}")
            return False
        finally:
            if temporary is not None:
                with suppress(OSError):
                    Path(temporary).unlink(missing_ok=True)
