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
from fast_agent.paths import resolve_home_dir

logger = get_logger(__name__)


class ToolCacheInfo(BaseModel):
    source: Literal["live", "disk"]
    fetched_at: float
    expires_at: float
    tool_count: int


class ToolSnapshot(BaseModel):
    version: Literal[1] = 1
    key: str
    fetched_at: float
    tools: list[Tool] = Field(default_factory=list)


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
        directory = (
            Path(config.tool_cache.directory).expanduser()
            if config.tool_cache.directory is not None
            else resolve_home_dir(settings if settings is not None else Settings())
            / "cache"
            / "mcp-tools"
        )
        self.path = directory / f"{self.key}.json"

    @property
    def enabled(self) -> bool:
        return (
            self.config.tool_cache.enabled
            and (
                self.config.transport == "stdio"
                or bool((self.config.tool_cache.auth_identity or "").strip())
            )
            and not (self.config.auth and self.config.auth.forward)
        )

    def load(self) -> ToolSnapshot | None:
        if not self.enabled:
            return None
        try:
            snapshot = ToolSnapshot.model_validate_json(self.path.read_bytes())
            age = time.time() - snapshot.fetched_at
            if snapshot.key == self.key and 0 <= age < self.config.tool_cache.ttl_seconds:
                return snapshot
        except (OSError, ValidationError):
            pass
        return None

    def clear(self) -> None:
        if self.enabled:
            with suppress(OSError):
                self.path.unlink(missing_ok=True)

    def save(self, snapshot: ToolSnapshot) -> None:
        if not self.enabled:
            return
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
        except OSError as exc:
            logger.warning(f"Unable to persist MCP tool catalog: {exc}")
        finally:
            if temporary is not None:
                with suppress(OSError):
                    Path(temporary).unlink(missing_ok=True)
