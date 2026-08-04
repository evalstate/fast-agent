"""Parent-scoped storage for temporary model-readable artifacts."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from fast_agent.tools.execution_environment import (
        EnvironmentTemporaryArtifacts,
        TemporaryArtifact,
    )

TRANSIENT_ARTIFACT_MAX_BYTES = 2 * 1024 * 1024
TRANSIENT_ARTIFACT_QUOTA_MARKER = "\n[fast-agent temporary-file quota reached]\n"
_ARTIFACT_NAME_PART = re.compile(r"^[A-Za-z0-9._-]+$")
logger = get_logger(__name__)


def validate_artifact_name_parts(*, prefix: str, suffix: str) -> None:
    """Reject path syntax in adapter-level filename fragments."""

    if not prefix or not suffix or not _ARTIFACT_NAME_PART.fullmatch(prefix + suffix):
        raise ValueError("Temporary artifact prefix and suffix must be filename-safe.")


def bounded_temporary_text(content: str, *, max_bytes: int) -> tuple[bytes, bool]:
    """Encode a bounded UTF-8 prefix without splitting a code point."""

    payload = content.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    if len(payload) <= max_bytes:
        return payload, True

    marker = TRANSIENT_ARTIFACT_QUOTA_MARKER.encode("utf-8")
    retained_limit = max(0, max_bytes - len(marker))
    retained = payload[:retained_limit]
    while retained:
        try:
            retained.decode("utf-8")
            break
        except UnicodeDecodeError as exc:
            retained = retained[: exc.start]
    return retained + marker[: max_bytes - len(retained)], False


def format_retained_artifact_notice(
    *,
    path: str,
    retained_bytes: int,
    complete: bool,
    description: str,
) -> str:
    """Format the shared model-facing locator contract."""

    availability = (
        f"The complete {description} is available during this session at {path}."
        if complete
        else (
            f"The first {retained_bytes} bytes of the {description} are available during "
            f"this session at {path}; the temporary-file quota was reached."
        )
    )
    return (
        f"{availability} Use read_text_file for selected line ranges or run a targeted "
        "search against that file; avoid reading the entire file unless necessary."
    )


@dataclass(frozen=True, slots=True)
class TransientArtifactResult:
    """Artifact details returned to a producer."""

    artifact: TemporaryArtifact
    notice: str


class TransientArtifactStore:
    """Track temporary files owned by one parent agent."""

    def __init__(self, environment: EnvironmentTemporaryArtifacts) -> None:
        self._environment = environment
        self._artifacts: list[TemporaryArtifact] = []
        self._lock = asyncio.Lock()
        self._closed = False

    async def write_text(
        self,
        *,
        producer: str,
        suffix: str,
        content: str,
        description: str,
        max_bytes: int = TRANSIENT_ARTIFACT_MAX_BYTES,
    ) -> TransientArtifactResult:
        async with self._lock:
            if self._closed:
                raise RuntimeError("Transient artifact store is closed.")
            artifact = await self._environment.write_temporary_text(
                prefix=f"fast-agent-{producer}-",
                suffix=suffix,
                content=content,
                max_bytes=max_bytes,
            )
            self._artifacts.append(artifact)
        return TransientArtifactResult(
            artifact=artifact,
            notice=format_retained_artifact_notice(
                path=artifact.path,
                retained_bytes=artifact.retained_bytes,
                complete=artifact.complete,
                description=description,
            ),
        )

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            artifacts = list(reversed(self._artifacts))
            self._artifacts.clear()
        for artifact in artifacts:
            try:
                await self._environment.remove_temporary_artifact(artifact)
            except Exception:
                logger.debug("Failed to remove temporary artifact")
                continue

    async def remove(self, artifact: TemporaryArtifact) -> None:
        """Remove one retained artifact without closing the parent store."""

        async with self._lock:
            if artifact not in self._artifacts:
                return
            await self._environment.remove_temporary_artifact(artifact)
            self._artifacts.remove(artifact)


__all__ = [
    "TRANSIENT_ARTIFACT_MAX_BYTES",
    "TRANSIENT_ARTIFACT_QUOTA_MARKER",
    "TransientArtifactResult",
    "TransientArtifactStore",
    "bounded_temporary_text",
    "format_retained_artifact_notice",
    "validate_artifact_name_parts",
]
