"""Typed marketplace source result models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

SourceT = TypeVar("SourceT")
StatusT = TypeVar("StatusT")
OriginT = TypeVar("OriginT")
SourceUpdateStatus = Literal["up_to_date", "update_available"]


@dataclass(frozen=True)
class ParsedInstalledSourceFields:
    source_origin: Literal["remote", "local"]
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None
    installed_commit: str | None
    installed_path_oid: str | None
    installed_revision: str
    installed_at: str
    content_fingerprint: str


class InstalledSourcePayloadFields(Protocol):
    schema_version: int
    installed_via: str
    source_origin: Literal["remote", "local"]
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None
    installed_commit: str | None
    installed_path_oid: str | None
    installed_revision: str
    installed_at: str
    content_fingerprint: str


@dataclass(frozen=True, slots=True)
class ParsedGitHubUrl:
    repo_url: str
    repo_ref: str | None
    repo_path: str


@dataclass(frozen=True, slots=True)
class SourceRevision(Generic[StatusT]):
    revision: str | None
    status: StatusT | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class SourcePathOid(Generic[StatusT]):
    path_oid: str | None
    status: StatusT | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class SourceCopyResult(Generic[OriginT]):
    origin: OriginT
    commit: str | None
    path_oid: str | None


@dataclass(frozen=True, slots=True)
class InstalledSourceReadResult(Generic[SourceT]):
    source: SourceT | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SourceUpdateDecision:
    status: SourceUpdateStatus
    detail: str


@dataclass(frozen=True, slots=True)
class MarketplaceSourceContext:
    source_url: str | None = None
    repo_url: str | None = None
    repo_ref: str | None = None

    def as_validation_context(self) -> dict[str, str | None]:
        return {
            "source_url": self.source_url,
            "repo_url": self.repo_url,
            "repo_ref": self.repo_ref,
        }


@dataclass(frozen=True, slots=True)
class MarketplaceRepoFields:
    repo_url: str | None = None
    repo_ref: str | None = None
    repo_path: str | None = None
