"""Small privacy-sanitizer interfaces shared by trace exporters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import timedelta


@dataclass(frozen=True, slots=True)
class RedactionSpan:
    """A detected private text span in the original input."""

    label: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SanitizedText:
    """Sanitized text plus spans redacted from the original input."""

    text: str
    spans: tuple[RedactionSpan, ...] = ()


@dataclass(frozen=True, slots=True)
class PrivacyFilterModelInfo:
    """Model/backend provenance for a privacy-filter run."""

    backend: str
    repo_id: str | None = None
    revision: str | None = None
    variant: str | None = None


@dataclass(frozen=True, slots=True)
class RedactionSummary:
    """Aggregate redaction counts for an exported trace."""

    total: int
    by_label: dict[str, int]
    model: PrivacyFilterModelInfo | None = None
    elapsed: timedelta | None = None


class TraceSanitizer(Protocol):
    """Sanitize text-bearing trace content."""

    @property
    def model_info(self) -> PrivacyFilterModelInfo | None: ...

    def sanitize_text(self, text: str) -> SanitizedText: ...


@dataclass(slots=True)
class RedactionAccumulator:
    """Mutable redaction-count accumulator for one export."""

    model: PrivacyFilterModelInfo | None = None
    total: int = 0
    by_label: dict[str, int] = field(default_factory=dict)
    elapsed: timedelta | None = None

    def add(self, spans: Iterable[RedactionSpan]) -> None:
        for span in spans:
            self.total += 1
            self.by_label[span.label] = self.by_label.get(span.label, 0) + 1

    def summary(self) -> RedactionSummary:
        return RedactionSummary(
            total=self.total,
            by_label=dict(sorted(self.by_label.items())),
            model=self.model,
            elapsed=self.elapsed,
        )
