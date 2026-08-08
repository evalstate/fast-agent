from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

ProfileT = TypeVar("ProfileT")


@dataclass(frozen=True, slots=True)
class RouterRoute:
    """Canonical model and router-selected backend used for profile lookup."""

    model: str
    backend: str | None


@dataclass(frozen=True, slots=True)
class RouterProfileRule(Generic[ProfileT]):
    """Ordered route rule; ``backends=None`` matches every backend."""

    model: str
    profile: ProfileT
    backends: frozenset[str] | None = None

    def matches(self, route: RouterRoute) -> bool:
        return self.model == route.model and (
            self.backends is None or route.backend in self.backends
        )


@dataclass(frozen=True, slots=True)
class RouterProfileRegistry(Generic[ProfileT]):
    """Resolve the first matching profile for a normalized router route."""

    rules: tuple[RouterProfileRule[ProfileT], ...]

    def __post_init__(self) -> None:
        wildcard_models: set[str] = set()
        covered_backends: dict[str, set[str]] = {}
        for rule in self.rules:
            if rule.model in wildcard_models:
                raise ValueError(
                    f"Router profile rule for '{rule.model}' is shadowed by an earlier wildcard"
                )
            if rule.backends is None:
                wildcard_models.add(rule.model)
                continue

            covered = covered_backends.setdefault(rule.model, set())
            overlap = covered.intersection(rule.backends)
            if overlap:
                duplicate = ", ".join(sorted(overlap))
                raise ValueError(
                    f"Router profile rule for '{rule.model}' duplicates backends: {duplicate}"
                )
            covered.update(rule.backends)

    def resolve(self, route: RouterRoute) -> ProfileT | None:
        return next((rule.profile for rule in self.rules if rule.matches(route)), None)
