"""Diagnostics helpers for model alias onboarding and repair flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from fast_agent.cards.manager import load_card_pack_manifest
from fast_agent.core.agent_card_loader import load_agent_cards
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.model_resolution import parse_model_alias_token
from fast_agent.llm.model_alias_config import ModelAliasConfigService
from fast_agent.paths import resolve_environment_paths

if TYPE_CHECKING:
    from pathlib import Path

type ModelAliasSetupPriority = Literal["required", "repair", "recommended"]
type ModelAliasSetupStatus = Literal["missing", "invalid"]


@dataclass(frozen=True)
class ModelAliasSetupItem:
    """Single alias issue that setup can guide the user through."""

    token: str
    priority: ModelAliasSetupPriority
    status: ModelAliasSetupStatus
    current_value: str | None
    summary: str
    references: tuple[str, ...]


@dataclass(frozen=True)
class ModelAliasSetupDiagnostics:
    """Collected alias setup/repair context for onboarding flows."""

    valid_aliases: dict[str, dict[str, str]]
    items: tuple[ModelAliasSetupItem, ...]


@dataclass
class _CollectedItem:
    token: str
    priority: ModelAliasSetupPriority
    status: ModelAliasSetupStatus
    current_value: str | None
    summary: str
    references: set[str]


def collect_model_alias_setup_diagnostics(
    *,
    cwd: Path,
    env_dir: str | Path | None,
) -> ModelAliasSetupDiagnostics:
    """Collect missing/invalid model aliases referenced by config, cards, and packs."""
    service = ModelAliasConfigService(cwd=cwd, env_dir=env_dir)
    model_settings = service.load_effective_model_settings()
    valid_aliases = service.list_aliases_tolerant()

    collected: dict[str, _CollectedItem] = {}
    _collect_invalid_alias_entries(
        collected,
        model_settings.get("model_aliases"),
    )

    default_model = model_settings.get("default_model")
    if isinstance(default_model, str):
        _collect_reference(
            collected,
            token=default_model,
            priority="required",
            reference="default_model",
            valid_aliases=valid_aliases,
        )

    for token, priority, reference in _collect_card_pack_alias_references(
        cwd=cwd,
        env_dir=env_dir,
    ):
        _collect_reference(
            collected,
            token=token,
            priority=priority,
            reference=reference,
            valid_aliases=valid_aliases,
        )

    for token, reference in _collect_agent_card_alias_references(
        cwd=cwd,
        env_dir=env_dir,
    ):
        _collect_reference(
            collected,
            token=token,
            priority="required",
            reference=reference,
            valid_aliases=valid_aliases,
        )

    items = sorted(
        (
            ModelAliasSetupItem(
                token=item.token,
                priority=item.priority,
                status=item.status,
                current_value=item.current_value,
                summary=item.summary,
                references=tuple(sorted(item.references)),
            )
            for item in collected.values()
        ),
        key=_sort_key,
    )

    return ModelAliasSetupDiagnostics(
        valid_aliases=valid_aliases,
        items=tuple(items),
    )


def _sort_key(item: ModelAliasSetupItem) -> tuple[int, int, str]:
    priority_rank = {
        "required": 0,
        "repair": 1,
        "recommended": 2,
    }
    status_rank = {
        "invalid": 0,
        "missing": 1,
    }
    return (
        priority_rank[item.priority],
        status_rank[item.status],
        item.token,
    )


def _collect_invalid_alias_entries(
    collected: dict[str, _CollectedItem],
    aliases_payload: object,
) -> None:
    if not isinstance(aliases_payload, dict):
        return

    for namespace, entries in aliases_payload.items():
        if not isinstance(namespace, str) or not isinstance(entries, dict):
            continue
        for key, raw_value in entries.items():
            if not isinstance(key, str):
                continue

            token = f"${namespace}.{key}"
            try:
                canonical_token = _canonicalize_token(token)
            except ModelConfigError:
                continue

            if isinstance(raw_value, str):
                current_value = raw_value.strip()
                if current_value:
                    continue
                summary = "Configured alias value is empty."
            else:
                current_value = None if raw_value is None else str(raw_value)
                summary = "Configured alias value must be a non-empty string."

            existing = collected.get(canonical_token)
            references = set() if existing is None else set(existing.references)
            collected[canonical_token] = _CollectedItem(
                token=canonical_token,
                priority="repair" if existing is None else existing.priority,
                status="invalid",
                current_value=current_value,
                summary=summary,
                references=references,
            )


def _collect_card_pack_alias_references(
    *,
    cwd: Path,
    env_dir: str | Path | None,
) -> list[tuple[str, ModelAliasSetupPriority, str]]:
    env_paths = resolve_environment_paths(cwd=cwd, override=env_dir)
    card_pack_root = env_paths.card_packs
    if not card_pack_root.exists() or not card_pack_root.is_dir():
        return []

    references: list[tuple[str, ModelAliasSetupPriority, str]] = []
    for pack_dir in sorted(card_pack_root.iterdir()):
        if not pack_dir.is_dir():
            continue
        try:
            manifest = load_card_pack_manifest(pack_dir)
        except Exception:
            continue

        for token in manifest.model_aliases_required:
            references.append((token, "required", f"card pack {manifest.name}"))
        for token in manifest.model_aliases_recommended:
            references.append((token, "recommended", f"card pack {manifest.name}"))

    return references


def _collect_agent_card_alias_references(
    *,
    cwd: Path,
    env_dir: str | Path | None,
) -> list[tuple[str, str]]:
    env_paths = resolve_environment_paths(cwd=cwd, override=env_dir)
    candidate_dirs = {
        (cwd / "agent-cards").resolve(),
        env_paths.agent_cards.resolve(),
    }

    references: list[tuple[str, str]] = []
    for card_dir in sorted(candidate_dirs):
        if not card_dir.exists() or not card_dir.is_dir():
            continue
        try:
            cards = load_agent_cards(card_dir)
        except Exception:
            continue

        for card in cards:
            config = card.agent_data.get("config")
            model = getattr(config, "model", None)
            if not isinstance(model, str):
                continue
            try:
                references.append(
                    (
                        _canonicalize_token(model),
                        f"agent card {card.name}",
                    )
                )
            except ModelConfigError:
                continue

    return references


def _collect_reference(
    collected: dict[str, _CollectedItem],
    *,
    token: str,
    priority: ModelAliasSetupPriority,
    reference: str,
    valid_aliases: dict[str, dict[str, str]],
) -> None:
    try:
        canonical_token = _canonicalize_token(token)
    except ModelConfigError:
        return

    missing_tokens = _collect_transitive_missing_aliases(canonical_token, valid_aliases)
    if missing_tokens:
        for missing_token in sorted(missing_tokens):
            detail = reference if missing_token == canonical_token else f"{reference} via {canonical_token}"
            _upsert_item(
                collected,
                token=missing_token,
                priority=priority,
                status="missing",
                current_value=None,
                summary="Referenced alias is not configured.",
                reference=detail,
            )
        return

    existing = collected.get(canonical_token)
    if existing is None:
        return
    existing.references.add(reference)
    if _priority_rank(priority) < _priority_rank(existing.priority):
        existing.priority = priority


def _collect_transitive_missing_aliases(
    token: str,
    aliases: dict[str, dict[str, str]],
    *,
    stack: tuple[str, ...] = (),
) -> set[str]:
    if token in stack:
        return set()

    namespace, key = parse_model_alias_token(token)
    namespace_entries = aliases.get(namespace)
    if namespace_entries is None:
        return {token}

    value = namespace_entries.get(key)
    if value is None:
        return {token}

    stripped_value = value.strip()
    if not stripped_value:
        return {token}
    if not stripped_value.startswith("$"):
        return set()

    try:
        next_token = _canonicalize_token(stripped_value)
    except ModelConfigError:
        return {token}
    return _collect_transitive_missing_aliases(next_token, aliases, stack=(*stack, token))


def _upsert_item(
    collected: dict[str, _CollectedItem],
    *,
    token: str,
    priority: ModelAliasSetupPriority,
    status: ModelAliasSetupStatus,
    current_value: str | None,
    summary: str,
    reference: str,
) -> None:
    existing = collected.get(token)
    if existing is None:
        collected[token] = _CollectedItem(
            token=token,
            priority=priority,
            status=status,
            current_value=current_value,
            summary=summary,
            references={reference},
        )
        return

    existing.references.add(reference)
    if _priority_rank(priority) < _priority_rank(existing.priority):
        existing.priority = priority
    if status == "invalid":
        existing.status = "invalid"
        existing.summary = summary
        existing.current_value = current_value


def _priority_rank(priority: ModelAliasSetupPriority) -> int:
    return {
        "required": 0,
        "repair": 1,
        "recommended": 2,
    }[priority]


def _canonicalize_token(token: str) -> str:
    namespace, key = parse_model_alias_token(token)
    return f"${namespace}.{key}"
