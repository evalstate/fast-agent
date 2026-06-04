"""Shared marketplace update status predicates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

from fast_agent.marketplace.source_models import SourceUpdateDecision

if TYPE_CHECKING:
    from collections.abc import Mapping, Set

CommonMarketplaceUpdateStatus: TypeAlias = Literal[
    "up_to_date",
    "update_available",
    "updated",
    "unmanaged",
    "invalid_metadata",
    "unknown_revision",
    "source_unreachable",
    "source_ref_missing",
    "source_path_missing",
    "skipped_dirty",
    "integrity_error",
]

MarketplaceUpdateStatus: TypeAlias = CommonMarketplaceUpdateStatus | Literal[
    "invalid_local_skill",
    "invalid_local_plugin",
    "invalid_local_pack",
    "ownership_conflict",
]

ALL_MARKETPLACE_UPDATE_STATUSES: tuple[MarketplaceUpdateStatus, ...] = (
    "up_to_date",
    "update_available",
    "updated",
    "unmanaged",
    "invalid_metadata",
    "invalid_local_skill",
    "invalid_local_plugin",
    "invalid_local_pack",
    "unknown_revision",
    "source_unreachable",
    "source_ref_missing",
    "source_path_missing",
    "skipped_dirty",
    "integrity_error",
    "ownership_conflict",
)

APPLICABLE_UPDATE_STATUSES: frozenset[MarketplaceUpdateStatus] = frozenset(
    {"update_available"}
)
APPLIED_UPDATE_STATUSES: frozenset[MarketplaceUpdateStatus] = frozenset({"updated"})
COMMON_UPDATE_STATUS_LABELS: dict[MarketplaceUpdateStatus, str] = {
    "up_to_date": "already up to date",
    "update_available": "update available",
    "updated": "updated",
    "unmanaged": "unmanaged",
    "invalid_metadata": "invalid metadata",
    "invalid_local_skill": "invalid local skill",
    "invalid_local_plugin": "invalid local plugin",
    "invalid_local_pack": "invalid local pack",
    "ownership_conflict": "ownership conflict",
    "unknown_revision": "unknown revision",
    "source_unreachable": "source unreachable",
    "source_ref_missing": "source ref missing",
    "source_path_missing": "source path missing",
    "skipped_dirty": "skipped (local modifications)",
    "integrity_error": "integrity error",
}
COMMON_UPDATE_DETAIL_STATUSES: frozenset[MarketplaceUpdateStatus] = frozenset(
    {
        "invalid_metadata",
        "invalid_local_skill",
        "invalid_local_plugin",
        "invalid_local_pack",
        "ownership_conflict",
        "update_available",
        "unknown_revision",
        "source_unreachable",
        "source_ref_missing",
        "source_path_missing",
        "skipped_dirty",
        "integrity_error",
    }
)
UNSTYLED_UPDATE_STATUSES: frozenset[MarketplaceUpdateStatus] = frozenset({"unmanaged"})
COMMON_UPDATE_STATUS_STYLES: dict[MarketplaceUpdateStatus, str] = {
    "up_to_date": "green",
    "updated": "green",
    "update_available": "bold bright_yellow",
    "invalid_metadata": "yellow",
    "invalid_local_skill": "yellow",
    "invalid_local_plugin": "yellow",
    "invalid_local_pack": "yellow",
    "ownership_conflict": "yellow",
    "unknown_revision": "yellow",
    "source_unreachable": "yellow",
    "source_ref_missing": "yellow",
    "source_path_missing": "yellow",
    "skipped_dirty": "yellow",
    "integrity_error": "yellow",
}


def clean_update_status_detail(detail: str | None) -> str | None:
    if detail is None:
        return None
    cleaned = " ".join(detail.split())
    return cleaned or None


def is_update_applicable(status: MarketplaceUpdateStatus) -> bool:
    """Return whether an update status represents work that should be applied."""
    return status in APPLICABLE_UPDATE_STATUSES


def is_update_applied(status: MarketplaceUpdateStatus) -> bool:
    """Return whether an update status means local content changed."""
    return status in APPLIED_UPDATE_STATUSES


def format_update_status_text(
    status: MarketplaceUpdateStatus,
    *,
    detail: str | None = None,
    labels: "Mapping[MarketplaceUpdateStatus, str] | None" = None,
    detail_statuses: "Set[MarketplaceUpdateStatus] | None" = None,
) -> str:
    """Return display text for a marketplace-style update status."""
    merged_labels = dict(COMMON_UPDATE_STATUS_LABELS)
    if labels:
        merged_labels.update(labels)
    status_text = merged_labels[status]
    merged_detail_statuses = COMMON_UPDATE_DETAIL_STATUSES | frozenset(detail_statuses or ())
    cleaned_detail = clean_update_status_detail(detail)
    if status in merged_detail_statuses and cleaned_detail:
        return f"{status_text}: {cleaned_detail}"
    return status_text


def decide_source_update_status(
    *,
    available_path_oid: str | None,
    current_path_oid: str | None,
    available_revision: str,
    current_revision: str,
    content_changed_detail: str,
) -> SourceUpdateDecision:
    if available_path_oid and current_path_oid:
        if available_path_oid != current_path_oid:
            return SourceUpdateDecision(
                status="update_available",
                detail=content_changed_detail,
            )
    elif available_revision != current_revision:
        return SourceUpdateDecision(
            status="update_available",
            detail="new revision available",
        )

    return SourceUpdateDecision(
        status="up_to_date",
        detail="already up to date",
    )
