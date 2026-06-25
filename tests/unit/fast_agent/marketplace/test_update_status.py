from __future__ import annotations

import pytest

from fast_agent.marketplace.update_status import (
    COMMON_UPDATE_DETAIL_STATUSES,
    COMMON_UPDATE_STATUS_LABELS,
    COMMON_UPDATE_STATUS_STYLES,
    CommonMarketplaceUpdateStatus,
    MarketplaceUpdateStatus,
    decide_source_update_status,
    format_update_status_text,
    is_update_applicable,
    is_update_applied,
)

KNOWN_MARKETPLACE_UPDATE_STATUSES: set[MarketplaceUpdateStatus] = {
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
}
UNSTYLED_UPDATE_STATUSES: set[MarketplaceUpdateStatus] = {"unmanaged"}


def test_shared_labels_cover_known_marketplace_update_statuses() -> None:
    assert set(COMMON_UPDATE_STATUS_LABELS) == KNOWN_MARKETPLACE_UPDATE_STATUSES


def test_detail_statuses_are_known_marketplace_update_statuses() -> None:
    assert COMMON_UPDATE_DETAIL_STATUSES < KNOWN_MARKETPLACE_UPDATE_STATUSES


def test_style_statuses_cover_known_marketplace_update_statuses() -> None:
    styled_statuses = set(COMMON_UPDATE_STATUS_STYLES)
    unstyled_statuses = set(UNSTYLED_UPDATE_STATUSES)

    assert styled_statuses | unstyled_statuses == KNOWN_MARKETPLACE_UPDATE_STATUSES
    assert styled_statuses.isdisjoint(unstyled_statuses)


def test_only_update_available_is_applicable() -> None:
    assert is_update_applicable("update_available") is True
    non_applicable_statuses = [
        status for status in KNOWN_MARKETPLACE_UPDATE_STATUSES if status != "update_available"
    ]
    for status in non_applicable_statuses:
        assert is_update_applicable(status) is False


def test_only_updated_is_applied() -> None:
    assert is_update_applied("updated") is True
    non_applied_statuses = [status for status in KNOWN_MARKETPLACE_UPDATE_STATUSES if status != "updated"]
    for status in non_applied_statuses:
        assert is_update_applied(status) is False


def test_format_update_status_text_uses_shared_labels_and_detail_rules() -> None:
    assert format_update_status_text("up_to_date") == "already up to date"
    assert (
        format_update_status_text("source_unreachable", detail="git failed")
        == "source unreachable: git failed"
    )
    assert (
        format_update_status_text("update_available", detail="content changed")
        == "update available: content changed"
    )
    assert format_update_status_text("updated", detail="ignored") == "updated"
    assert (
        format_update_status_text(
            "invalid_local_skill",
            detail="missing description",
        )
        == "invalid local skill: missing description"
    )
    assert (
        format_update_status_text("ownership_conflict", detail="owned by another pack")
        == "ownership conflict: owned by another pack"
    )


def test_format_update_status_text_normalizes_detail_whitespace() -> None:
    assert (
        format_update_status_text("source_unreachable", detail="  git\n\nfailed  ")
        == "source unreachable: git failed"
    )
    assert format_update_status_text("source_unreachable", detail="   ") == "source unreachable"


@pytest.mark.parametrize(
    (
        "available_path_oid",
        "current_path_oid",
        "available_revision",
        "current_revision",
        "expected_status",
        "expected_detail",
    ),
    [
        ("new-tree", "old-tree", "new-rev", "old-rev", "update_available", "content changed"),
        ("same-tree", "same-tree", "new-rev", "old-rev", "up_to_date", "already up to date"),
        (None, "old-tree", "new-rev", "old-rev", "update_available", "new revision available"),
        (None, "old-tree", "same-rev", "same-rev", "up_to_date", "already up to date"),
    ],
)
def test_decide_source_update_status(
    available_path_oid: str | None,
    current_path_oid: str | None,
    available_revision: str,
    current_revision: str,
    expected_status: CommonMarketplaceUpdateStatus,
    expected_detail: str,
) -> None:
    decision = decide_source_update_status(
        available_path_oid=available_path_oid,
        current_path_oid=current_path_oid,
        available_revision=available_revision,
        current_revision=current_revision,
        content_changed_detail="content changed",
    )

    assert decision.status == expected_status
    assert decision.detail == expected_detail
