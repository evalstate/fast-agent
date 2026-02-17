from pathlib import Path

from fast_agent.ui.enhanced_prompt import (
    _can_fit_shell_path_and_version,
    _fit_shell_identity_for_toolbar,
    _fit_shell_path_for_toolbar,
    _format_parent_current_path,
    _left_truncate_with_ellipsis,
)


def test_left_truncate_with_ellipsis_keeps_short_text() -> None:
    assert _left_truncate_with_ellipsis("current", 10) == "current"


def test_left_truncate_with_ellipsis_truncates_from_left() -> None:
    assert _left_truncate_with_ellipsis("superlongcurrent", 8) == "…current"


def test_left_truncate_with_ellipsis_handles_single_char_budget() -> None:
    assert _left_truncate_with_ellipsis("superlongcurrent", 1) == "…"


def test_format_parent_current_path_prefers_parent_and_current() -> None:
    assert _format_parent_current_path(Path("parent/current")) == "parent/current"


def test_fit_shell_path_for_toolbar_prefers_parent_current_when_it_fits() -> None:
    path = Path("parent/current")

    assert _fit_shell_path_for_toolbar(path, len("parent/current")) == "parent/current"


def test_fit_shell_path_for_toolbar_falls_back_to_current_folder() -> None:
    path = Path("verylongparent/current")

    assert _fit_shell_path_for_toolbar(path, len("current")) == "current"


def test_fit_shell_path_for_toolbar_left_truncates_when_current_folder_too_long() -> None:
    path = Path("verylongparent/superlongcurrent")

    assert _fit_shell_path_for_toolbar(path, 8) == "…current"


def test_fit_shell_path_for_toolbar_returns_empty_when_no_space() -> None:
    assert _fit_shell_path_for_toolbar(Path("parent/current"), 0) == ""


def test_fit_shell_identity_for_toolbar_includes_version_when_room_exists() -> None:
    path = Path("parent/current")
    version = "fast-agent 1.2.3"
    expected = "parent/current | fast-agent 1.2.3"

    assert _fit_shell_identity_for_toolbar(path, version, len(expected)) == expected


def test_fit_shell_identity_for_toolbar_uses_current_with_version_when_needed() -> None:
    path = Path("verylongparent/current")
    version = "fast-agent 1.2.3"
    expected = "current | fast-agent 1.2.3"

    assert _fit_shell_identity_for_toolbar(path, version, len(expected)) == expected


def test_fit_shell_identity_for_toolbar_falls_back_to_path_only_when_tight() -> None:
    path = Path("verylongparent/superlongcurrent")
    version = "fast-agent 1.2.3"

    assert _fit_shell_identity_for_toolbar(path, version, 8) == "…current"


def test_can_fit_shell_path_and_version_when_parent_current_fits() -> None:
    path = Path("parent/current")
    version = "fast-agent 1.2.3"
    width = len("parent/current | fast-agent 1.2.3")

    assert _can_fit_shell_path_and_version(path, version, width)


def test_can_fit_shell_path_and_version_when_only_current_fits() -> None:
    path = Path("verylongparent/current")
    version = "fast-agent 1.2.3"
    width = len("current | fast-agent 1.2.3")

    assert _can_fit_shell_path_and_version(path, version, width)


def test_can_fit_shell_path_and_version_false_when_no_combination_fits() -> None:
    path = Path("verylongparent/superlongcurrent")
    version = "fast-agent 1.2.3"

    assert not _can_fit_shell_path_and_version(path, version, 12)
