from __future__ import annotations

from pathlib import Path

from fast_agent.utils.path_display import (
    fit_path_for_display,
    format_home_relative_path,
    format_parent_current_path,
    format_relative_path,
    format_working_directory,
    left_truncate_with_ellipsis,
)


def test_format_relative_path_prefers_cwd_relative_path() -> None:
    cwd = Path("/tmp/project")

    assert format_relative_path(cwd / "src/app.py", cwd=cwd) == "src/app.py"


def test_format_relative_path_uses_absolute_path_outside_cwd() -> None:
    assert format_relative_path(Path("/opt/app.py"), cwd=Path("/tmp/project")) == "/opt/app.py"


def test_format_home_relative_path_prefers_tilde_path_under_home() -> None:
    home = Path("/tmp/home")

    assert format_home_relative_path(home / "fast-agent/config.yaml", home=home) == (
        "~/fast-agent/config.yaml"
    )


def test_format_home_relative_path_displays_home_directory_as_tilde() -> None:
    home = Path("/tmp/home")

    assert format_home_relative_path(home, home=home) == "~"


def test_format_home_relative_path_keeps_expanded_path_outside_home() -> None:
    assert format_home_relative_path(Path("/opt/config.yaml"), home=Path("/tmp/home")) == (
        "/opt/config.yaml"
    )


def test_format_working_directory_names_current_directory() -> None:
    assert format_working_directory(Path("/tmp/project"), cwd=Path("/tmp/project")) == "tmp/project"


def test_format_working_directory_keeps_relative_child_directory() -> None:
    cwd = Path("/tmp/project")

    assert format_working_directory(cwd / "subdir", cwd=cwd) == "subdir"


def test_left_truncate_with_ellipsis_truncates_from_left() -> None:
    assert left_truncate_with_ellipsis("superlongcurrent", 8) == "…current"
    assert left_truncate_with_ellipsis("superlongcurrent", 1) == "…"
    assert left_truncate_with_ellipsis("current", 10) == "current"


def test_left_truncate_with_ellipsis_respects_custom_ellipsis_width() -> None:
    assert left_truncate_with_ellipsis("superlongcurrent", 8, ellipsis="...") == "...rrent"
    assert left_truncate_with_ellipsis("superlongcurrent", 2, ellipsis="...") == ".."


def test_format_parent_current_path_prefers_parent_and_current() -> None:
    assert format_parent_current_path("parent/current") == "parent/current"
    assert format_parent_current_path("current") == "current"


def test_format_parent_current_path_preserves_empty_path() -> None:
    assert format_parent_current_path("") == ""


def test_format_parent_current_path_strips_outer_whitespace() -> None:
    assert format_parent_current_path("  parent/current  ") == "parent/current"
    assert format_parent_current_path("   ") == ""


def test_format_parent_current_path_handles_windows_separators() -> None:
    assert format_parent_current_path(r"C:\Users\me\file.txt") == "me/file.txt"


def test_fit_path_for_display_prefers_compact_then_current_name() -> None:
    assert fit_path_for_display("parent/current", 20) == "parent/current"
    assert fit_path_for_display("very-long-parent/current", 7) == "current"
    assert fit_path_for_display("parent/superlongcurrent", 8) == "…current"


def test_fit_path_for_display_preserves_empty_path() -> None:
    assert fit_path_for_display("", 10) == ""


def test_fit_path_for_display_strips_outer_whitespace() -> None:
    assert fit_path_for_display("  parent/current  ", 20) == "parent/current"
    assert fit_path_for_display("   ", 10) == ""


def test_fit_path_for_display_uses_windows_basename_for_fallback() -> None:
    assert fit_path_for_display(r"C:\very-long-parent\current", 7) == "current"
    assert fit_path_for_display(r"C:\parent\superlongcurrent", 8) == "…current"
