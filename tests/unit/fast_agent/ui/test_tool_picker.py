"""Tests for the /tools select picker: selection logic and a keyboard smoke test."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from mcp_types import Tool
from prompt_toolkit.application import create_app_session
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from fast_agent.tools.tool_surface import (
    ProgrammaticSurface,
    SurfaceTool,
    ToolExposure,
    ToolSurface,
)
from fast_agent.ui.tool_picker import (
    ToolPickerState,
    _ToolPicker,
    run_tool_picker_async,
    tool_picker_items,
)


def _function(name: str, *, source: str, tokens: int, deferred: bool = True) -> SurfaceTool:
    return SurfaceTool(
        tool=Tool(name=f"{source}__{name}", description=f"{name} docs", input_schema={}),
        exposure=ToolExposure.PROGRAMMATIC,
        context_tokens=0 if deferred else tokens,
        python_name=name,
        stub=f"async def {name}() -> Any: ...",
        deferred=deferred,
        stub_tokens=tokens,
        source=source,
    )


SURFACE = ToolSurface(
    tools=(
        SurfaceTool(
            tool=Tool(name="exec_sandbox", input_schema={}),
            exposure=ToolExposure.MODEL,
            context_tokens=50,
        ),
        _function("read_file", source="local", tokens=40, deferred=False),
        _function("list_orders", source="orders", tokens=150),
        _function("create_issue", source="github", tokens=200),
        _function("list_issues", source="github", tokens=210),
    ),
    programmatic=ProgrammaticSurface(
        tool_name="exec_sandbox",
        vocabulary=1,
        tool_format="json",
        instructions="",
        type_check=False,
        max_duration_seconds=30,
        max_memory_mb=256,
        search_active=True,
    ),
)


def _state() -> ToolPickerState:
    return ToolPickerState(tool_picker_items(SURFACE))


def _names(state: ToolPickerState) -> list[str]:
    return [item.name for item in state.visible]


def test_items_list_deferred_functions_first_grouped_by_source() -> None:
    state = _state()

    assert _names(state) == ["create_issue", "list_issues", "list_orders", "read_file"]
    assert state.sources == (None, "github", "orders", "local")


def test_rows_show_short_names_without_changing_the_selected_function() -> None:
    name = "my_github__issues"
    function = replace(
        _function("issues", source="my-github", tokens=210),
        python_name=name,
        stub=f"async def {name}() -> Any: ...",
    )
    surface = replace(SURFACE, tools=(function,))
    picker = _ToolPicker(tool_picker_items(surface), tool_name="exec_sandbox")

    all_rows = fragment_list_to_text(picker._render_rows())
    assert "issues" in all_rows
    assert name not in all_rows
    assert "my-github" in all_rows

    picker.state.cycle_source(1)
    picker.state.type_text(name)
    server_rows = fragment_list_to_text(picker._render_rows())
    assert "issues" in server_rows
    assert name not in server_rows
    assert "my-github" not in server_rows
    assert f"^tool:{name}" in fragment_list_to_text(picker._render_preview())
    assert picker.state.chosen_names() == (name,)


def test_picker_marks_known_functions_without_disabling_selection() -> None:
    surface = replace(
        SURFACE,
        tools=(
            replace(_function("seen_before", source="github", tokens=30), seen=True),
            _function("still_new", source="github", tokens=40),
            _function("advertised", source="github", tokens=50, deferred=False),
        ),
    )
    picker = _ToolPicker(tool_picker_items(surface), tool_name="exec_sandbox")
    picker.state.cycle_source(1)

    seen, unseen, upfront = fragment_list_to_text(picker._render_rows()).splitlines()
    assert "[-] seen_before" in seen
    assert "[ ] still_new" in unseen
    assert "[-] advertised" in upfront
    assert "[-] already seen" in fragment_list_to_text(picker._render_footer())

    picker.state.toggle()
    assert "[x] seen_before" in fragment_list_to_text(picker._render_rows())
    picker.state.move(2)
    assert "already in the exec_sandbox description" in fragment_list_to_text(
        picker._render_preview()
    )
    picker.state.toggle()
    assert "[x] advertised" in fragment_list_to_text(picker._render_rows())
    assert picker.state.chosen_names() == ("seen_before", "advertised")

    picker.state.toggle()
    assert "[-] advertised" in fragment_list_to_text(picker._render_rows())


def test_typing_filters_and_backspace_edits_the_query() -> None:
    state = _state()

    for char in "LIST":
        state.type_text(char)
    assert _names(state) == ["list_issues", "list_orders"]

    state.type_text("_o")
    assert _names(state) == ["list_orders"]

    state.backspace()
    state.backspace()
    assert _names(state) == ["list_issues", "list_orders"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("issue", ["create_issue", "list_issues"]),
        ("ssue", ["create_issue", "list_issues"]),
        ("list_or", ["list_orders"]),
        ("git/list", ["list_issues"]),
        ("hub_list", []),
        ("issue-cre", ["create_issue"]),
    ],
)
def test_filter_terms_split_on_separators_must_all_start_words(
    query: str, expected: list[str]
) -> None:
    state = _state()
    state.type_text(query)

    assert _names(state) == expected


def test_server_named_all_gets_its_own_filterable_tab() -> None:
    items = (
        *tool_picker_items(SURFACE),
        *tool_picker_items(
            ToolSurface(
                tools=(_function("everything", source="all", tokens=10),),
                programmatic=SURFACE.programmatic,
            )
        ),
    )
    state = ToolPickerState(items)

    assert state.sources == (None, "github", "orders", "local", "all")
    state.cycle_source(-1)
    assert _names(state) == ["everything"]
    state.cycle_source(1)
    assert len(state.visible) == len(items)


def test_cycling_source_groups_wraps_and_combines_with_the_query() -> None:
    state = _state()

    state.cycle_source(1)
    assert state.source == "github"
    assert _names(state) == ["create_issue", "list_issues"]

    state.type_text("list")
    assert _names(state) == ["list_issues"]

    state.cycle_source(-2)
    assert state.source == "local"
    assert _names(state) == []
    assert state.current is None


def test_toggle_selects_in_order_and_footer_sums_tokens() -> None:
    state = _state()
    assert state.footer_text() == "Selected: 0 functions"
    assert state.chosen_names() == ("create_issue",)

    state.move(2)
    state.toggle()
    state.move(-2)
    state.toggle()
    assert state.chosen_names() == ("list_orders", "create_issue")
    assert state.footer_text() == "Selected: 2 functions (~350 tokens)"

    state.toggle()
    assert state.footer_text() == "Selected: 1 function (~150 tokens)"


def test_selection_survives_filtering() -> None:
    state = _state()
    state.toggle()
    state.type_text("orders")
    state.toggle()

    assert state.chosen_names() == ("create_issue", "list_orders")


def test_move_clamps_to_the_visible_rows() -> None:
    state = _state()
    state.move(-5)
    assert state.cursor == 0
    state.move(50)
    assert state.cursor == 3


async def _run_picker(keys: str) -> tuple[str, ...] | None:
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        task = asyncio.create_task(run_tool_picker_async(SURFACE))
        await asyncio.sleep(0.05)
        pipe.send_text(keys)
        return await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_picker_keyboard_selection_returns_python_names() -> None:
    # Type "issue", toggle both matches, confirm.
    assert await _run_picker("issue \x1b[B \r") == ("create_issue", "list_issues")


@pytest.mark.asyncio
async def test_picker_tab_cycles_source_before_enter() -> None:
    assert await _run_picker("\t\t\r") == ("list_orders",)


@pytest.mark.asyncio
async def test_picker_cancel_returns_none() -> None:
    assert await _run_picker(" \x03") is None
