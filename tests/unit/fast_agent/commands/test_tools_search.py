"""Contract tests for /tools when PTC Tool Search defers function stubs."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from mcp_types import Tool
from rich.text import Text

from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.commands.context import CommandContext, NonInteractiveCommandIOBase
from fast_agent.commands.handlers.tools import (
    ToolsAddQuery,
    ToolsQuery,
    ToolsSelectQuery,
    handle_list_tools,
    parse_tools_query,
)
from fast_agent.commands.renderers.tools_markdown import render_tools_markdown
from fast_agent.commands.tool_summaries import build_tools_listing
from fast_agent.config import Settings
from fast_agent.core.fastagent import AgentInstance
from fast_agent.tools.tool_surface import (
    ProgrammaticSurface,
    SurfaceTool,
    ToolExposure,
    ToolSurface,
)
from tests.unit.fast_agent.commands.test_tools_handler import _Agent, _MutableLlm, _Provider

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.commands.results import CommandOutcome
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol

PTC = ProgrammaticSurface(
    tool_name="exec_sandbox",
    vocabulary=1,
    tool_format="json",
    instructions="Run Python.",
    type_check=False,
    max_duration_seconds=30,
    max_memory_mb=256,
    search_active=True,
)


def _tool(name: str, description: str = "") -> Tool:
    return Tool(name=name, description=description, input_schema={"type": "object"})


def _function(
    tool_name: str,
    python_name: str,
    *,
    source: str,
    tokens: int,
    deferred: bool,
) -> SurfaceTool:
    return SurfaceTool(
        tool=_tool(tool_name, f"{python_name} description"),
        exposure=ToolExposure.PROGRAMMATIC,
        context_tokens=0 if deferred else tokens,
        python_name=python_name,
        signature=f"{python_name}(*, value: str) -> Any",
        stub=f"async def {python_name}(*, value: str) -> Any: ...",
        deferred=deferred,
        stub_tokens=tokens,
        source=source,
    )


def _surface(*, deferred: bool = True) -> ToolSurface:
    return ToolSurface(
        tools=(
            SurfaceTool(tool=_tool("exec_sandbox"), exposure=ToolExposure.MODEL, context_tokens=90),
            SurfaceTool(
                tool=_tool("lookup_order"),
                exposure=ToolExposure.MODEL,
                context_tokens=20,
                direct_reason="declared by tool",
            ),
            _function("read_file", "read_file", source="local", tokens=50, deferred=False),
            _function(
                "github__create_issue",
                "create_issue",
                source="github",
                tokens=120,
                deferred=deferred,
            ),
            _function(
                "github__list_issues", "list_issues", source="github", tokens=80, deferred=deferred
            ),
            _function(
                "orders__cancel_order",
                "cancel_order",
                source="orders",
                tokens=60,
                deferred=deferred,
            ),
        ),
        programmatic=PTC,
    )


class _SurfaceAgent(_Agent):
    def __init__(self, surface: ToolSurface) -> None:
        super().__init__(_MutableLlm())
        self.surface = surface

    async def tool_surface(self) -> ToolSurface:
        return self.surface


class _PickerIO(NonInteractiveCommandIOBase):
    def __init__(self, picked: Sequence[str] | None) -> None:
        self.picked = picked
        self.surfaces: list[ToolSurface] = []

    async def pick_tools(self, surface: ToolSurface) -> Sequence[str] | None:
        self.surfaces.append(surface)
        return self.picked


def _context(surface: ToolSurface, io: NonInteractiveCommandIOBase | None = None) -> CommandContext:
    return CommandContext(
        agent_provider=_Provider(_SurfaceAgent(surface)),
        current_agent_name="main",
        io=io or NonInteractiveCommandIOBase(),
        settings=Settings(),
    )


async def _run(argument: str, *, surface: ToolSurface | None = None) -> CommandOutcome:
    return await handle_list_tools(
        _context(surface or _surface()), agent_name="main", argument=argument
    )


def _rendered(outcome: CommandOutcome) -> str:
    return "\n".join(
        message.text.plain if isinstance(message.text, Text) else str(message.text)
        for message in outcome.messages
    )


def test_upfront_token_totals_exclude_deferred_functions() -> None:
    listing = build_tools_listing(object(), _surface())

    assert [summary.name for summary in listing.upfront] == ["read_file"]
    assert listing.programmatic_tokens == 50
    assert listing.deferred_tokens == 260
    assert [group for group, _ in listing.deferred_groups()] == ["github", "orders"]


@pytest.mark.asyncio
async def test_tools_listing_separates_upfront_from_searchable_functions() -> None:
    rendered = _rendered(await _run(""))

    assert "Callable from exec_sandbox (1 upfront · ~50 tokens inside exec_sandbox)" in rendered
    assert "Runs Python that calls the 1 function listed below. 3 more are found" in rendered
    assert "Searchable from exec_sandbox (3 · ~260 tokens on demand)" in rendered
    github_line = next(line for line in rendered.splitlines() if "create_issue" in line)
    assert "github" in github_line and "list_issues" in github_line
    assert "search_tools(query=..., server=...)" in rendered
    assert "^tool:<name>" in rendered
    assert "/tools select" in rendered


@pytest.mark.asyncio
async def test_tools_all_lists_deferred_signatures_with_on_demand_cost() -> None:
    rendered = _rendered(await _run("--all"))

    issue_line = next(line for line in rendered.splitlines() if "create_issue " in line)
    assert "~120 tokens on demand" in issue_line
    assert "await create_issue(*, value: str) -> Any" in rendered
    assert rendered.index("read_file") < rendered.index("Searchable from")


@pytest.mark.asyncio
async def test_tools_listing_without_deferred_functions_has_no_search_section() -> None:
    rendered = _rendered(await _run("", surface=_surface(deferred=False)))

    assert "Callable from exec_sandbox (4 · ~310 tokens inside exec_sandbox)" in rendered
    assert "Searchable" not in rendered
    assert "upfront" not in rendered
    assert "/tools select" not in rendered


def test_tools_markdown_separates_upfront_from_searchable_functions() -> None:
    surface = _surface()
    listing = build_tools_listing(object(), surface)

    summary = render_tools_markdown(listing, heading="tools")
    full = render_tools_markdown(listing, heading="tools", show_all=True)

    assert "## Callable from exec\\_sandbox (1 upfront · ~50 tokens inside" in summary
    assert "## Searchable from exec\\_sandbox (3 · ~260 tokens on demand)" in summary
    assert "- **github** (2): `create_issue`, `list_issues`" in summary
    issue_line = next(line for line in full.splitlines() if "`create_issue(" in line)
    assert issue_line.endswith("· ~120 tokens on demand")


@pytest.mark.asyncio
async def test_tools_detail_explains_deferred_function() -> None:
    rendered = _rendered(await _run("create_issue"))

    assert (
        "Callable from `exec_sandbox` as `await create_issue(...)`; deferred from the initial "
        "`exec_sandbox` description (discoverable via `search_tools` or attachable with "
        "`^tool:create_issue`, ~120 tokens)."
    ) in rendered
    assert "async def create_issue(*, value: str) -> Any: ..." in rendered


@pytest.mark.asyncio
async def test_tools_detail_for_upfront_function_is_unchanged() -> None:
    rendered = _rendered(await _run("read_file"))

    assert "The stub adds ~50 tokens to the `exec_sandbox` description." in rendered
    assert "deferred" not in rendered


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("add create_issue ^tool:list_issues", ToolsAddQuery(("create_issue", "list_issues"))),
        ("select", ToolsSelectQuery()),
        ("browse", ToolsSelectQuery()),
        ("explore", ToolsSelectQuery()),
        ("--all", ToolsQuery(show_all=True)),
        ("create_issue", ToolsQuery(tool_name="create_issue")),
        ("-- add", ToolsQuery(tool_name="add")),
    ],
)
def test_parse_tools_query_subcommands(argument: str, expected: object) -> None:
    assert parse_tools_query(argument) == expected


@pytest.mark.parametrize("argument", ["add", "add --all", "select extra"])
def test_parse_tools_query_rejects_malformed_subcommands(argument: str) -> None:
    with pytest.raises(ValueError, match="Usage: /tools"):
        parse_tools_query(argument)


@pytest.mark.asyncio
async def test_tools_add_prefills_deduplicated_mentions() -> None:
    outcome = await _run("add create_issue github__create_issue cancel_order")

    assert outcome.buffer_prefill == "^tool:create_issue ^tool:cancel_order"
    assert "2 function definitions (~180 tokens)" in _rendered(outcome)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("argument", "surface", "message"),
    [
        ("add create_issue nope", _surface(), "^tool:nope: no `exec_sandbox` function"),
        ("add lookup_order", _surface(), "lookup_order is sent to the model as a tool"),
        (
            "add create_issue",
            ToolSurface.direct([_tool("create_issue")]),
            "Programmatic tool calling is not enabled",
        ),
    ],
)
async def test_tools_add_rejects_names_that_are_not_ptc_functions(
    argument: str, surface: ToolSurface, message: str
) -> None:
    outcome = await _run(argument, surface=surface)

    assert outcome.buffer_prefill is None
    assert [item.channel for item in outcome.messages] == ["error"]
    assert message in _rendered(outcome)


@pytest.mark.asyncio
async def test_tools_select_without_picker_suggests_tools_add() -> None:
    outcome = await _run("select")

    assert outcome.buffer_prefill is None
    assert "/tools add <function>" in _rendered(outcome)


@pytest.mark.asyncio
async def test_tools_select_prefills_picked_functions() -> None:
    io = _PickerIO(["list_issues", "create_issue"])

    outcome = await handle_list_tools(
        _context(_surface(), io), agent_name="main", argument="browse"
    )

    assert io.surfaces == [_surface()]
    assert outcome.buffer_prefill == "^tool:list_issues ^tool:create_issue"


@pytest.mark.asyncio
async def test_tools_select_cancel_has_no_side_effects() -> None:
    outcome = await handle_list_tools(
        _context(_surface(), _PickerIO(None)), agent_name="main", argument="select"
    )

    assert outcome.buffer_prefill is None
    assert outcome.messages == []


class _AcpApp:
    def __init__(self, agent: _SurfaceAgent) -> None:
        self.agent = agent

    def _agent(self, name: str) -> _SurfaceAgent:
        del name
        return self.agent


@pytest.mark.asyncio
async def test_acp_tools_add_shows_draft_text_and_select_suggests_add() -> None:
    agent = _SurfaceAgent(_surface())
    handler = SlashCommandHandler(
        session_id="s1",
        instance=AgentInstance(
            app=cast("AgentApp", _AcpApp(agent)),
            agents={"main": cast("AgentProtocol", agent)},
            registry_version=0,
        ),
        primary_agent_name="main",
    )

    added = await handler.execute_command("tools", "add create_issue")
    selected = await handler.execute_command("tools", "select")

    assert "Command produced draft text:\n\n```text\n^tool:create_issue\n```" in added
    assert "/tools add <function>" in selected
