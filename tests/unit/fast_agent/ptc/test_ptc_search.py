"""PTC tool search: definition visibility, stable names, snapshots and ``search_tools``."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.config import ProgrammaticToolCallingSettings, Settings, ShellSettings
from fast_agent.constants import FAST_AGENT_PTC_CALLS
from fast_agent.ptc.catalog import PTC_SEARCH_EXTRA_THRESHOLD
from fast_agent.ptc.runtime import ProgrammaticToolRuntime
from fast_agent.ptc.search import SEARCH_FUNCTION_NAME, search_tools, stem, tokenize
from fast_agent.ptc.stubs import UNSTRUCTURED_RETURNS
from fast_agent.tools.elicitation import get_elicitation_tool
from fast_agent.tools.shell_runtime import ShellRuntime
from fast_agent.tools.tool_surface import ToolExposure

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("pydantic_monty")


def _text(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _shell_tools(tmp_path: Path) -> list[Tool]:
    return ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        model_shell_tool_name="shell",
        config=Settings(shell_execution=ShellSettings(tool_profile="minimal_process")),
        working_directory=tmp_path,
    ).tools


def _tool(name: str, description: str = "", **properties: dict[str, Any]) -> Tool:
    return Tool(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": properties},
    )


def _runtime(**settings: Any) -> ProgrammaticToolRuntime:
    return ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(**settings))


GITHUB = [
    _tool("github__list_issues", "List issues in a repository.", repo={"type": "string"}),
    _tool("github__create_issue", "Open a new issue.", title={"type": "string"}),
    _tool("github__get_file_contents", "Read a file from a repository.", path={"type": "string"}),
]
HF = [
    _tool(
        "hf__hub_repo_details",
        "Details for models, datasets or spaces.",
        repo_type={"type": "string", "enum": ["model", "dataset", "space"]},
    ),
    _tool("hf__model_search", "Find models on the Hub by task or author."),
]


async def _ok(dispatch_name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=f"called {dispatch_name}")])


def _description(runtime: ProgrammaticToolRuntime, tools: list[Tool]) -> str:
    return runtime.model_tools(tools)[0].description or ""


# Visibility contract ---------------------------------------------------------------------------


def test_threshold_extras_inline(tmp_path: Path) -> None:
    runtime = _runtime()
    description = _description(runtime, [*_shell_tools(tmp_path), *GITHUB])

    assert len(GITHUB) == PTC_SEARCH_EXTRA_THRESHOLD
    assert not runtime.catalog.search_active
    assert SEARCH_FUNCTION_NAME not in description
    assert all(f"async def {tool.name}(" in description for tool in GITHUB)
    assert runtime.catalog.roster.entries == ()


def test_search_defers_extras_and_keeps_builtins_upfront(tmp_path: Path) -> None:
    runtime = _runtime()
    builtins = [*_shell_tools(tmp_path), get_elicitation_tool()]
    description = _description(runtime, [*builtins, *GITHUB, *HF])
    catalog = runtime.catalog

    assert catalog.search_active
    assert catalog.deferred == {tool.name for tool in [*GITHUB, *HF]}
    assert "async def search_tools(" in description
    assert "searchable on demand" in description
    assert "async def bash(" in description and "async def process(" in description
    assert "async def tool__human_input(" in description
    # No deferred name leaks into the description, so it stays stable as sources change.
    for tool in [*GITHUB, *HF]:
        assert tool.name not in description
    # Deferred functions stay registered for the type checker.
    assert "async def github__list_issues(" in catalog.type_check_stubs
    assert "async def search_tools(" in catalog.type_check_stubs


def test_surface_reports_deferred_functions(tmp_path: Path) -> None:
    runtime = _runtime()
    tools = runtime.model_tools([*_shell_tools(tmp_path), *GITHUB, *HF])
    surface = runtime.surface(tools)
    assert surface.programmatic is not None and surface.programmatic.search_active

    deferred = surface.programmatic_functions(deferred=True)
    upfront = surface.programmatic_functions(deferred=False)
    assert {tool.python_name for tool in deferred} == {tool.name for tool in [*GITHUB, *HF]}
    assert all(tool.context_tokens == 0 and tool.stub_tokens > 0 for tool in deferred)
    assert all(tool.context_tokens == tool.stub_tokens > 0 for tool in upfront)
    assert {tool.source for tool in deferred} == {"github", "hf"}
    assert {tool.source for tool in upfront} == {"builtin"}
    assert len(surface.exposed(ToolExposure.PROGRAMMATIC)) == len(deferred) + len(upfront)


def test_ptc_all_and_profile_names_show_every_stub(tmp_path: Path) -> None:
    tools = [*_shell_tools(tmp_path), *GITHUB, *HF]
    for runtime in (_runtime(search=False), _runtime(function_names="profile")):
        description = _description(runtime, tools)
        assert not runtime.catalog.search_active
        assert SEARCH_FUNCTION_NAME not in description
        assert SEARCH_FUNCTION_NAME not in runtime.catalog.type_check_stubs
        assert all(f"async def {tool.name}(" in description for tool in [*GITHUB, *HF])


def test_directly_callable_functions_stay_upfront(tmp_path: Path) -> None:
    both = Tool(
        name="lookup",
        input_schema={"type": "object"},
        meta={"fast_agent": {"ptc": {"expose": "both"}}},
    )
    runtime = _runtime()
    description = _description(runtime, [*_shell_tools(tmp_path), both, *GITHUB])

    # Three extras: the shared function is upfront and does not count towards search.
    assert not runtime.catalog.search_active
    description = _description(runtime, [*_shell_tools(tmp_path), both, *GITHUB, *HF])
    assert runtime.catalog.search_active
    assert "async def lookup(" in description
    assert "lookup" not in runtime.catalog.deferred


def test_crossing_threshold_and_description_stability(tmp_path: Path) -> None:
    runtime = _runtime()
    shell = _shell_tools(tmp_path)
    inline = _description(runtime, [*shell, *GITHUB])
    assert not runtime.catalog.search_active

    searching = _description(runtime, [*shell, *GITHUB, HF[0]])
    assert runtime.catalog.search_active and searching != inline

    # Connecting another source while search is active leaves the description byte-identical.
    assert _description(runtime, [*shell, *GITHUB, *HF]) == searching
    assert runtime.catalog.roster.find("hf") is not None

    assert _description(runtime, [*shell, *GITHUB]) == inline
    assert not runtime.catalog.search_active


def test_search_without_upfront_tool_functions() -> None:
    runtime = _runtime()
    description = _description(runtime, [*GITHUB, *HF])

    assert runtime.catalog.search_active and not runtime.catalog.upfront
    assert "call the function of the same name here" in description
    assert "network, find them with `search_tools`" in description
    for tool in [*GITHUB, *HF]:
        assert tool.name not in description


def test_schemaless_tools_return_text_or_parsed_json() -> None:
    runtime = _runtime()
    runtime.model_tools([_tool("fetch"), _tool("listing")])
    assert runtime.functions["fetch"].returns == UNSTRUCTURED_RETURNS
    assert "async def fetch() -> str | Any:" in (runtime.tool().description or "")


@pytest.mark.asyncio
async def test_unstructured_returns_type_check_as_text_and_json() -> None:
    runtime = _runtime()
    runtime.model_tools([_tool("fetch"), _tool("listing")])

    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        text = "alpha\nbeta" if name == "fetch" else '{"items": [1, 2]}'
        return CallToolResult(content=[TextContent(type="text", text=text)])

    code = (
        "r = await fetch()\n"
        "lines = r.splitlines()\n"
        "data = await listing()\n"
        '[len(lines), "alpha" in r, data["items"][0]]'
    )
    try:
        result = await runtime.execute({"code": code}, dispatch)
    finally:
        await runtime.close()
    assert not result.is_error, _text(result)
    assert "[2, true, 1]" in _text(result)


# Execution -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deferred_functions_are_callable_and_type_checked() -> None:
    runtime = _runtime()
    runtime.model_tools([*GITHUB, *HF])
    try:
        ok = await runtime.execute({"code": 'await github__list_issues(repo="a/b")'}, _ok)
        bad = await runtime.execute({"code": "await github__list_issues(nope=1)"}, _ok)
    finally:
        await runtime.close()
    assert not ok.is_error and "called github__list_issues" in _text(ok)
    assert bad.is_error and "Type check failed" in _text(bad)


@pytest.mark.asyncio
async def test_search_tools_runs_in_sandbox_and_is_recorded() -> None:
    runtime = _runtime()
    runtime.model_tools([*GITHUB, *HF])
    seen: list[tuple[str, dict[str, Any]]] = []
    try:
        result = await runtime.execute(
            {"code": 'print(await search_tools(query="repository issues", limit=2))'},
            _ok,
            parent_call_id="call-1",
            on_search=lambda call_id, arguments, text: seen.append((call_id, arguments)),
        )
    finally:
        await runtime.close()
    text = _text(result)
    assert not result.is_error, text
    assert "# search_tools: showing 2 of" in text
    assert "async def github__list_issues(" in text
    calls = (result.meta or {})[FAST_AGENT_PTC_CALLS]
    assert [(c["tool"], c["function"], c["status"], c["is_error"]) for c in calls] == [
        ("search_tools", "search_tools", "completed", False)
    ]
    assert calls[0]["arguments"] == {"query": "repository issues", "limit": 2}
    assert seen == [(calls[0]["id"], {"query": "repository issues", "limit": 2})]


@pytest.mark.asyncio
async def test_inactive_search_explains_itself(tmp_path: Path) -> None:
    runtime = _runtime()
    runtime.model_tools([*_shell_tools(tmp_path), *GITHUB])
    try:
        result = await runtime.execute({"code": 'await search_tools(query="x")'}, _ok)
    finally:
        await runtime.close()
    assert result.is_error
    assert "`search_tools` is not active because all available functions" in _text(result)


@pytest.mark.asyncio
async def test_running_program_keeps_its_catalogue() -> None:
    runtime = _runtime()
    runtime.model_tools([*GITHUB, *HF])
    started = asyncio.Event()
    release = asyncio.Event()

    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        if name == "github__list_issues":
            started.set()
            await release.wait()
        return CallToolResult(content=[TextContent(type="text", text=name)])

    code = 'await github__list_issues()\nawait hf__model_search()\nprint(await search_tools(query="model"))'
    try:
        task = asyncio.create_task(runtime.execute({"code": code}, dispatch))
        await started.wait()
        # A refresh mid-run (e.g. /mcp disconnect) applies only to the next run.
        runtime.model_tools(GITHUB)
        release.set()
        result = await task
        after = await runtime.execute({"code": "await hf__model_search()"}, dispatch)
    finally:
        await runtime.close()
    assert not result.is_error, _text(result)
    assert "async def hf__model_search(" in _text(result)
    assert after.is_error and "no longer available" in _text(after)


# Naming ----------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_names_stay_bound_to_their_tools_for_the_conversation() -> None:
    runtime = _runtime()
    first = _tool("a-b", "first")
    second = _tool("a+b", "second")  # sorts before "a-b"; both sanitize to a_b
    runtime.model_tools([first])
    assert runtime.functions["a_b"].tool_name == "a-b"

    runtime.model_tools([second, first])
    assert runtime.functions["a_b"].tool_name == "a-b"
    assert runtime.functions["a_b_2"].tool_name == "a+b"

    runtime.model_tools([second])
    # The removed tool's name is not reassigned: old code cannot reach a different tool.
    assert set(runtime.functions) == {"a_b_2"}
    try:
        result = await runtime.execute({"code": "await a_b()"}, _ok)
    finally:
        await runtime.close()
    assert result.is_error
    assert "`a_b` is no longer available (its tool source was disconnected" in _text(result)

    runtime.reset_conversation()
    runtime.model_tools([second])
    assert set(runtime.functions) == {"a_b"}


def test_search_tools_name_is_reserved() -> None:
    runtime = _runtime()
    runtime.model_tools([_tool("search_tools"), _tool("other")])
    assert runtime.functions["search_tools_tool"].tool_name == "search_tools"
    assert SEARCH_FUNCTION_NAME not in runtime.functions


# Retrieval -------------------------------------------------------------------------------------


CATALOG_TOOLS = [
    *GITHUB,
    *HF,
    _tool("hf__listRepositoryFiles", "Enumerate files.", repo_id={"type": "string"}),
    _tool("hf__space_search", "Search the Hub for the best Spaces to use with the task."),
    _tool("local_notes", "Write notes to the local scratchpad for the user."),
]


def _catalog_runtime(tools: list[Tool] = CATALOG_TOOLS) -> ProgrammaticToolRuntime:
    runtime = _runtime()
    runtime.model_tools(tools)
    assert runtime.catalog.search_active
    return runtime


def _result_names(text: str) -> list[str]:
    return re.findall(r"^async def (\w+)\(", text, flags=re.MULTILINE)


def test_tokenizer_splits_identifiers_and_folds_plurals() -> None:
    assert tokenize("listRepositoryFiles") == tokenize("list repository file")
    assert tokenize("hub_repo-details HTTPServer") == tokenize("hub repo detail http server")
    assert len(tokenize("hub_repo-details HTTPServer")) == 5
    for singular, plural in [
        ("cache", "caches"),
        ("cookie", "cookies"),
        ("movie", "movies"),
        ("size", "sizes"),
        ("status", "statuses"),
        ("space", "spaces"),
        ("search", "searches"),
        ("branch", "branches"),
        ("box", "boxes"),
        ("repository", "repositories"),
        ("query", "queries"),
        ("process", "processes"),
        ("file", "files"),
        ("model", "models"),
        ("match", "matches"),
        ("issue", "issues"),
    ]:
        assert stem(singular) == stem(plural), (singular, plural)


def test_exact_name_ranks_first() -> None:
    catalog = _catalog_runtime().catalog
    for query in ("hf__model_search", "model_search"):
        assert _result_names(search_tools(catalog, query=query))[0] == "hf__model_search"


def test_identifier_and_plural_matching() -> None:
    catalog = _catalog_runtime().catalog
    assert _result_names(search_tools(catalog, query="list repositories files"))[0] == (
        "hf__listRepositoryFiles"
    )
    assert _result_names(search_tools(catalog, query="repo details"))[0] == "hf__hub_repo_details"
    assert "hf__model_search" in _result_names(search_tools(catalog, query="models"))


def test_filler_words_do_not_match_prose() -> None:
    catalog = _catalog_runtime().catalog
    names = _result_names(search_tools(catalog, query="the scratchpad"))
    assert names == ["local_notes"]
    # "the" alone matches several descriptions; with a content word it adds no noise.
    assert _result_names(search_tools(catalog, query="the best")) == ["hf__space_search"]


def test_upfront_functions_found_only_by_exact_name(tmp_path: Path) -> None:
    catalog = _catalog_runtime([*_shell_tools(tmp_path), *CATALOG_TOOLS]).catalog
    assert _result_names(search_tools(catalog, query="bash")) == ["bash"]
    assert "bash" not in _result_names(search_tools(catalog, query="run command"))


def test_server_scoping() -> None:
    catalog = _catalog_runtime().catalog
    scoped = search_tools(catalog, query="search", server="HF")
    assert "(source: hf)" in scoped
    assert set(_result_names(scoped)) <= {"hf__model_search", "hf__space_search"}

    unknown = search_tools(catalog, query="issues", server="gitlab")
    assert "unknown server `gitlab`" in unknown
    assert "- github: 3 functions" in unknown and "- hf: 4 functions" in unknown


def test_empty_query_lists_sources_or_signatures() -> None:
    catalog = _catalog_runtime().catalog
    roster = search_tools(catalog)
    assert "- github: 3 functions" in roster and "- local: 1 function" in roster

    listing = search_tools(catalog, query="", server="github")
    assert "# search_tools: 3 functions in `github`" in listing
    assert "async def github__create_issue(*, title: str = ...) -> str | Any" in listing


@pytest.mark.parametrize(("query", "server"), [(None, None), (None, "missing"), ("zzzzz", None)])
def test_long_source_lists_and_errors_can_be_read_in_full(
    query: str | None, server: str | None
) -> None:
    tools = [_tool(f"source_{i:02d}_検索__fetch") for i in range(80)]
    catalog = _catalog_runtime(tools).catalog
    full = search_tools(catalog, query=query, server=server)
    pages: list[str] = []
    offset = 0
    while True:
        page = search_tools(catalog, query=query, server=server, byte_limit=1024, offset=offset)
        assert len(page.encode()) <= 512
        content, more, cursor = page.partition("\n# Continue with offset=")
        pages.append(content)
        if not more:
            break
        assert int(cursor) > offset
        offset = int(cursor)
    assert len(pages) > 1
    assert "".join(pages) == full
    assert all(f"- source_{i:02d}_検索:" in full for i in range(80))


def test_results_never_cut_a_stub() -> None:
    big = [
        _tool(f"docs__page_{index}", f"Page reader {index}. " + "lorem ipsum " * 300)
        for index in range(6)
    ]
    runtime = _catalog_runtime(big)
    catalog = runtime.catalog
    text = search_tools(catalog, query="page reader", limit=6)
    full = [name for name in catalog.functions if catalog.functions[name].stub() in text]
    compact = text.split("# Additional matches (search exact name for full stub):")[1]

    assert full, text[:200]
    assert len(text) < 12_000
    assert len(full) + len(_result_names(compact)) == 6
    assert set(full).isdisjoint(_result_names(compact))


def test_limit_is_clamped_and_validated() -> None:
    catalog = _catalog_runtime().catalog
    assert "showing 1 of" in search_tools(catalog, query="search", limit=0)
    with pytest.raises(TypeError):
        search_tools(catalog, query="search", limit="2")  # ty: ignore[invalid-argument-type]


def test_results_fit_the_byte_limit_with_multibyte_text() -> None:
    wide = [
        _tool(f"docs__page_{index}", f"Page reader {index}. " + "検索" * 3000) for index in range(6)
    ]
    catalog = _catalog_runtime(wide).catalog
    text = search_tools(catalog, query="page reader", limit=6)

    # The oversized top match keeps its signature; the rest are compact, never partial.
    assert len(text.encode()) < 16_000
    assert "Docstring omitted" in text
    assert _result_names(text) == [f"docs__page_{index}" for index in range(6)]


def test_exact_python_names_outrank_unprefixed_names(tmp_path: Path) -> None:
    catalog = _catalog_runtime(
        [*_shell_tools(tmp_path), *CATALOG_TOOLS, _tool("svc__bash", "Another bash.")]
    ).catalog
    assert _result_names(search_tools(catalog, query="bash", limit=1)) == ["bash"]
    assert _result_names(search_tools(catalog, query="svc__bash", limit=1)) == ["svc__bash"]


def test_server_spelling_prefers_exact_then_unique_matches() -> None:
    catalog = _catalog_runtime(
        [*CATALOG_TOOLS, _tool("DEV__deploy", "Deploy."), _tool("dev__deploy", "Deploy.")]
    ).catalog
    assert _result_names(search_tools(catalog, query="deploy", server="dev")) == ["dev__deploy"]
    assert "unknown server `Dev`" in search_tools(catalog, query="deploy", server="Dev")
    assert "(source: github)" in search_tools(catalog, query="issue", server="GitHub")


def test_filler_name_matches_only_when_nothing_else_matches() -> None:
    catalog = _catalog_runtime().catalog
    fallback = _result_names(search_tools(catalog, query="get potatoes"))
    assert fallback == ["github__get_file_contents"]
    issues = _result_names(search_tools(catalog, query="get issues"))
    assert sorted(issues) == ["github__create_issue", "github__list_issues"]


def test_filler_only_queries_match_names_and_parameters_not_prose() -> None:
    catalog = _catalog_runtime(
        [
            *CATALOG_TOOLS,
            _tool("svc__fetch", "Get the latest value."),
            _tool("svc__lookup", "Look up a key.", get={"type": "string"}),
        ]
    ).catalog
    names = _result_names(search_tools(catalog, query="get", limit=20))
    assert {"github__get_file_contents", "svc__lookup"} <= set(names)
    assert "svc__fetch" not in names
    assert _result_names(search_tools(catalog, query="the", limit=20)) == []


@pytest.mark.asyncio
async def test_exclude_skips_seen_functions_in_code_and_direct_search() -> None:
    runtime = _catalog_runtime()
    catalog = runtime.catalog

    # Exclude works with Python name or unprefixed local name, advancing lower-ranked hits into limit.
    first = _result_names(search_tools(catalog, query="issue", server="github", limit=1))
    assert first == ["github__create_issue"]
    second = _result_names(
        search_tools(catalog, query="issue", server="github", limit=1, exclude=first)
    )
    assert second == ["github__list_issues"]
    by_local = _result_names(
        search_tools(
            catalog,
            query="issue",
            server="github",
            exclude=("list_issues", "create_issue"),
        )
    )
    assert by_local == []
    exhausted = search_tools(
        catalog,
        query="issue",
        server="github",
        exclude={"github__list_issues", "github__create_issue"},
    )
    assert "no remaining functions match `issue` in `github` (2 matches excluded)" in exhausted

    # Source signature listing also respects exclude.
    listed = search_tools(catalog, server="github", exclude=["github__list_issues"])
    assert "2 functions in `github` (after `exclude`)" in listed
    assert "github__list_issues" not in _result_names(listed)
    assert sorted(_result_names(listed)) == ["github__create_issue", "github__get_file_contents"]

    # Rejects bare strings (prevents iterating over characters) and non-string items.
    with pytest.raises(TypeError, match="exclude must be a list, tuple or set"):
        search_tools(catalog, query="issue", exclude="github__list_issues")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="exclude items must be str"):
        search_tools(catalog, query="issue", exclude=[123])  # ty: ignore[invalid-argument-type]

    # Works end-to-end inside Monty with type checking enabled.
    try:
        result = await runtime.execute(
            {
                "code": (
                    'print(await search_tools(query="issue", server="github", '
                    'limit=1, exclude=["github__list_issues"]))'
                )
            },
            _ok,
        )
        assert not result.is_error, _text(result)
        assert "async def github__create_issue(" in _text(result)
        assert "async def github__list_issues(" not in _text(result)

        bad_type = await runtime.execute(
            {"code": 'await search_tools(query="issue", exclude="github__list_issues")'},
            _ok,
        )
        assert bad_type.is_error and "NOT executed" in _text(bad_type)
    finally:
        await runtime.close()
