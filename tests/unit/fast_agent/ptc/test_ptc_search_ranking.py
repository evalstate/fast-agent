"""``search_tools`` ranking, retired names, Unicode queries, caching and the byte limit."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import Tool

from fast_agent.config import ProgrammaticToolCallingSettings
from fast_agent.ptc import search
from fast_agent.ptc.runtime import ProgrammaticToolRuntime
from fast_agent.ptc.search import search_tools

if TYPE_CHECKING:
    from fast_agent.ptc.catalog import PtcCatalog

pytest.importorskip("pydantic_monty")


def _tool(name: str, description: str = "", **properties: dict[str, Any]) -> Tool:
    return Tool(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": properties},
    )


TOOLS = [
    _tool("github__list_issues", "List issues in a repository.", repo={"type": "string"}),
    _tool("github__create_issue", "Open a new issue.", title={"type": "string"}),
    _tool("github__get_file_contents", "Read a file from a repository."),
    _tool("hf__listRepositoryFiles", "Enumerate files.", repo_id={"type": "string"}),
    _tool("hf__model_search", "Find models on the Hub by task or author."),
]


def _catalog(tools: list[Tool]) -> PtcCatalog:
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings())
    runtime.model_tools(tools)
    assert runtime.catalog.search_active
    return runtime.catalog


def _names(text: str) -> list[str]:
    return re.findall(r"^async def (\w+)\(", text, flags=re.MULTILINE)


def test_verbatim_name_wins_case_folded_collisions() -> None:
    catalog = _catalog([*TOOLS, _tool("DEV__deploy", "Deploy."), _tool("dev__deploy", "Deploy.")])
    assert _names(search_tools(catalog, query="dev__deploy", limit=1)) == ["dev__deploy"]
    assert _names(search_tools(catalog, query="DEV__deploy", limit=1)) == ["DEV__deploy"]
    # Neither spelling is verbatim: the tie breaks the same way every time.
    assert _names(search_tools(catalog, query="Dev__Deploy", limit=2)) == [
        "DEV__deploy",
        "dev__deploy",
    ]


def test_run_together_name_matches_without_shared_tokens() -> None:
    catalog = _catalog(TOOLS)
    assert _names(search_tools(catalog, query="listrepositoryfiles"))[0] == (
        "hf__listRepositoryFiles"
    )


def test_retired_name_is_reported_before_sibling_matches() -> None:
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings())
    runtime.model_tools([*TOOLS, _tool("hf__space_search", "Search Spaces.")])
    runtime.model_tools([tool for tool in TOOLS if tool.name != "github__list_issues"])
    catalog = runtime.catalog
    assert "github__list_issues" in catalog.retired

    text = search_tools(catalog, query="github__list_issues")
    assert text.startswith(
        "# Note: `github__list_issues` is no longer available "
        "(its tool source was disconnected or removed)."
    )
    assert "github__create_issue" in _names(text)
    assert search_tools(catalog, query="GitHub__List_Issues").startswith(
        "# Note: `github__list_issues`"
    )
    gone = search_tools(catalog, query="hf__space_search", server="github")
    assert gone == (
        "# search_tools: `hf__space_search` is no longer available "
        "(its tool source was disconnected or removed)."
    )


def test_unicode_queries_match_descriptions() -> None:
    catalog = _catalog(
        [
            *TOOLS,
            _tool("menu__order", "Commander un café au comptoir."),
            _tool("docs__find", "リポジトリを検索する。"),
        ]
    )
    assert _names(search_tools(catalog, query="café"))[0] == "menu__order"
    assert _names(search_tools(catalog, query="cafés"))[0] == "menu__order"
    assert _names(search_tools(catalog, query="検索"))[0] == "docs__find"


def test_documents_are_built_once_per_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _catalog(TOOLS)
    built: list[str] = []
    document = search._document

    def counting(*args: Any) -> search.SearchDocument:
        result = document(*args)
        built.append(result.function.python_name)
        return result

    monkeypatch.setattr(search, "_document", counting)
    search_tools(catalog)
    search_tools(catalog, server="github")
    assert built == []  # listings need no search documents

    search_tools(catalog, query="issues")
    search_tools(catalog, query="files", server="hf")
    assert sorted(built) == sorted(catalog.functions)


def _signatures_are_whole(text: str) -> None:
    for line in text.splitlines():
        if line.startswith("async def "):
            assert re.fullmatch(r"async def \w+\(.*\) -> .+", line), line


def test_results_fit_a_small_byte_limit() -> None:
    wide = [
        _tool(
            f"docs__page_reader_{index}",
            f"Page reader {index}. " + "lorem ipsum " * 300,
            **{f"option_{n}": {"type": "string"} for n in range(index * 3)},
        )
        for index in range(20)
    ]
    catalog = _catalog(wide)
    for byte_limit in (1000, 3000, 6000):
        text = search_tools(catalog, query="page reader", limit=15, byte_limit=byte_limit)
        # Within the limit, leaving room for exec_sandbox's framing of the result.
        assert len(text.encode()) < byte_limit - 100
        _signatures_are_whole(text)
        shown = re.search(r"showing (\d+) of 20 matches", text)
        assert shown is not None and int(shown[1]) == len(_names(text))
        if byte_limit == 1000:
            # Cut by size, not by `limit`: a higher limit would not show more.
            assert int(shown[1]) < 15
            assert "more matches: refine the query." in text and "raise `limit`" not in text

        listing = search_tools(catalog, server="docs", byte_limit=byte_limit)
        assert len(listing.encode()) < byte_limit - 100
        _signatures_are_whole(listing)
        assert "more; pass a `query` to search this source." in listing


MANY_PARAMETERS = {f"parameter_with_a_long_name_{n}": {"type": "string"} for n in range(300)}


def test_oversized_signature_collapses_its_parameters() -> None:
    huge = _tool("docs__huge", "A tool with very many parameters.", **MANY_PARAMETERS)
    text = search_tools(_catalog([*TOOLS, huge]), query="docs__huge", byte_limit=4000)
    assert len(text.encode()) <= 4000
    assert "async def docs__huge(...) -> ...:" in text
    assert "parameter_with_a_long_name" not in text

    # A compact signature too large to fit is collapsed; smaller ones after it still list.
    widgets = [
        _tool("docs__alpha", "Report widget."),
        _tool("docs__huge", "Report widget.", **MANY_PARAMETERS),
        _tool("docs__zeta", "Report widget."),
    ]
    text = search_tools(_catalog([*TOOLS, *widgets]), query="widget", byte_limit=4000)
    assert _names(text) == ["docs__alpha", "docs__huge", "docs__zeta"]
    assert "async def docs__huge(...) -> ...\nasync def docs__zeta() -> str | Any" in text
    _signatures_are_whole(text)


def test_counts_and_omitted_hints() -> None:
    catalog = _catalog([*TOOLS, _tool("notes__write", "Write a note.")])
    assert "# search_tools: 1 function in `notes`" in search_tools(catalog, server="notes")
    limited = search_tools(catalog, query="repository", limit=1)
    assert "showing 1 of 3 matches" in limited
    assert "# 2 more matches: raise `limit` or refine the query." in limited
    single = search_tools(catalog, query="note")
    assert "showing 1 of 1 match" in single and "more match" not in single
