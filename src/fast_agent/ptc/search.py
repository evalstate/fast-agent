"""``search_tools``: find deferred sandbox functions and return their definitions.

Retrieval is deterministic lexical search: identifier-aware Unicode tokens, light plural folding
and field-weighted BM25 (BM25F), with verbatim names ranked first, then case-insensitive and
identifier matches. Documents are built once per catalogue. Results are the same canonical stubs
the ``exec_sandbox`` description and ``^tool:`` mentions use, sized to fit the result the model
sees and never cut partway through. A query naming a function that has gone says so first.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from fast_agent.mcp.common import get_resource_name, is_namespaced_name
from fast_agent.ptc.catalog import unavailable_tool_message
from fast_agent.ptc.stubs import tool_ptc_options

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mcp_types import Tool

    from fast_agent.ptc.catalog import PtcCatalog
    from fast_agent.ptc.stubs import PtcFunction

SEARCH_FUNCTION_NAME: Final = "search_tools"
SEARCH_STUBS: Final = '''\
async def search_tools(
    *,
    query: str | None = None,
    server: str | None = None,
    limit: int = 5,
    exclude: list[str] | tuple[str, ...] | set[str] | None = None,
    offset: int = 0,
) -> str:
    """Search deferred sandbox functions by task keywords (e.g. "list repository files") or exact function name.
    Pass `server` to scope to one source from <fastagent:tool-sources>, or `exclude` to skip function names already seen.
    Omit `query` to list searchable sources. If a result continues, repeat the call with the supplied `offset`.
    Print or return the result to inspect full callable Python stubs."""
    ...
'''
DEFAULT_LIMIT: Final = 5
MAX_LIMIT: Final = 20
# Budgets are UTF-8 bytes, like the limit on what the model sees of an exec_sandbox result.
_SEARCH_RESULT_BYTE_LIMIT = 13_000
"""The whole search result, unless the model's result limit is smaller."""
_RESULT_FRAMING_BYTES = 512
"""Room left for exec_sandbox's own framing (status header, ``Output:``) around the text."""
_FULL_STUBS_BYTE_BUDGET = 8000
"""Full definitions after the top match stop here; further matches are compact signatures."""
_TOP_STUB_BYTE_LIMIT = 12_000
"""The top match is shown in full up to this size; larger definitions lose their docstring."""
_K1 = 1.2
_B = 0.75
_PRIMARY_WEIGHT = 3.5
_PARAMETER_WEIGHT = 2.0
_SECONDARY_WEIGHT = 1.0
# Verbatim Python or tool names outrank case-insensitive ones, which outrank exact names
# without the server prefix, which outrank identifier matches, which outrank prose.
_VERBATIM_NAME_BOOST = 4_000.0
_EXACT_NAME_BOOST = 2_000.0
_LOCAL_NAME_BOOST = 1_000.0
_NAME_MATCH_BOOST = 100.0
STOP_WORDS: Final = frozenset(
    {"a", "an", "the", "to", "for", "in", "of", "on", "and", "with", "get", "use", "run", "tool"}
)
# Han and kana are written without spaces, so each character is a token.
_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_WORD = re.compile(rf"[{_CJK}]|[^\W\d_{_CJK}]+|\d+")
_ASCII_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")
_COMPACT_HEADING = "\n# Additional matches (search exact name for full stub):"


def tokenize(text: str) -> list[str]:
    """Split text and identifiers (snake_case, camelCase, kebab-case) into folded stems.

    Words are runs of Unicode letters; ASCII words are also split at camelCase boundaries.
    """
    return [
        stem(part.casefold())
        for word in _WORD.findall(text)
        for part in (_ASCII_PART.findall(word) if word.isascii() and word.isalpha() else [word])
    ]


def stem(token: str) -> str:
    """Fold singular and plural to one stem: ``repositories`` and ``repository`` -> ``repositori``.

    Drop a plural ``s``, then a final ``e`` (``caches``/``cache`` -> ``cach``), then turn a final
    ``y`` into ``i`` (``queries``/``query`` -> ``queri``). Words ending ``ss``, ``us`` or ``is``
    are singular (``process``, ``status``, ``analysis``).
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]
    if len(token) > 2 and token.endswith("y"):
        token = token[:-1] + "i"
    return token


def _identifier(text: str) -> str:
    return re.sub(r"[\W_]+", "_", text.casefold()).strip("_")


@dataclass(frozen=True, slots=True)
class SearchDocument:
    """A function's searchable text, built once per catalogue (``PtcCatalog.search_index``)."""

    function: PtcFunction
    source: str
    names: frozenset[str]
    """Exact names, case-folded: the Python name and the tool name."""
    local_name: str
    """The case-folded tool name without its server prefix."""
    name_tokens: frozenset[str]
    fields: tuple[tuple[float, tuple[str, ...]], ...]
    """(weight, tokens) for the primary, parameter and secondary fields."""


def _document(function: PtcFunction, tool: Tool, source: str) -> SearchDocument:
    local_name = (
        get_resource_name(function.tool_name)
        if is_namespaced_name(function.tool_name)
        else function.tool_name
    )
    titles = [tool.title or ""]
    if tool.annotations is not None and tool.annotations.title:
        titles.append(tool.annotations.title)
    primary = tokenize(" ".join([function.python_name, function.tool_name, *titles, source]))
    parameters: list[str] = []
    secondary = tokenize(function.description)
    schema = _input_schema(tool)
    properties = schema.get("properties")
    for name, property_schema in properties.items() if isinstance(properties, dict) else []:
        parameters.extend(tokenize(str(name)))
        if isinstance(property_schema, dict):
            values = property_schema.get("enum")
            if isinstance(values, list):
                parameters.extend(tokenize(" ".join(str(value) for value in values)))
            description = property_schema.get("description")
            if isinstance(description, str):
                secondary.extend(tokenize(description))
    return SearchDocument(
        function=function,
        source=source,
        names=frozenset({function.python_name.casefold(), function.tool_name.casefold()}),
        local_name=local_name.casefold(),
        name_tokens=frozenset(tokenize(f"{function.python_name} {local_name}")),
        fields=(
            (_PRIMARY_WEIGHT, tuple(primary)),
            (_PARAMETER_WEIGHT, tuple(parameters)),
            (_SECONDARY_WEIGHT, tuple(secondary)),
        ),
    )


def _input_schema(tool: Tool) -> Mapping[str, Any]:
    ptc_schema = tool_ptc_options(tool).get("input_schema")
    return ptc_schema if isinstance(ptc_schema, dict) else tool.input_schema


def _documents(catalog: PtcCatalog) -> dict[str, SearchDocument]:
    """Every function's document, tokenized on the catalogue's first search and then reused."""
    index = catalog.search_index
    if not index:
        index.update(
            (name, _document(function, catalog.tools[function.tool_name], catalog.sources[name]))
            for name, function in catalog.functions.items()
        )
    return index


@dataclass(frozen=True, slots=True)
class _Match:
    document: SearchDocument
    score: float
    primary_hits: int


@dataclass(frozen=True, slots=True)
class _Query:
    verbatim: str
    folded: str
    identifier: str
    meaningful: tuple[str, ...]

    def name_boost(self, document: SearchDocument) -> float:
        function = document.function
        if self.verbatim in (function.python_name, function.tool_name):
            return _VERBATIM_NAME_BOOST
        if self.folded in document.names:
            return _EXACT_NAME_BOOST
        if self.folded == document.local_name:
            return _LOCAL_NAME_BOOST
        if (self.identifier and any(self.identifier in name for name in document.names)) or (
            self.meaningful and all(term in document.name_tokens for term in self.meaningful)
        ):
            return _NAME_MATCH_BOOST
        return 0.0


def rank(
    documents: list[SearchDocument], query: str, *, exact_only: Iterable[SearchDocument] = ()
) -> list[_Match]:
    """Rank ``documents`` for ``query``; ``exact_only`` documents match only by exact name."""
    terms = list(dict.fromkeys(tokenize(query)))
    meaningful = tuple(term for term in terms if term not in STOP_WORDS)
    wanted = _Query(query.strip(), query.strip().casefold(), _identifier(query), meaningful)
    exact = [
        _Match(document, boost, 0)
        for document in exact_only
        if (boost := wanted.name_boost(document)) >= _EXACT_NAME_BOOST
    ]
    count = len(documents)
    averages = [
        max(sum(len(d.fields[i][1]) for d in documents) / count, 1.0) if count else 1.0
        for i in range(3)
    ]
    frequency = {
        term: sum(1 for d in documents if any(term in tokens for _, tokens in d.fields))
        for term in terms
    }
    strong: list[_Match] = []
    weak: list[_Match] = []
    for document in documents:
        score = 0.0
        primary_hits = 0
        meaningful_hit = filler_name_hit = False
        for term in terms:
            weighted = 0.0
            for index, (weight, tokens) in enumerate(document.fields):
                hits = tokens.count(term)
                if not hits:
                    continue
                if term not in STOP_WORDS:
                    meaningful_hit = True
                elif index < 2:
                    filler_name_hit = True
                if index == 0:
                    primary_hits += hits
                norm = 1 - _B + _B * len(tokens) / averages[index]
                weighted += weight * hits / norm
            if weighted:
                idf = math.log(1 + (count - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * weighted / (_K1 + weighted)
        # A name can match without any token in common, e.g. ``listrepositoryfiles``.
        boost = wanted.name_boost(document)
        if score <= 0 and not boost:
            continue
        match = _Match(document, score + boost, primary_hits)
        # Filler words ("get", "the", ...) count only when no content word matches anything,
        # and then only in names and parameters, never in prose.
        if meaningful_hit or boost or (not meaningful and filler_name_hit):
            strong.append(match)
        elif filler_name_hit:
            weak.append(match)
    return sorted(
        [*exact, *(strong or weak)],
        key=lambda m: (
            -m.score,
            -m.primary_hits,
            m.document.function.python_name.casefold(),
            m.document.function.python_name,
        ),
    )


def _excluded_names(
    exclude: list[str] | tuple[str, ...] | set[str] | frozenset[str] | None,
) -> frozenset[str]:
    if exclude is None:
        return frozenset()
    if isinstance(exclude, str | bytes) or not isinstance(exclude, list | tuple | set | frozenset):
        raise TypeError("search_tools() exclude must be a list, tuple or set of str, or None")
    names: set[str] = set()
    for item in exclude:
        if not isinstance(item, str):
            raise TypeError("search_tools() exclude items must be str")
        stripped = item.strip()
        if stripped:
            names.add(stripped)
    return frozenset(names)


def _is_excluded(function: PtcFunction, excluded: frozenset[str]) -> bool:
    if not excluded:
        return False
    local_name = (
        get_resource_name(function.tool_name)
        if is_namespaced_name(function.tool_name)
        else function.tool_name
    )
    return (
        function.python_name in excluded or function.tool_name in excluded or local_name in excluded
    )


def search_tools(
    catalog: PtcCatalog,
    *,
    query: str | None = None,
    server: str | None = None,
    limit: int = DEFAULT_LIMIT,
    exclude: list[str] | tuple[str, ...] | set[str] | frozenset[str] | None = None,
    offset: int = 0,
    byte_limit: int | None = None,
) -> str:
    """Run a ``search_tools`` call from sandbox code against ``catalog``.

    ``byte_limit`` is how much of an ``exec_sandbox`` result the model sees; the text returned
    fits inside it with room for the result's framing.
    """
    if query is not None and not isinstance(query, str):
        raise TypeError("search_tools() query must be a str or None")
    if server is not None and not isinstance(server, str):
        raise TypeError("search_tools() server must be a str or None")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("search_tools() limit must be an int")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("search_tools() offset must be a non-negative integer")
    excluded = _excluded_names(exclude)
    limit = min(max(limit, 1), MAX_LIMIT)
    budget = _SEARCH_RESULT_BYTE_LIMIT
    if byte_limit is not None:
        framing = min(_RESULT_FRAMING_BYTES, byte_limit // 2)
        budget = max(0, min(budget, byte_limit - framing))
    result = _search_result(catalog, query, server, limit, excluded, budget)
    return _result_page(result, offset, budget)


def _result_page(result: str, offset: int, budget: int) -> str:
    """Fit the returned text and its continuation in the budget, preferring whole lines."""
    remaining = result[offset:]
    encoded = remaining.encode()
    if len(encoded) <= budget:
        return remaining
    continuation = "\n# Continue with offset={}"
    room = budget - len(continuation.format(len(result)))
    if room < 4:  # Insufficient room for one UTF-8 character and a continuation.
        return "Result limit too small."[:budget]
    prefix = encoded[:room].decode(errors="ignore")
    end = prefix.rfind("\n") + 1 or len(prefix)
    return prefix[:end] + continuation.format(offset + end)


def _search_result(
    catalog: PtcCatalog,
    query: str | None,
    server: str | None,
    limit: int,
    excluded: frozenset[str],
    budget: int,
) -> str:
    source = None
    if server is not None and server.strip():
        source = _resolve_source(catalog, server)
        if source is None:
            return _unknown_source(catalog, server)
        entry = catalog.roster.find(source)
        if entry is not None and entry.functions == 0:
            note = f" ({entry.note})" if entry.note else ""
            return f"# search_tools: source `{source}` has no searchable functions{note}."
    if query is None or not query.strip():
        if source is None:
            return _list_sources(catalog)
        functions = [
            function
            for name, function in catalog.functions.items()
            if name in catalog.deferred
            and catalog.sources[name] == source
            and not _is_excluded(function, excluded)
        ]
        return _list_source(functions, source, budget, excluded=bool(excluded))
    index = _documents(catalog)
    documents = [
        document
        for name, document in index.items()
        if name in catalog.deferred and (source is None or document.source == source)
    ]
    upfront = (
        [
            document
            for name, document in index.items()
            if name not in catalog.deferred and not _is_excluded(document.function, excluded)
        ]
        if source is None
        else []
    )
    all_matches = rank(documents, query, exact_only=upfront)
    matches = [
        match for match in all_matches if not _is_excluded(match.document.function, excluded)
    ]
    retired = _retired_name(catalog, query)
    if not matches:
        if retired is not None:
            return f"# search_tools: {unavailable_tool_message(retired)}"
        return _no_matches(
            catalog, query, source, excluded_hits=len(all_matches) if excluded else 0
        )
    note = f"# Note: {unavailable_tool_message(retired)}" if retired is not None else None
    return _render_matches(matches, limit, source, budget, note)


def _retired_name(catalog: PtcCatalog, query: str) -> str | None:
    """The function ``query`` names, verbatim or case-insensitively, if its tool has gone."""
    name = query.strip()
    if name in catalog.retired:
        return name
    if name in catalog.functions:
        return None
    folded = name.casefold()
    return next((old for old in sorted(catalog.retired) if old.casefold() == folded), None)


def _resolve_source(catalog: PtcCatalog, server: str) -> str | None:
    """Match ``server`` exactly, else case-insensitively or ignoring punctuation if unique."""
    sources = set(catalog.sources.values()) | {entry.name for entry in catalog.roster.entries}
    wanted = server.strip()
    if wanted in sources:
        return wanted
    for normalize in (_fold, _compact):
        matches = [name for name in sources if normalize(name) == normalize(wanted)]
        if len(matches) == 1:
            return matches[0]
    return None


def _fold(name: str) -> str:
    return name.casefold()


def _compact(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", name.casefold())


def _unknown_source(catalog: PtcCatalog, server: str) -> str:
    return (
        f"# search_tools: unknown server `{server}`.\n"
        + _source_lines(catalog)
        + "\nPass one of these names as `server`, or omit `server` to search every source."
    )


def _source_lines(catalog: PtcCatalog) -> str:
    entries = catalog.roster.entries
    if not entries:
        return "No sources have searchable functions."
    lines = ["Searchable sources:"]
    for entry in entries:
        note = f" ({entry.note})" if entry.note else ""
        lines.append(f"- {entry.name}: {_count(entry.functions, 'function')}{note}")
    return "\n".join(lines)


def _list_sources(catalog: PtcCatalog) -> str:
    total = sum(entry.functions for entry in catalog.roster.entries)
    return (
        f"# search_tools: {_count(total, 'searchable function')}\n"
        + _source_lines(catalog)
        + '\nSearch with `await search_tools(query="operation object", server="...")`.'
    )


def _list_source(
    functions: list[PtcFunction], source: str, budget: int, *, excluded: bool = False
) -> str:
    suffix = " (after `exclude`)" if excluded else ""
    header = f"# search_tools: {_count(len(functions), 'function')} in `{source}`{suffix}"
    footer = "# Search by exact name for a function's full definition."
    more = "# ... {} more; pass a `query` to search this source."
    room = budget - _size(header) - _size(footer) - _size(more.format(len(functions))) - 2
    lines = [header]
    skipped = 0
    for function in functions:
        line = _signature_line(function, room)
        if line is None:
            skipped += 1
            continue
        lines.append(line)
        room -= _size(line) + 1
    if skipped:
        lines.append(more.format(skipped))
    lines.append(footer)
    return "\n".join(lines)


def _signature_line(function: PtcFunction, room: int) -> str | None:
    """``function``'s one-line signature, or with its parameters collapsed, in ``room`` bytes."""
    for line in (
        f"async def {function.signature()}",
        f"async def {function.python_name}(...) -> ...",
    ):
        if _size(line) + 1 <= room:
            return line
    return None


def _no_matches(
    catalog: PtcCatalog, query: str, source: str | None, *, excluded_hits: int = 0
) -> str:
    scope = f" in `{source}`" if source else ""
    if excluded_hits:
        return (
            f"# search_tools: no remaining functions match `{query.strip()}`{scope} "
            f"({_count(excluded_hits, 'match', 'matches')} excluded)."
        )
    return (
        f"# search_tools: no functions match `{query.strip()}`{scope}. Try other task words "
        "(an operation and an object), or a broader query.\n" + _source_lines(catalog)
    )


def _render_matches(
    matches: list[_Match], limit: int, source: str | None, budget: int, note: str | None
) -> str:
    """Render full stubs, then compact signatures, within ``budget``; never part of a stub."""
    shown = matches[:limit]
    scope = f" (source: {source})" if source else ""
    total = _count(len(matches), "match", "matches")
    lines = [note] if note else []
    header_line = len(lines)
    lines.append(f"# search_tools: showing {len(shown)} of {total}{scope}")
    # Keep room for the compact heading and the omitted-matches line.
    room = (
        budget
        - _size("\n".join(lines))
        - _size(_COMPACT_HEADING)
        - _size(_omitted_line(len(matches), raise_limit=True))
        - 2
    )
    full_room = min(room, _FULL_STUBS_BYTE_BUDGET)
    compact: list[str] = []
    compacting = False
    skipped = 0
    for index, match in enumerate(shown):
        document = match.document
        block = f"\n# source: {document.source}\n{document.function.stub()}"
        if index == 0:
            block = _top_block(document, block, room)
        elif compacting or _size(block) + 1 > full_room:
            # Once one definition does not fit, the rest keep rank order as signatures.
            compacting = True
            line = _signature_line(document.function, room)
            if line is None:
                skipped += 1
            else:
                compact.append(line)
                room -= _size(line) + 1
            continue
        lines.append(block)
        room -= _size(block) + 1
        full_room -= _size(block) + 1
    if compact:
        lines.append(_COMPACT_HEADING)
        lines.extend(compact)
    omitted = len(matches) - len(shown) + skipped
    if omitted:
        lines.append(_omitted_line(omitted, raise_limit=not skipped and limit < MAX_LIMIT))
    lines[header_line] = f"# search_tools: showing {len(matches) - omitted} of {total}{scope}"
    return "\n".join(lines)


def _top_block(document: SearchDocument, block: str, room: int) -> str:
    """The top match in full if it fits, else without its docstring or, at worst, parameters."""
    if _size(block) + 1 <= min(room, _TOP_STUB_BYTE_LIMIT):
        return block
    function = document.function
    source = f"\n# source: {document.source}\n"
    without_docstring = (
        f"{source}async def {function.signature()}:\n"
        f"    # Docstring omitted: the definition is {_size(block)} bytes.\n    ..."
    )
    if _size(without_docstring) + 1 <= room:
        return without_docstring
    return (
        f"{source}async def {function.python_name}(...) -> ...:\n"
        f"    # Parameters omitted: the definition is {_size(block)} bytes, over the result limit.\n"
        "    ..."
    )


def _omitted_line(count: int, *, raise_limit: bool) -> str:
    advice = "raise `limit` or refine the query" if raise_limit else "refine the query"
    return f"\n# {_count(count, 'more match', 'more matches')}: {advice}."


def _count(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def _size(text: str) -> int:
    return len(text.encode())
