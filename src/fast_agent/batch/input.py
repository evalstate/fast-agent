"""Input row loading and selection for batch runs."""

from __future__ import annotations

import csv
import importlib
import json
import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import ParseResult, parse_qs, urlparse

from fast_agent.utils.action_normalization import split_first_token
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import BinaryIO, TextIO


SUPPORTED_INPUT_SUFFIXES = frozenset({".jsonl", ".csv", ".parquet"})
REMOTE_HTTP_PREFIXES = ("http://", "https://")
HF_DATASET_QUERY_KEYS = frozenset({"config", "split"})


class HfInputFileSystem(Protocol):
    def open(self, path: str, mode: str = "rb") -> BinaryIO: ...
    def find(self, path: str) -> list[str] | dict[str, dict[str, Any]]: ...


def _is_remote_http_source(source: str) -> bool:
    return source.startswith(REMOTE_HTTP_PREFIXES)


def _path_suffix(path: str | Path) -> str:
    return strip_casefold(Path(path).suffix)


def _source_path_suffix(source: str) -> str:
    return _path_suffix(urlparse(source).path)


def _has_supported_input_suffix(path: str | Path) -> bool:
    return _path_suffix(path) in SUPPORTED_INPUT_SUFFIXES


def _is_parquet_suffix(path: str | Path) -> bool:
    return _path_suffix(path) == ".parquet"


@dataclass(frozen=True)
class RowError:
    type: str
    message: str


@dataclass(frozen=True)
class RowCandidate:
    row_number: int
    row: dict[str, Any] | None
    error: RowError | None = None


@dataclass(frozen=True, slots=True)
class ParsedHfDatasetInput:
    repo_id: str
    config: str | None
    split: str | None


@dataclass(frozen=True, slots=True)
class HfDatasetSource:
    is_repository: bool
    suffix: str
    has_query: bool


def iter_jsonl_stream(handle: TextIO) -> Iterable[RowCandidate]:
    """Yield JSON object rows, preserving invalid lines as row-error candidates."""
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            yield RowCandidate(
                row_number=line_number,
                row=None,
                error=RowError("InvalidJSON", f"Line {line_number}: {exc.msg}"),
            )
            continue

        if not isinstance(payload, dict):
            yield RowCandidate(
                row_number=line_number,
                row=None,
                error=RowError(
                    "InvalidRow",
                    f"Line {line_number}: expected a JSON object, got {type(payload).__name__}",
                ),
            )
            continue

        yield RowCandidate(row_number=line_number, row=payload)


def iter_jsonl_rows(path: Path) -> Iterable[RowCandidate]:
    with path.open("r", encoding="utf-8") as handle:
        yield from iter_jsonl_stream(handle)


def iter_csv_stream(handle: TextIO) -> Iterable[RowCandidate]:
    """Yield CSV rows as dictionaries keyed by header name."""
    reader = csv.DictReader(handle)
    header_error = _csv_header_error(reader.fieldnames)
    if header_error is not None:
        yield RowCandidate(row_number=1, row=None, error=header_error)
        return

    for row_number, row in enumerate(reader, start=2):
        if None in row:
            yield RowCandidate(
                row_number=row_number,
                row=None,
                error=RowError("InvalidRow", f"Line {row_number}: too many CSV columns"),
            )
            continue
        yield RowCandidate(row_number=row_number, row=dict(row))


def _csv_header_error(fieldnames: Sequence[str] | None) -> RowError | None:
    if fieldnames is None:
        return RowError("InvalidRow", "CSV input is missing a header row")

    blank_headers = [index + 1 for index, name in enumerate(fieldnames) if not name.strip()]
    if blank_headers:
        formatted = ", ".join(str(index) for index in blank_headers)
        return RowError("InvalidRow", f"CSV header contains blank column names: {formatted}")

    duplicates = _duplicate_csv_headers(fieldnames)
    if duplicates:
        formatted = ", ".join(duplicates)
        return RowError("InvalidRow", f"CSV header contains duplicate column names: {formatted}")

    return None


def _duplicate_csv_headers(fieldnames: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in fieldnames:
        if name in seen:
            duplicates.add(name)
        else:
            seen.add(name)
    return sorted(duplicates)


def iter_csv_rows(path: Path) -> Iterable[RowCandidate]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from iter_csv_stream(handle)


def iter_hf_rows(
    source: str,
    *,
    filesystem: HfInputFileSystem | None = None,
    offset: int | None = None,
    limit: int | None = None,
    sql: str | None = None,
) -> Iterable[RowCandidate]:
    """Yield input rows from a Hugging Face Hub file addressed by an hf:// URI."""
    fs = filesystem if filesystem is not None else _default_hf_filesystem()
    resolved_source = _resolve_hf_rows_source(source, fs, sql=sql)
    yield from _iter_hf_resolved_rows(
        resolved_source,
        fs,
        offset=offset,
        limit=limit,
        sql=sql,
    )


def _resolve_hf_rows_source(
    source: str,
    filesystem: HfInputFileSystem,
    *,
    sql: str | None,
) -> str:
    parsed = urlparse(source)
    dataset_source = _parse_hf_dataset_source(parsed, source)
    if sql is None or dataset_source is None or not dataset_source.is_repository:
        return _resolve_hf_input_source(source, filesystem)

    parquet_urls = _list_hf_dataset_parquet_urls(source)
    if not parquet_urls:
        raise ValueError(f"Hugging Face dataset input {source} has no matching parquet files")
    return parquet_urls[0] if len(parquet_urls) == 1 else _parquet_sources_token(parquet_urls)


def _iter_hf_resolved_rows(
    source: str,
    filesystem: HfInputFileSystem,
    *,
    offset: int | None,
    limit: int | None,
    sql: str | None,
) -> Iterable[RowCandidate]:
    parquet_sources = _parquet_sources_from_token(source)
    if parquet_sources is not None:
        yield from iter_parquet_rows(parquet_sources, offset=offset, limit=limit, sql=sql)
        return

    suffix = _source_path_suffix(source)
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(_unsupported_input_format(source))

    if suffix == ".parquet":
        if _is_remote_http_source(source):
            yield from iter_parquet_rows([source], offset=offset, limit=limit, sql=sql)
        else:
            yield from _iter_hf_parquet_file_rows(
                source,
                filesystem,
                offset=offset,
                limit=limit,
                sql=sql,
            )
        return

    if sql is not None:
        raise ValueError("--sql is only supported for parquet input")

    yield from _iter_hf_text_rows(source, filesystem, suffix=suffix)


def _iter_hf_text_rows(
    source: str,
    filesystem: HfInputFileSystem,
    *,
    suffix: str,
) -> Iterable[RowCandidate]:
    try:
        with filesystem.open(source, "rb") as binary_handle:
            import io

            with io.TextIOWrapper(binary_handle, encoding="utf-8", newline="") as text_handle:
                if suffix == ".jsonl":
                    yield from iter_jsonl_stream(text_handle)
                else:
                    yield from iter_csv_stream(text_handle)
    except UnicodeDecodeError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read Hugging Face input {source}: {exc}") from exc


def iter_input_rows(
    source: str | Path,
    *,
    offset: int | None = None,
    limit: int | None = None,
    sql: str | None = None,
) -> Iterable[RowCandidate]:
    source_text = str(source)
    if urlparse(source_text).scheme == "hf":
        return iter_hf_rows(source_text, offset=offset, limit=limit, sql=sql)

    path = Path(source).expanduser()
    suffix = _path_suffix(path)
    if suffix == ".jsonl":
        if sql is not None:
            raise ValueError("--sql is only supported for parquet input")
        return iter_jsonl_rows(path)
    if suffix == ".csv":
        if sql is not None:
            raise ValueError("--sql is only supported for parquet input")
        return iter_csv_rows(path)
    if suffix == ".parquet":
        return iter_parquet_rows([str(path)], offset=offset, limit=limit, sql=sql)
    if sql is not None:
        raise ValueError("--sql is only supported for parquet input")
    raise ValueError(_unsupported_input_format(source_text))


def is_parquet_input_source(source: str | Path) -> bool:
    source_text = str(source)
    parsed = urlparse(source_text)
    if parsed.scheme == "hf":
        dataset_source = _parse_hf_dataset_source(parsed, source_text)
        if dataset_source is None:
            return _is_parquet_suffix(parsed.path)
        return dataset_source.suffix == ".parquet" or dataset_source.is_repository
    return _is_parquet_suffix(source_text)


def iter_parquet_rows(
    sources: list[str],
    *,
    offset: int | None = None,
    limit: int | None = None,
    sql: str | None = None,
) -> Iterable[RowCandidate]:
    """Yield rows from one or more parquet files using optional DuckDB support."""
    start = 1 if sql is not None else 1 + (offset or 0)
    for row_number, row in enumerate(
        _read_parquet_records(sources, offset=offset, limit=limit, sql=sql),
        start=start,
    ):
        yield RowCandidate(row_number=row_number, row=_json_safe_row(row))


def count_parquet_input_rows(source: str | Path) -> int:
    source_text = str(source)
    if urlparse(source_text).scheme == "hf":
        fs = _default_hf_filesystem()
        resolved = _resolve_hf_input_source(source_text, fs)
        parquet_sources = _parquet_sources_from_token(resolved)
        if parquet_sources is not None:
            return _count_parquet_records(parquet_sources)
        if _is_parquet_suffix(urlparse(resolved).path):
            if _is_remote_http_source(resolved):
                return _count_parquet_records([resolved])
            with tempfile.NamedTemporaryFile(suffix=".parquet", prefix="fast-agent-batch-") as temp_file:
                with fs.open(resolved, "rb") as source_handle:
                    shutil.copyfileobj(source_handle, temp_file)
                temp_file.flush()
                return _count_parquet_records([temp_file.name])
    return _count_parquet_records([str(Path(source_text).expanduser())])


def _unsupported_input_format(source: str) -> str:
    return (
        f"Unsupported input format for {source}; expected .jsonl, .csv, or .parquet. "
        "For Hugging Face dataset repositories, use hf://datasets/owner/name or point "
        "--input at a JSONL/CSV/parquet file in the dataset repository."
    )


def _resolve_hf_input_source(source: str, filesystem: HfInputFileSystem) -> str:
    parsed = urlparse(source)
    dataset_source = _parse_hf_dataset_source(parsed, source)
    if dataset_source is None or not dataset_source.is_repository:
        if dataset_source is not None and dataset_source.has_query:
            _parse_hf_dataset_input(source)
        return source
    if dataset_source.has_query:
        parquet_urls = _list_hf_dataset_parquet_urls(source)
        if parquet_urls:
            return parquet_urls[0] if len(parquet_urls) == 1 else _parquet_sources_token(parquet_urls)
        raise ValueError(f"Hugging Face dataset input {source} has no matching parquet files")

    try:
        paths = _hf_find_paths(filesystem.find(source))
    except Exception as exc:
        raise ValueError(f"Could not list Hugging Face dataset input {source}: {exc}") from exc

    supported = sorted(path for path in paths if _has_supported_input_suffix(path))
    if len(supported) == 1:
        return _as_hf_uri(supported[0])
    if len(supported) > 1:
        formatted = ", ".join(_as_hf_uri(path) for path in supported[:5])
        extra = "" if len(supported) <= 5 else f", and {len(supported) - 5} more"
        raise ValueError(
            f"Hugging Face dataset input {source} contains multiple JSONL/CSV/parquet files; "
            f"point --input at one file: {formatted}{extra}"
        )

    parquet_urls = _list_hf_dataset_parquet_urls(source)
    if parquet_urls:
        return parquet_urls[0] if len(parquet_urls) == 1 else _parquet_sources_token(parquet_urls)

    raise ValueError(f"Hugging Face dataset input {source} has no JSONL, CSV, or parquet files")


def _as_hf_uri(path: str) -> str:
    return path if path.startswith("hf://") else f"hf://{path}"


def _hf_find_paths(result: list[str] | dict[str, dict[str, Any]]) -> list[str]:
    if isinstance(result, dict):
        return list(result)
    return result


def _parse_hf_dataset_source(parsed: ParseResult, source: str) -> HfDatasetSource | None:
    if parsed.scheme != "hf" or parsed.netloc != "datasets":
        return None
    path_parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if not path_parts:
        raise ValueError(f"Expected a Hugging Face dataset URI, got {source}")
    return HfDatasetSource(
        is_repository=len(path_parts) <= 2,
        suffix=_path_suffix(parsed.path),
        has_query=bool(parsed.query),
    )


def _parquet_sources_token(urls: list[str]) -> str:
    return "parquet://" + json.dumps(urls, separators=(",", ":"))


def _parquet_sources_from_token(source: str) -> list[str] | None:
    if not source.startswith("parquet://"):
        return None
    payload = source.removeprefix("parquet://")
    try:
        urls = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid internal parquet source token") from exc
    if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
        raise ValueError("Invalid internal parquet source token")
    return urls


def _list_hf_dataset_parquet_urls(source: str) -> list[str]:
    dataset = _parse_hf_dataset_input(source)
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise ValueError("huggingface_hub is not available") from exc

    api = HfApi()
    try:
        entries = api.list_dataset_parquet_files(dataset.repo_id, config=dataset.config)
    except Exception as exc:
        raise ValueError(
            f"Could not list parquet files for Hugging Face dataset {dataset.repo_id}: {exc}"
        ) from exc

    urls: list[str] = []
    for entry in entries:
        if dataset.split is not None and entry.split != dataset.split:
            continue
        urls.append(entry.url)
    return urls


def _parse_hf_dataset_input(source: str) -> ParsedHfDatasetInput:
    parsed = urlparse(source)
    if parsed.scheme != "hf" or parsed.netloc != "datasets":
        raise ValueError(f"Expected a Hugging Face dataset URI, got {source}")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if not parts:
        raise ValueError(f"Expected a Hugging Face dataset URI, got {source}")
    if len(parts) > 2:
        raise ValueError(
            "Hugging Face dataset repository URIs must be hf://datasets/name or "
            f"hf://datasets/owner/name; point at a file for nested paths ({source})"
        )
    repo_id = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
    query = parse_qs(parsed.query, keep_blank_values=True)
    unknown_keys = sorted(set(query) - HF_DATASET_QUERY_KEYS)
    if unknown_keys:
        formatted = ", ".join(unknown_keys)
        raise ValueError(
            f"Unsupported Hugging Face dataset query parameter for {source}: {formatted}. "
            "Supported query parameters are config and split."
        )
    return ParsedHfDatasetInput(
        repo_id=repo_id,
        config=_single_query_value(query, "config"),
        split=_single_query_value(query, "split"),
    )


def _single_query_value(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    if values is None or not values:
        return None
    if len(values) > 1:
        raise ValueError(f"Expected at most one {name}= query parameter")
    if values[0] == "":
        raise ValueError(f"Expected non-empty {name}= query parameter")
    return values[0]


def _iter_hf_parquet_file_rows(
    source: str,
    filesystem: HfInputFileSystem,
    *,
    offset: int | None = None,
    limit: int | None = None,
    sql: str | None = None,
) -> Iterable[RowCandidate]:
    with tempfile.NamedTemporaryFile(suffix=".parquet", prefix="fast-agent-batch-") as temp_file:
        with filesystem.open(source, "rb") as source_handle:
            shutil.copyfileobj(source_handle, temp_file)
        temp_file.flush()
        yield from iter_parquet_rows([temp_file.name], offset=offset, limit=limit, sql=sql)


def _read_parquet_records(
    sources: list[str],
    *,
    offset: int | None = None,
    limit: int | None = None,
    sql: str | None = None,
) -> list[dict[str, Any]]:
    if not sources:
        return []
    try:
        return _read_parquet_records_with_python_duckdb(
            sources,
            offset=offset,
            limit=limit,
            sql=sql,
        )
    except ImportError:
        return _read_parquet_records_with_duckdb_cli(sources, offset=offset, limit=limit, sql=sql)


def _count_parquet_records(sources: list[str]) -> int:
    if not sources:
        return 0
    try:
        return _count_parquet_records_with_python_duckdb(sources)
    except ImportError:
        return _count_parquet_records_with_duckdb_cli(sources)


def _read_parquet_records_with_python_duckdb(
    sources: list[str],
    *,
    offset: int | None = None,
    limit: int | None = None,
    sql: str | None = None,
) -> list[dict[str, Any]]:
    try:
        duckdb = importlib.import_module("duckdb")
    except ImportError:
        raise

    connection = duckdb.connect()
    try:
        for statement in _duckdb_secret_statements():
            connection.execute(statement)
        if sql is not None:
            connection.execute(_parquet_view_query(sources))
            relation = connection.sql(_normalize_user_sql(sql))
        else:
            relation = connection.sql(_parquet_query(sources, offset=offset, limit=limit))
        columns = tuple(column[0] for column in relation.description)
        rows = tuple(tuple(row) for row in relation.fetchall())
        return [dict(zip(columns, row)) for row in rows]
    finally:
        connection.close()


def _count_parquet_records_with_python_duckdb(sources: list[str]) -> int:
    try:
        duckdb = importlib.import_module("duckdb")
    except ImportError:
        raise

    connection = duckdb.connect()
    try:
        for statement in _duckdb_secret_statements():
            connection.execute(statement)
        row = connection.sql(_parquet_count_query(sources)).fetchone()
        if row is None:
            return 0
        return int(row[0])
    finally:
        connection.close()


def _read_parquet_records_with_duckdb_cli(
    sources: list[str],
    *,
    offset: int | None = None,
    limit: int | None = None,
    sql: str | None = None,
) -> list[dict[str, Any]]:
    duckdb_binary = shutil.which("duckdb")
    if duckdb_binary is None:
        raise ValueError(
            "Parquet input requires DuckDB. Install the `duckdb` Python package, "
            "install the DuckDB CLI, or install fast-agent-mcp[batch-parquet]."
        )
    if sql is not None:
        rows = _run_duckdb_cli_json(_normalize_user_sql(sql), setup_queries=[_parquet_view_query(sources)])
    else:
        rows = _run_duckdb_cli_json(_parquet_query(sources, offset=offset, limit=limit))
    return rows


def _run_duckdb_cli_json(query: str, *, setup_queries: list[str] | None = None) -> list[dict[str, Any]]:
    duckdb_binary = shutil.which("duckdb")
    if duckdb_binary is None:
        raise ValueError(
            "Parquet input requires DuckDB. Install the `duckdb` Python package, "
            "install the DuckDB CLI, or install fast-agent-mcp[batch-parquet]."
        )
    secret_statements = _duckdb_secret_statements()
    setup = (
        [f".output {os.devnull}", *(f"{statement};" for statement in secret_statements), ".output"]
        if secret_statements
        else []
    )
    setup.extend(f"{setup_query};" for setup_query in setup_queries or [])
    result = subprocess.run(
        [duckdb_binary, "-json"],
        input="\n".join([*setup, query + ";"]),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "DuckDB CLI command failed"
        raise ValueError(f"Could not read parquet input with DuckDB: {message}")
    if not result.stdout.strip():
        return []
    rows = json.loads(result.stdout)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("DuckDB returned an unexpected parquet result shape")
    return rows


def _count_parquet_records_with_duckdb_cli(sources: list[str]) -> int:
    rows = _run_duckdb_cli_json(_parquet_count_query(sources))
    if len(rows) != 1:
        raise ValueError("DuckDB returned an unexpected parquet count result shape")
    value = rows[0].get("count")
    if not isinstance(value, int):
        raise ValueError("DuckDB returned an unexpected parquet count value")
    return value


def _parquet_query(
    sources: list[str],
    *,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    source_list = ", ".join(_sql_string_literal(source) for source in sources)
    query = f"SELECT * FROM read_parquet([{source_list}], union_by_name=true)"
    if limit is not None:
        query += f" LIMIT {limit}"
    if offset is not None and offset > 0:
        query += f" OFFSET {offset}"
    return query


def _parquet_view_query(sources: list[str]) -> str:
    source_list = ", ".join(_sql_string_literal(source) for source in sources)
    return f"CREATE OR REPLACE VIEW input AS SELECT * FROM read_parquet([{source_list}], union_by_name=true)"


def _normalize_user_sql(sql: str) -> str:
    query = sql.strip()
    semicolon_positions = _statement_semicolon_positions(query)
    if semicolon_positions:
        if len(semicolon_positions) > 1 or not _sql_tail_is_comment_or_whitespace(
            query, semicolon_positions[0] + 1
        ):
            raise ValueError("--sql must contain exactly one SELECT query")
        query = query[: semicolon_positions[0]].strip()
    first_token = strip_casefold(split_first_token(query, default_token="")[0] or "")
    if first_token == "with":
        first_token = _statement_token_after_with(query)
    if first_token != "select":
        raise ValueError("--sql must be a SELECT query")
    return query


def _statement_token_after_with(query: str) -> str:
    index = len("with")
    recursive_token, index = _read_sql_identifier(query, index)
    if recursive_token != "recursive":
        index -= len(recursive_token)

    depth = 0
    cte_body_pending = False
    cte_body_depth: int | None = None
    cte_body_closed = False

    while index < len(query):
        char = query[index]
        next_char = query[index + 1] if index + 1 < len(query) else ""

        if char.isspace():
            index += 1
            continue
        if char == "-" and next_char == "-":
            index = _skip_sql_line_comment(query, index)
            continue
        if char == "/" and next_char == "*":
            index = _skip_sql_block_comment(query, index)
            continue
        if char in {"'", '"'}:
            index = _skip_sql_quoted_string(query, index)
            continue
        if char == "(":
            depth += 1
            if depth == 1 and cte_body_pending:
                cte_body_depth = depth
                cte_body_pending = False
            index += 1
            continue
        if char == ")":
            if depth > 0:
                if cte_body_depth == depth:
                    cte_body_depth = None
                    cte_body_closed = True
                depth -= 1
            index += 1
            continue
        if depth > 0:
            index += 1
            continue
        if cte_body_closed and char == ",":
            cte_body_closed = False
            index += 1
            continue

        token, next_index = _read_sql_identifier(query, index)
        if not token:
            return ""
        if cte_body_closed:
            return token
        if token == "as":
            cte_body_pending = True
        index = next_index

    return ""


def _read_sql_identifier(query: str, start: int) -> tuple[str, int]:
    index = start
    while index < len(query) and query[index].isspace():
        index += 1
    token_start = index
    while index < len(query) and (query[index].isalnum() or query[index] == "_"):
        index += 1
    return strip_casefold(query[token_start:index]), index


def _skip_sql_quoted_string(query: str, start: int) -> int:
    quote = query[start]
    index = start + 1
    while index < len(query):
        if query[index] == quote:
            next_char = query[index + 1] if index + 1 < len(query) else ""
            if next_char == quote:
                index += 2
                continue
            return index + 1
        index += 1
    return index


def _statement_semicolon_positions(query: str) -> list[int]:
    positions: list[int] = []
    index = 0
    quote: str | None = None
    while index < len(query):
        char = query[index]
        next_char = query[index + 1] if index + 1 < len(query) else ""
        if quote is not None:
            if char == quote:
                if next_char == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "-" and next_char == "-":
            index = _skip_sql_line_comment(query, index)
            continue
        if char == "/" and next_char == "*":
            index = _skip_sql_block_comment(query, index)
            continue
        if char == ";":
            positions.append(index)
        index += 1
    return positions


def _sql_tail_is_comment_or_whitespace(query: str, start: int) -> bool:
    index = start
    while index < len(query):
        char = query[index]
        next_char = query[index + 1] if index + 1 < len(query) else ""
        if char.isspace():
            index += 1
            continue
        if char == "-" and next_char == "-":
            index = _skip_sql_line_comment(query, index)
            continue
        if char == "/" and next_char == "*":
            end = query.find("*/", index + 2)
            if end == -1:
                return False
            index = _skip_sql_block_comment(query, index)
            continue
        return False
    return True


def _parquet_count_query(sources: list[str]) -> str:
    source_list = ", ".join(_sql_string_literal(source) for source in sources)
    return f"SELECT count(*) AS count FROM read_parquet([{source_list}], union_by_name=true)"


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _skip_sql_line_comment(query: str, index: int) -> int:
    newline = query.find("\n", index + 2)
    return len(query) if newline == -1 else newline + 1


def _skip_sql_block_comment(query: str, index: int) -> int:
    end = query.find("*/", index + 2)
    return len(query) if end == -1 else end + 2


def _duckdb_secret_statements() -> list[str]:
    token = os.getenv("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub.utils import get_token

            token = get_token()
        except Exception:
            token = None
    if not token:
        return []
    escaped = token.replace("'", "''")
    return [
        "CREATE OR REPLACE SECRET hf_hub_token "
        f"(TYPE HTTP, BEARER_TOKEN '{escaped}', SCOPE 'https://huggingface.co')",
        f"CREATE OR REPLACE SECRET hf_token (TYPE HUGGINGFACE, TOKEN '{escaped}')",
    ]


def _json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe_value(value) for key, value in row.items()}


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        converted = value
    elif isinstance(value, Decimal):
        converted = str(value)
    elif isinstance(value, datetime | date | time):
        converted = value.isoformat()
    elif isinstance(value, bytes):
        converted = value.decode("utf-8", errors="replace")
    elif isinstance(value, list | tuple):
        converted = [_json_safe_value(item) for item in value]
    elif isinstance(value, dict):
        converted = {str(key): _json_safe_value(item) for key, item in value.items()}
    else:
        converted = str(value)
    return converted


def _default_hf_filesystem() -> HfInputFileSystem:
    try:
        from huggingface_hub import HfFileSystem
    except Exception as exc:
        raise ValueError("huggingface_hub is not available") from exc

    return HfFileSystem()


def select_rows(
    rows: Iterable[RowCandidate],
    *,
    offset: int | None = None,
    sample: int | None = None,
    seed: int | None = None,
    limit: int | None = None,
) -> list[RowCandidate]:
    """Apply offset, deterministic sample, input-order restoration, and limit."""
    candidates = list(rows)
    if offset is not None and offset > 0:
        candidates = candidates[offset:]

    if sample is not None and sample < len(candidates):
        rng = random.Random(0 if seed is None else seed)
        indexed = list(enumerate(candidates))
        sampled = rng.sample(indexed, sample)
        candidates = [candidate for _, candidate in sorted(sampled, key=lambda item: item[0])]

    if limit is not None:
        candidates = candidates[:limit]

    return candidates
