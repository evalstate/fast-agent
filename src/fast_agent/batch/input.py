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
from typing import TYPE_CHECKING, Any, Iterable, Protocol
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from typing import BinaryIO, TextIO


SUPPORTED_INPUT_SUFFIXES = frozenset({".jsonl", ".csv", ".parquet"})


class HfInputFileSystem(Protocol):
    def open(self, path: str, mode: str = "rb") -> BinaryIO: ...
    def find(self, path: str) -> list[str] | dict[str, dict[str, Any]]: ...


@dataclass(frozen=True)
class RowError:
    type: str
    message: str


@dataclass(frozen=True)
class RowCandidate:
    row_number: int
    row: dict[str, Any] | None
    error: RowError | None = None


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
    for row_number, row in enumerate(reader, start=1):
        yield RowCandidate(row_number=row_number, row=dict(row))


def iter_csv_rows(path: Path) -> Iterable[RowCandidate]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from iter_csv_stream(handle)


def iter_hf_rows(
    source: str,
    *,
    filesystem: HfInputFileSystem | None = None,
    offset: int | None = None,
    limit: int | None = None,
) -> Iterable[RowCandidate]:
    """Yield input rows from a Hugging Face Hub file addressed by an hf:// URI."""
    fs = filesystem if filesystem is not None else _default_hf_filesystem()
    source = _resolve_hf_input_source(source, fs)
    parquet_sources = _parquet_sources_from_token(source)
    if parquet_sources is not None:
        yield from iter_parquet_rows(parquet_sources, offset=offset, limit=limit)
        return

    suffix = Path(urlparse(source).path).suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(_unsupported_input_format(source))

    if suffix == ".parquet":
        if source.startswith("http://") or source.startswith("https://"):
            yield from iter_parquet_rows([source], offset=offset, limit=limit)
        else:
            yield from _iter_hf_parquet_file_rows(source, fs, offset=offset, limit=limit)
        return

    try:
        with fs.open(source, "rb") as binary_handle:
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
) -> Iterable[RowCandidate]:
    source_text = str(source)
    if urlparse(source_text).scheme == "hf":
        return iter_hf_rows(source_text, offset=offset, limit=limit)

    path = Path(source).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return iter_jsonl_rows(path)
    if suffix == ".csv":
        return iter_csv_rows(path)
    if suffix == ".parquet":
        return iter_parquet_rows([str(path)], offset=offset, limit=limit)
    raise ValueError(_unsupported_input_format(source_text))


def is_parquet_input_source(source: str | Path) -> bool:
    source_text = str(source)
    parsed = urlparse(source_text)
    if parsed.scheme == "hf":
        suffix = Path(parsed.path).suffix.lower()
        return suffix == ".parquet" or (
            parsed.netloc == "datasets" and (suffix == "" or bool(parsed.query))
        )
    return Path(source_text).suffix.lower() == ".parquet"


def iter_parquet_rows(
    sources: list[str],
    *,
    offset: int | None = None,
    limit: int | None = None,
) -> Iterable[RowCandidate]:
    """Yield rows from one or more parquet files using optional DuckDB support."""
    start = 1 + (offset or 0)
    for row_number, row in enumerate(
        _read_parquet_records(sources, offset=offset, limit=limit),
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
        if Path(urlparse(resolved).path).suffix.lower() == ".parquet":
            if resolved.startswith("http://") or resolved.startswith("https://"):
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
    suffix = Path(parsed.path).suffix.lower()
    if suffix:
        return source
    if parsed.query:
        parquet_urls = _list_hf_dataset_parquet_urls(source)
        if parquet_urls:
            return parquet_urls[0] if len(parquet_urls) == 1 else _parquet_sources_token(parquet_urls)
        raise ValueError(f"Hugging Face dataset input {source} has no matching parquet files")

    try:
        paths = _hf_find_paths(filesystem.find(source))
    except Exception as exc:
        raise ValueError(f"Could not list Hugging Face dataset input {source}: {exc}") from exc

    supported = sorted(path for path in paths if Path(path).suffix.lower() in {".jsonl", ".csv"})
    if len(supported) == 1:
        return _as_hf_uri(supported[0])
    if len(supported) > 1:
        formatted = ", ".join(_as_hf_uri(path) for path in supported[:5])
        extra = "" if len(supported) <= 5 else f", and {len(supported) - 5} more"
        raise ValueError(
            f"Hugging Face dataset input {source} contains multiple JSONL/CSV files; "
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
    repo_id, config, split = _parse_hf_dataset_input(source)
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise ValueError("huggingface_hub is not available") from exc

    api = HfApi()
    try:
        entries = api.list_dataset_parquet_files(repo_id, config=config)
    except Exception as exc:
        raise ValueError(f"Could not list parquet files for Hugging Face dataset {repo_id}: {exc}") from exc

    urls: list[str] = []
    for entry in entries:
        if split is not None and entry.split != split:
            continue
        urls.append(entry.url)
    return urls


def _parse_hf_dataset_input(source: str) -> tuple[str, str | None, str | None]:
    parsed = urlparse(source)
    if parsed.scheme != "hf" or parsed.netloc != "datasets":
        raise ValueError(f"Expected a Hugging Face dataset URI, got {source}")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if not parts:
        raise ValueError(f"Expected a Hugging Face dataset URI, got {source}")
    repo_id = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
    query = parse_qs(parsed.query)
    return repo_id, _single_query_value(query, "config"), _single_query_value(query, "split")


def _single_query_value(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    if values is None or not values:
        return None
    if len(values) > 1:
        raise ValueError(f"Expected at most one {name}= query parameter")
    return values[0]


def _iter_hf_parquet_file_rows(
    source: str,
    filesystem: HfInputFileSystem,
    *,
    offset: int | None = None,
    limit: int | None = None,
) -> Iterable[RowCandidate]:
    with tempfile.NamedTemporaryFile(suffix=".parquet", prefix="fast-agent-batch-") as temp_file:
        with filesystem.open(source, "rb") as source_handle:
            shutil.copyfileobj(source_handle, temp_file)
        temp_file.flush()
        yield from iter_parquet_rows([temp_file.name], offset=offset, limit=limit)


def _read_parquet_records(
    sources: list[str],
    *,
    offset: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not sources:
        return []
    try:
        return _read_parquet_records_with_python_duckdb(sources, offset=offset, limit=limit)
    except ImportError:
        return _read_parquet_records_with_duckdb_cli(sources, offset=offset, limit=limit)


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
) -> list[dict[str, Any]]:
    try:
        duckdb = importlib.import_module("duckdb")
    except ImportError:
        raise

    connection = duckdb.connect()
    try:
        for statement in _duckdb_secret_statements():
            connection.execute(statement)
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
) -> list[dict[str, Any]]:
    duckdb_binary = shutil.which("duckdb")
    if duckdb_binary is None:
        raise ValueError(
            "Parquet input requires DuckDB. Install the `duckdb` Python package, "
            "install the DuckDB CLI, or install fast-agent-mcp[batch-parquet]."
        )
    rows = _run_duckdb_cli_json(_parquet_query(sources, offset=offset, limit=limit))
    return rows


def _run_duckdb_cli_json(query: str) -> list[dict[str, Any]]:
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


def _parquet_count_query(sources: list[str]) -> str:
    source_list = ", ".join(_sql_string_literal(source) for source in sources)
    return f"SELECT count(*) AS count FROM read_parquet([{source_list}], union_by_name=true)"


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list | tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    return str(value)


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

    if sample is not None:
        if sample < len(candidates):
            rng = random.Random(0 if seed is None else seed)
            indexed = list(enumerate(candidates))
            sampled = rng.sample(indexed, sample)
            candidates = [candidate for _, candidate in sorted(sampled, key=lambda item: item[0])]

    if limit is not None:
        candidates = candidates[:limit]

    return candidates
