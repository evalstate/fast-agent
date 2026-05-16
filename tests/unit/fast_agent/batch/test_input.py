import json
from datetime import date
from decimal import Decimal
from io import BytesIO

import pytest

from fast_agent.batch.input import (
    _parquet_query,
    iter_csv_rows,
    iter_hf_rows,
    iter_input_rows,
    iter_jsonl_rows,
    iter_parquet_rows,
    select_rows,
)


class FakeHfFileSystem:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.opened: list[tuple[str, str]] = []
        self.found: list[str] = list(files)

    def open(self, path: str, mode: str = "rb") -> BytesIO:
        self.opened.append((path, mode))
        return BytesIO(self.files[path])

    def find(self, path: str) -> list[str]:
        return self.found


def test_jsonl_rows_are_dicts(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "1", "message": "hello"}\n\n{"id": "2"}\n', encoding="utf-8")

    rows = list(iter_jsonl_rows(path))

    assert [row.row_number for row in rows] == [1, 3]
    assert rows[0].row == {"id": "1", "message": "hello"}
    assert rows[1].row == {"id": "2"}


def test_invalid_jsonl_lines_become_row_errors(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"ok": true}\nnot-json\n[]\n', encoding="utf-8")

    rows = list(iter_jsonl_rows(path))

    assert rows[1].error is not None
    assert rows[1].error.type == "InvalidJSON"
    assert rows[2].error is not None
    assert rows[2].error.type == "InvalidRow"


def test_csv_rows_are_dicts(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("id,message\n1,hello\n2,world\n", encoding="utf-8")

    rows = list(iter_csv_rows(path))

    assert [row.row for row in rows] == [
        {"id": "1", "message": "hello"},
        {"id": "2", "message": "world"},
    ]


def test_hf_jsonl_rows_are_read_from_dataset_file_uri():
    source = "hf://datasets/evalstate/example/data/train.jsonl"
    filesystem = FakeHfFileSystem(
        {source: b'{"id": "1", "message": "hello"}\n{"id": "2", "message": "world"}\n'}
    )

    rows = list(iter_hf_rows(source, filesystem=filesystem))

    assert filesystem.opened == [(source, "rb")]
    assert [row.row for row in rows] == [
        {"id": "1", "message": "hello"},
        {"id": "2", "message": "world"},
    ]


def test_hf_csv_rows_are_read_from_dataset_file_uri():
    source = "hf://datasets/evalstate/example/data/train.csv"
    filesystem = FakeHfFileSystem({source: b"id,message\n1,hello\n2,world\n"})

    rows = list(iter_hf_rows(source, filesystem=filesystem))

    assert [row.row for row in rows] == [
        {"id": "1", "message": "hello"},
        {"id": "2", "message": "world"},
    ]


def test_hf_dataset_uri_uses_single_supported_file():
    source = "hf://datasets/evalstate/example"
    file_source = "hf://datasets/evalstate/example/data/train.jsonl"
    filesystem = FakeHfFileSystem({file_source: b'{"id": "1"}\n'})

    rows = list(iter_hf_rows(source, filesystem=filesystem))

    assert filesystem.opened == [(file_source, "rb")]
    assert rows[0].row == {"id": "1"}


def test_hf_dataset_uri_with_many_supported_files_requires_explicit_file():
    source = "hf://datasets/evalstate/example"
    filesystem = FakeHfFileSystem(
        {
            "hf://datasets/evalstate/example/data/train.csv": b"id\n1\n",
            "hf://datasets/evalstate/example/data/test.csv": b"id\n2\n",
        }
    )

    with pytest.raises(ValueError, match="contains multiple JSONL/CSV files"):
        list(iter_hf_rows(source, filesystem=filesystem))


def test_parquet_rows_are_read_with_duckdb(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_read_parquet_records(
        sources: list[str],
        *,
        offset: int | None,
        limit: int | None,
    ) -> list[dict[str, object]]:
        captured["sources"] = sources
        captured["offset"] = [str(offset)]
        captured["limit"] = [str(limit)]
        return [{"id": "1", "score": 0.5}, {"id": "2", "score": 1}]

    monkeypatch.setattr("fast_agent.batch.input._read_parquet_records", fake_read_parquet_records)

    rows = list(iter_parquet_rows(["rows.parquet"]))

    assert captured["sources"] == ["rows.parquet"]
    assert captured["offset"] == ["None"]
    assert captured["limit"] == ["None"]
    assert [row.row for row in rows] == [{"id": "1", "score": 0.5}, {"id": "2", "score": 1}]


def test_parquet_rows_push_down_offset_and_limit(monkeypatch):
    captured: dict[str, int | None] = {}

    def fake_read_parquet_records(
        sources: list[str],
        *,
        offset: int | None,
        limit: int | None,
    ) -> list[dict[str, object]]:
        captured["offset"] = offset
        captured["limit"] = limit
        return [{"id": "3"}, {"id": "4"}]

    monkeypatch.setattr("fast_agent.batch.input._read_parquet_records", fake_read_parquet_records)

    rows = list(iter_parquet_rows(["rows.parquet"], offset=2, limit=2))

    assert captured == {"offset": 2, "limit": 2}
    assert [row.row_number for row in rows] == [3, 4]


def test_parquet_query_includes_offset_and_limit():
    assert _parquet_query(["rows.parquet"], offset=2, limit=5) == (
        "SELECT * FROM read_parquet(['rows.parquet'], union_by_name=true) LIMIT 5 OFFSET 2"
    )


def test_parquet_rows_are_json_safe_for_templates(monkeypatch):
    monkeypatch.setattr(
        "fast_agent.batch.input._read_parquet_records",
        lambda sources, *, offset, limit: [
            {"created": date(2026, 5, 16), "amount": Decimal("12.34")}
        ],
    )

    rows = list(iter_parquet_rows(["rows.parquet"]))

    assert rows[0].row == {"created": "2026-05-16", "amount": "12.34"}


def test_parquet_rows_require_duckdb_package_or_cli(monkeypatch):
    def fake_import_module(name: str) -> object:
        if name == "duckdb":
            raise ImportError("missing duckdb")
        raise AssertionError(name)

    monkeypatch.setattr("fast_agent.batch.input.importlib.import_module", fake_import_module)
    monkeypatch.setattr("fast_agent.batch.input.shutil.which", lambda name: None)

    with pytest.raises(ValueError, match="Parquet input requires DuckDB"):
        list(iter_parquet_rows(["rows.parquet"]))


def test_local_parquet_routes_to_duckdb_reader(monkeypatch, tmp_path):
    path = tmp_path / "rows.parquet"
    path.write_bytes(b"not real parquet")
    monkeypatch.setattr(
        "fast_agent.batch.input._read_parquet_records",
        lambda sources, *, offset, limit: [{"source": sources[0]}],
    )

    rows = list(iter_input_rows(path))

    assert rows[0].row == {"source": str(path)}


def test_hf_parquet_file_uri_materializes_and_reads_with_duckdb(monkeypatch):
    source = "hf://datasets/evalstate/example/data/train.parquet"
    filesystem = FakeHfFileSystem({source: b"parquet bytes"})
    captured: dict[str, list[str]] = {}

    def fake_read_parquet_records(
        sources: list[str],
        *,
        offset: int | None,
        limit: int | None,
    ) -> list[dict[str, object]]:
        captured["sources"] = sources
        captured["offset"] = [str(offset)]
        captured["limit"] = [str(limit)]
        return [{"id": "1"}]

    monkeypatch.setattr("fast_agent.batch.input._read_parquet_records", fake_read_parquet_records)

    rows = list(iter_hf_rows(source, filesystem=filesystem))

    assert filesystem.opened == [(source, "rb")]
    assert captured["sources"][0].endswith(".parquet")
    assert captured["offset"] == ["None"]
    assert captured["limit"] == ["None"]
    assert rows[0].row == {"id": "1"}


def test_hf_dataset_uri_uses_parquet_listing_when_no_jsonl_or_csv(monkeypatch):
    source = "hf://datasets/evalstate/example"
    filesystem = FakeHfFileSystem({})
    monkeypatch.setattr(
        "fast_agent.batch.input._list_hf_dataset_parquet_urls",
        lambda value: [
            "https://huggingface.co/datasets/evalstate/example/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "https://huggingface.co/datasets/evalstate/example/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet",
        ],
    )
    monkeypatch.setattr(
        "fast_agent.batch.input._read_parquet_records",
        lambda sources, *, offset, limit: [{"sources": sources, "offset": offset, "limit": limit}],
    )

    rows = list(iter_hf_rows(source, filesystem=filesystem))

    assert rows[0].row == {
        "sources": [
            "https://huggingface.co/datasets/evalstate/example/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "https://huggingface.co/datasets/evalstate/example/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet",
        ],
        "offset": None,
        "limit": None,
    }


def test_hf_dataset_uri_passes_config_and_split_to_parquet_listing(monkeypatch):
    captured: dict[str, str] = {}

    def fake_list_hf_dataset_parquet_urls(source: str) -> list[str]:
        captured["source"] = source
        return ["https://example.com/train.parquet"]

    monkeypatch.setattr(
        "fast_agent.batch.input._list_hf_dataset_parquet_urls",
        fake_list_hf_dataset_parquet_urls,
    )
    monkeypatch.setattr(
        "fast_agent.batch.input._read_parquet_records",
        lambda sources, *, offset, limit: [{"ok": True}],
    )

    rows = list(
        iter_hf_rows(
            "hf://datasets/evalstate/example?config=default&split=train",
            filesystem=FakeHfFileSystem({}),
        )
    )

    assert captured["source"] == "hf://datasets/evalstate/example?config=default&split=train"
    assert rows[0].row == {"ok": True}


def test_iter_input_rows_routes_hf_uri_to_hf_filesystem(monkeypatch):
    source = "hf://datasets/evalstate/example/data/train.jsonl"
    filesystem = FakeHfFileSystem({source: b'{"id": "1"}\n'})
    monkeypatch.setattr("fast_agent.batch.input._default_hf_filesystem", lambda: filesystem)

    rows = list(iter_input_rows(source))

    assert rows[0].row == {"id": "1"}


def test_selection_order_is_offset_sample_restore_order_then_limit(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": index}) for index in range(10)) + "\n",
        encoding="utf-8",
    )
    rows = list(iter_jsonl_rows(path))

    selected = select_rows(rows, offset=2, sample=5, seed=7, limit=2)

    full_sample = select_rows(rows, offset=2, sample=5, seed=7)
    assert selected == full_sample[:2]
    assert [row.row_number for row in selected] == sorted(row.row_number for row in selected)
