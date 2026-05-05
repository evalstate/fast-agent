import json

from fast_agent.batch.input import iter_csv_rows, iter_jsonl_rows, select_rows


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

