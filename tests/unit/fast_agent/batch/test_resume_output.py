import json

import pytest

from fast_agent.batch.input import RowError
from fast_agent.batch.output import error_envelope, success_envelope
from fast_agent.batch.resume import load_completed_ids


def test_output_envelopes_are_stable():
    success = success_envelope(
        identity="001",
        row_number=1,
        result={"category": "billing"},
        row={"id": "001"},
        include_input=True,
    )
    failure = error_envelope(
        identity="002",
        row_number=2,
        error=RowError("Oops", "bad row"),
        row={"id": "002"},
        include_input=False,
    )

    assert success == {
        "id": "001",
        "row_number": 1,
        "ok": True,
        "result": {"category": "billing"},
        "error": None,
        "input": {"id": "001"},
    }
    assert failure == {
        "id": "002",
        "row_number": 2,
        "ok": False,
        "result": None,
        "error": {"type": "Oops", "message": "bad row"},
    }


def test_resume_loads_only_successful_ids_without_string_coercion(tmp_path):
    path = tmp_path / "out.jsonl"
    records = [
        {"id": 123, "ok": True},
        {"id": "456", "ok": False},
        {"id": "789", "ok": True},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    assert load_completed_ids(path) == {123, "789"}


@pytest.mark.parametrize("value", [None, True, False, "", [], {}])
def test_resume_rejects_ambiguous_existing_output_ids(tmp_path, value):
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps({"id": value, "ok": True}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Existing output id at line 1"):
        load_completed_ids(path)


def test_resume_fails_on_malformed_existing_output(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text('{"id": "1", "ok": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSONL"):
        load_completed_ids(path)
