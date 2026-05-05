import json
import sys

from typer.testing import CliRunner

from fast_agent.cli.main import app


def test_batch_structured_direct_mode_with_passthrough(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "out.jsonl"
    schema_path = tmp_path / "schema.json"
    template_path = tmp_path / "row.md"

    input_path.write_text('{"id":"1","x":2}\n', encoding="utf-8")
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    template_path.write_text("{{row_json}}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "--no-update-check",
            "--env",
            str(env_dir),
            "batch",
            "structured",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--schema",
            str(schema_path),
            "--template",
            str(template_path),
            "--model",
            "passthrough",
            "--id-field",
            "id",
            "--include-input",
            "--no-final-summary",
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record == {
        "id": "1",
        "row_number": 1,
        "ok": True,
        "result": {"id": "1", "x": 2},
        "error": None,
        "input": {"id": "1", "x": 2},
    }


def test_batch_structured_accepts_pydantic_schema_model(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "out.jsonl"
    template_path = tmp_path / "row.md"
    schema_module = tmp_path / "batch_schemas.py"

    input_path.write_text('{"id":"1","x":2}\n', encoding="utf-8")
    template_path.write_text("{{row_json}}", encoding="utf-8")
    schema_module.write_text(
        "from pydantic import BaseModel\n\n"
        "class RowResult(BaseModel):\n"
        "    id: str\n"
        "    x: int\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    try:
        result = CliRunner().invoke(
            app,
            [
                "--no-update-check",
                "--env",
                str(env_dir),
                "batch",
                "structured",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--schema-model",
                "batch_schemas:RowResult",
                "--template",
                str(template_path),
                "--model",
                "passthrough",
                "--id-field",
                "id",
                "--no-final-summary",
            ],
        )
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("batch_schemas", None)

    assert result.exit_code == 0, result.output
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["ok"] is True
    assert record["result"] == {"id": "1", "x": 2}


def test_batch_structured_accepts_shell_runtime_flag(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "out.jsonl"
    schema_path = tmp_path / "schema.json"
    summary_path = tmp_path / "summary.json"
    template_path = tmp_path / "row.md"

    input_path.write_text('{"id":"1","x":2}\n', encoding="utf-8")
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    template_path.write_text("{{row_json}}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "--no-update-check",
            "--env",
            str(env_dir),
            "batch",
            "structured",
            "-x",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--schema",
            str(schema_path),
            "--template",
            str(template_path),
            "--model",
            "passthrough",
            "--summary-output",
            str(summary_path),
            "--no-final-summary",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["shell_runtime"] is True
