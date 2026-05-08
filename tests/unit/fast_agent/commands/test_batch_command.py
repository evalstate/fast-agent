import json
import sys

from typer.testing import CliRunner

from fast_agent.cli.main import app


def test_batch_run_direct_mode_with_passthrough(tmp_path):
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
            "run",
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


def test_batch_run_without_schema_writes_text_result(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "out.jsonl"
    summary_path = tmp_path / "summary.json"
    template_path = tmp_path / "row.md"

    input_path.write_text('{"id":"1","x":2}\n', encoding="utf-8")
    template_path.write_text("Plain {{id}} {{x}}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "--no-update-check",
            "--env",
            str(env_dir),
            "batch",
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--template",
            str(template_path),
            "--model",
            "passthrough",
            "--id-field",
            "id",
            "--summary-output",
            str(summary_path),
            "--no-final-summary",
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["ok"] is True
    assert record["result"] == "Plain 1 2"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["output_mode"] == "text"


def test_batch_run_export_traces_writes_row_trace_and_manifest(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "out.jsonl"
    trace_dir = tmp_path / "traces"
    template_path = tmp_path / "row.md"

    input_path.write_text('{"id":"1","x":2}\n', encoding="utf-8")
    template_path.write_text("Trace {{id}} {{x}}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "--no-update-check",
            "--env",
            str(env_dir),
            "batch",
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--template",
            str(template_path),
            "--model",
            "passthrough",
            "--id-field",
            "id",
            "--export-traces",
            str(trace_dir),
            "--no-final-summary",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = [json.loads(line) for line in (trace_dir / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 1
    assert manifest[0]["id"] == "1"
    assert manifest[0]["ok"] is True
    trace_name = manifest[0]["trace"]
    assert isinstance(trace_name, str)
    trace_text = (trace_dir / trace_name).read_text(encoding="utf-8")
    assert "Trace 1 2" in trace_text


def test_batch_run_accepts_pydantic_schema_model(tmp_path):
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
                "run",
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


def test_batch_run_card_mode_with_passthrough(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "out.jsonl"
    schema_path = tmp_path / "schema.json"
    summary_path = tmp_path / "summary.json"
    template_path = tmp_path / "row.md"
    card_path = tmp_path / "extractor.md"

    input_path.write_text('{"id":"1","x":2}\n', encoding="utf-8")
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    template_path.write_text("{{row_json}}", encoding="utf-8")
    card_path.write_text(
        "---\nname: extractor\nmodel: passthrough\n---\n\nExtract row data.\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "--no-update-check",
            "--env",
            str(env_dir),
            "batch",
            "run",
            "--agent-card",
            str(card_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--schema",
            str(schema_path),
            "--template",
            str(template_path),
            "--summary-output",
            str(summary_path),
            "--no-final-summary",
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["ok"] is True
    assert record["result"] == {"id": "1", "x": 2}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["instruction"] is None
    assert summary["agent_card"] == str(card_path)
    assert summary["agent"] == "extractor"


def test_batch_run_rejects_instruction_with_agent_card(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "--no-update-check",
            "--env",
            str(env_dir),
            "batch",
            "run",
            "--agent-card",
            str(tmp_path / "agent.md"),
            "--instruction",
            str(tmp_path / "instruction.md"),
            "--input",
            str(tmp_path / "rows.jsonl"),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--schema",
            str(tmp_path / "schema.json"),
            "--no-final-summary",
        ],
    )

    assert result.exit_code != 0
    assert "--agent-card and --instruction cannot be used together" in result.output


def test_batch_run_rejects_agent_without_agent_card(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "--no-update-check",
            "--env",
            str(env_dir),
            "batch",
            "run",
            "--agent",
            "extractor",
            "--input",
            str(tmp_path / "rows.jsonl"),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--schema",
            str(tmp_path / "schema.json"),
            "--no-final-summary",
        ],
    )

    assert result.exit_code != 0
    assert "--agent requires --agent-card" in result.output


def test_batch_run_accepts_shell_runtime_flag(tmp_path):
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
            "run",
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


def test_batch_run_hf_dataset_requires_export_traces(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "--no-update-check",
            "--env",
            str(env_dir),
            "batch",
            "run",
            "--input",
            str(tmp_path / "rows.jsonl"),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--hf-dataset",
            "owner/dataset",
            "--no-final-summary",
        ],
    )

    assert result.exit_code != 0
    assert "--hf-dataset requires --export-traces" in result.output
