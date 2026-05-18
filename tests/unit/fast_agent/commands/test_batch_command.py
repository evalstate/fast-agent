import asyncio
import json
import sys
from io import BytesIO

from typer.testing import CliRunner

from fast_agent.batch.structured import StructuredBatchOptions, run_parallel_structured_batch
from fast_agent.cli.main import app


class FakeHfFileSystem:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def open(self, path: str, mode: str = "rb") -> BytesIO:
        return BytesIO(self.files[path])


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


def test_batch_run_missing_input_reports_error_without_traceback(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "missing.jsonl"
    output_path = tmp_path / "out.jsonl"

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
            "--model",
            "passthrough",
            "--no-final-summary",
        ],
    )

    assert result.exit_code == 1
    assert f"Error: Input file not found: {input_path}" in result.output
    assert "Traceback" not in result.output
    assert not output_path.exists()


def test_batch_run_parallel_missing_parquet_input_reports_error_without_traceback(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "missing.parquet"
    output_path = tmp_path / "out.jsonl"

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
            "--model",
            "passthrough",
            "--parallel",
            "2",
            "--no-final-summary",
        ],
    )

    assert result.exit_code == 1
    assert f"Error: Input file not found: {input_path}" in result.output
    assert "Traceback" not in result.output
    assert not output_path.exists()


def test_batch_run_missing_template_reports_error_without_traceback(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "rows.jsonl"
    template_path = tmp_path / "missing.md"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text('{"id":"1","x":2}\n', encoding="utf-8")

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
            "--no-final-summary",
        ],
    )

    assert result.exit_code == 1
    assert f"Error: File not found: {template_path}" in result.output
    assert "Traceback" not in result.output
    assert not output_path.exists()


def test_batch_run_accepts_inline_prompt_template(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "out.jsonl"
    summary_path = tmp_path / "summary.json"
    input_path.write_text('{"id":"1","product":"battery"}\n', encoding="utf-8")

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
            "--prompt",
            "Classify this {{product}} into A, B, or C",
            "--output",
            str(output_path),
            "--model",
            "passthrough",
            "--include-input",
            "--summary-output",
            str(summary_path),
            "--no-final-summary",
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record == {
        "id": 1,
        "row_number": 1,
        "ok": True,
        "result": "Classify this battery into A, B, or C",
        "error": None,
        "input": {"id": "1", "product": "battery"},
    }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["input"] == str(input_path)
    assert summary["selected_rows"] == 1


def test_batch_run_rejects_prompt_and_template_together(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "rows.jsonl"
    template_path = tmp_path / "row.md"
    input_path.write_text('{"id":"1","x":2}\n', encoding="utf-8")
    template_path.write_text("{{x}}", encoding="utf-8")

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
            "--prompt",
            "prompt",
            "--template",
            str(template_path),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--model",
            "passthrough",
            "--no-final-summary",
        ],
    )

    assert result.exit_code == 2
    assert "--prompt and --template cannot be used together" in result.output


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


def test_batch_run_accepts_hf_input_uri(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    source = "hf://datasets/evalstate/example/data/train.jsonl"
    output_path = tmp_path / "out.jsonl"
    schema_path = tmp_path / "schema.json"
    template_path = tmp_path / "row.md"
    filesystem = FakeHfFileSystem({source: b'{"id":"1","x":2}\n'})
    monkeypatch.setattr("fast_agent.batch.input._default_hf_filesystem", lambda: filesystem)

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
            source,
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
            "--no-final-summary",
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["result"] == {"id": "1", "x": 2}


def test_batch_run_accepts_parquet_input(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "rows.parquet"
    output_path = tmp_path / "out.jsonl"
    schema_path = tmp_path / "schema.json"
    template_path = tmp_path / "row.md"

    input_path.write_bytes(b"not real parquet")
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    template_path.write_text("{{row_json}}", encoding="utf-8")
    monkeypatch.setattr(
        "fast_agent.batch.input._read_parquet_records",
        lambda sources, *, offset, limit, sql: [{"id": "1", "x": 2}],
    )

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
            "--no-final-summary",
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["result"] == {"id": "1", "x": 2}


def test_batch_run_parallel_merges_shard_outputs(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "out.jsonl"
    summary_path = tmp_path / "summary.json"
    work_dir = tmp_path / "work"
    template_path = tmp_path / "row.md"
    schema_path = tmp_path / "schema.json"

    input_path.write_text(
        "\n".join(json.dumps({"id": str(index), "x": index}) for index in range(4)) + "\n",
        encoding="utf-8",
    )
    template_path.write_text("{{row_json}}", encoding="utf-8")
    schema_path.write_text('{"type":"object"}', encoding="utf-8")

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
            "--schema",
            str(schema_path),
            "--model",
            "passthrough",
            "--id-field",
            "id",
            "--parallel",
            "2",
            "--work-dir",
            str(work_dir),
            "--summary-output",
            str(summary_path),
            "--keep-temp",
            "--no-progress",
            "--no-final-summary",
        ],
    )

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in records] == ["0", "1", "2", "3"]
    assert [record["result"]["x"] for record in records] == [0, 1, 2, 3]
    assert (work_dir / "part-000.jsonl").exists()
    assert (work_dir / "part-001.jsonl").exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["parallel"] == 2
    assert summary["selected_rows"] == 4
    assert summary["processed_rows"] == 4
    assert summary["failed_rows"] == 0


def test_batch_run_parallel_resume_uses_saved_shard_manifest(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "out.jsonl"
    summary_path = tmp_path / "summary.json"
    work_dir = tmp_path / "work"
    template_path = tmp_path / "row.md"
    schema_path = tmp_path / "schema.json"

    input_path.write_text(
        "\n".join(json.dumps({"id": str(index), "x": index}) for index in range(6)) + "\n",
        encoding="utf-8",
    )
    template_path.write_text("{{row_json}}", encoding="utf-8")
    schema_path.write_text('{"type":"object"}', encoding="utf-8")

    asyncio.run(
        run_parallel_structured_batch(
            StructuredBatchOptions(
                input_path=input_path,
                output_path=output_path,
                schema_path=schema_path,
                template_path=template_path,
                model="passthrough",
                limit=4,
                offset=1,
                id_field="id",
                summary_output_path=summary_path,
                final_summary=False,
                environment_dir=env_dir,
                parallel=2,
                work_dir=work_dir,
                keep_temp=True,
                progress=False,
            )
        )
    )
    output_path.unlink()

    asyncio.run(
        run_parallel_structured_batch(
            StructuredBatchOptions(
                input_path=input_path,
                output_path=output_path,
                schema_path=schema_path,
                template_path=template_path,
                model="passthrough",
                limit=6,
                offset=0,
                resume=True,
                id_field="id",
                summary_output_path=summary_path,
                final_summary=False,
                environment_dir=env_dir,
                parallel=4,
                work_dir=work_dir,
                keep_temp=True,
                progress=False,
            )
        )
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in records] == ["1", "2", "3", "4"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["parallel"] == 2
    assert summary["selected_rows"] == 4


def test_batch_run_parallel_rejects_final_output_inside_work_dir(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    input_path = tmp_path / "rows.jsonl"
    work_dir = tmp_path / "work"
    output_path = work_dir / "out.jsonl"
    template_path = tmp_path / "row.md"

    input_path.write_text('{"id":"1","x":2}\n{"id":"2","x":3}\n', encoding="utf-8")
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
            "--template",
            str(template_path),
            "--model",
            "passthrough",
            "--parallel",
            "2",
            "--work-dir",
            str(work_dir),
            "--no-final-summary",
        ],
    )

    assert result.exit_code != 0
    assert "--output must be outside --work-dir for parallel batches" in result.output
    assert not output_path.exists()


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


def test_batch_run_rejects_sql_with_limit(tmp_path):
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
            str(tmp_path / "rows.parquet"),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--sql",
            "SELECT * FROM input",
            "--limit",
            "1",
            "--no-final-summary",
        ],
    )

    assert result.exit_code != 0
    assert "--sql cannot be used with --limit, --offset, or --sample" in result.output


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
