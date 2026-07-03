import json

import pytest

from fast_agent import FastAgent
from fast_agent.batch import BatchRunner, extract_structured_output, extract_text_output


@pytest.mark.asyncio
async def test_batch_runner_returns_rows_and_artifact_paths(tmp_path):
    home = tmp_path / "env"
    home.mkdir()
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "out.jsonl"
    summary_path = tmp_path / "summary.json"
    input_path.write_text('{"id":"1","topic":"billing"}\n', encoding="utf-8")

    runner = BatchRunner(home=home)
    result = await runner.run(
        input=input_path,
        output_path=output_path,
        template="Topic: {{topic}}",
        model="passthrough",
        include_input=True,
        summary_path=summary_path,
        overwrite=True,
    )

    assert result.rows == [
        {
            "id": 1,
            "row_number": 1,
            "ok": True,
            "result": "Topic: billing",
            "error": None,
            "input": {"id": "1", "topic": "billing"},
        }
    ]
    assert result.summary["processed_rows"] == 1
    assert result.artifact_paths["output"] == output_path
    assert result.artifact_paths["summary"] == summary_path


@pytest.mark.asyncio
async def test_batch_runner_process_backend_matches_result_contract(tmp_path):
    home = tmp_path / "env"
    home.mkdir()
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text('{"id":"1","topic":"billing"}\n', encoding="utf-8")

    runner = BatchRunner(home=home, backend="process")
    result = await runner.run(
        input=input_path,
        output_path=output_path,
        template="Topic: {{topic}}",
        model="passthrough",
        overwrite=True,
    )

    assert result.rows[0]["result"] == "Topic: billing"
    assert result.summary["processed_rows"] == 1
    assert result.summary_path == output_path.with_suffix(".summary.json")


@pytest.mark.asyncio
async def test_agent_card_variables_are_run_scoped_prompt_context(tmp_path):
    home = tmp_path / "env"
    home.mkdir()
    card_path = tmp_path / "worker.md"
    card_path.write_text(
        "---\nname: worker\nmodel: passthrough\n---\n\nPolicy:\n{{policy}}\n",
        encoding="utf-8",
    )

    fast = FastAgent(
        "test",
        parse_cli_args=False,
        ignore_unknown_args=True,
        quiet=True,
        home=home,
    )
    fast.load_agents(str(card_path))
    fast.set_prompt_context({"policy": "Route support requests only."})

    async with fast.run() as app:
        worker = app._agent("worker")
        assert "Route support requests only." in worker.instruction
        assert "{{policy}}" not in worker.instruction


def test_output_extractors_accept_current_and_legacy_shapes():
    assert extract_structured_output({"result": {"label": "ok"}}) == {"label": "ok"}
    assert extract_structured_output({"result": {"parsed": {"label": "ok"}}}) == {"label": "ok"}
    assert extract_text_output({"result": "plain"}) == "plain"
    assert extract_text_output({"result": {"text": "nested"}}) == "nested"
    assert extract_text_output({"text": "legacy"}) == "legacy"


def test_batch_cli_var_json_shape_is_string_only(tmp_path):
    path = tmp_path / "vars.json"
    path.write_text(json.dumps({"policy": "strict"}), encoding="utf-8")

    from fast_agent.cli.commands.batch import _load_vars_json

    assert _load_vars_json(path) == {"policy": "strict"}
