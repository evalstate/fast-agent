import pytest
from pydantic import BaseModel

from fast_agent.batch.structured import (
    StructuredBatchOptions,
    _extract_timing,
    _extract_usage,
    _row_call,
    load_json_schema,
    load_pydantic_model,
    load_schema_source,
    run_structured_batch,
)
from fast_agent.constants import FAST_AGENT_TIMING, FAST_AGENT_USAGE
from fast_agent.llm.request_params import BatchRequestContext, RequestParams
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class ImportedResult(BaseModel):
    value: str


def test_schema_load_failure_is_preflight_error(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_json_schema(schema)


@pytest.mark.asyncio
async def test_resume_and_overwrite_are_mutually_exclusive(tmp_path):
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text('{"id":"1"}\n', encoding="utf-8")
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")

    options = StructuredBatchOptions(
        input_path=input_path,
        output_path=tmp_path / "out.jsonl",
        schema_path=schema,
        resume=True,
        overwrite=True,
    )

    with pytest.raises(ValueError, match="cannot be used together"):
        await run_structured_batch(options)


def test_schema_file_and_schema_model_are_mutually_exclusive(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")
    options = StructuredBatchOptions(
        input_path=tmp_path / "rows.jsonl",
        output_path=tmp_path / "out.jsonl",
        schema_path=schema,
        schema_model="example:Result",
    )

    with pytest.raises(ValueError, match="cannot be used together"):
        load_schema_source(options)


def test_schema_source_is_optional(tmp_path):
    options = StructuredBatchOptions(
        input_path=tmp_path / "rows.jsonl",
        output_path=tmp_path / "out.jsonl",
    )

    assert load_schema_source(options) is None


def test_load_pydantic_model_from_import_path():
    loaded = load_pydantic_model(f"{__name__}:ImportedResult")

    assert loaded is ImportedResult


def test_extracts_timing_and_usage_channels() -> None:
    response = PromptMessageExtended(
        role="assistant",
        content=[],
        channels={
            FAST_AGENT_TIMING: [text_content('{"duration_ms": 12.5}')],
            FAST_AGENT_USAGE: [text_content('{"summary": {"total_tokens": 42}}')],
        },
    )

    assert _extract_timing(response) == {"duration_ms": 12.5}
    assert _extract_usage(response) == {"summary": {"total_tokens": 42}}


@pytest.mark.asyncio
async def test_row_call_attaches_batch_context_to_request_params() -> None:
    class Worker:
        request_params: RequestParams | None = None

        async def generate(
            self,
            rendered: str,
            request_params: RequestParams,
        ) -> PromptMessageExtended:
            self.request_params = request_params
            return PromptMessageExtended(role="assistant", content=[text_content(rendered)])

    worker = Worker()
    parsed, _response = await _row_call(
        worker,
        rendered="hello",
        schema_source=None,
        batch_context=BatchRequestContext(row_number=7, identity="row-7"),
    )

    assert parsed == "hello"
    assert worker.request_params is not None
    assert worker.request_params.batch_context == BatchRequestContext(
        row_number=7,
        identity="row-7",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("duplicate_field", "expected_flag"),
    [
        ("error_output_path", "--error-output"),
        ("telemetry_output_path", "--telemetry-output"),
        ("summary_output_path", "--summary-output"),
    ],
)
async def test_optional_output_paths_cannot_match_primary_output(
    tmp_path,
    duplicate_field,
    expected_flag,
):
    input_path = tmp_path / "rows.jsonl"
    schema = tmp_path / "schema.json"
    output_path = tmp_path / "out.jsonl"

    options = StructuredBatchOptions(
        input_path=input_path,
        output_path=output_path,
        schema_path=schema,
        **{duplicate_field: output_path},
    )

    with pytest.raises(ValueError, match=rf"{expected_flag}.*--output"):
        await run_structured_batch(options)


@pytest.mark.asyncio
async def test_optional_output_paths_cannot_match_each_other_after_resolution(tmp_path):
    input_path = tmp_path / "rows.jsonl"
    schema = tmp_path / "schema.json"
    error_output = tmp_path / "errors.jsonl"
    telemetry_link = tmp_path / "telemetry.jsonl"
    error_output.touch()
    telemetry_link.symlink_to(error_output)

    options = StructuredBatchOptions(
        input_path=input_path,
        output_path=tmp_path / "out.jsonl",
        schema_path=schema,
        error_output_path=error_output,
        telemetry_output_path=telemetry_link,
    )

    with pytest.raises(ValueError, match=r"--telemetry-output.*--error-output"):
        await run_structured_batch(options)
