import json
from typing import cast

import pytest
from pydantic import BaseModel

from fast_agent.batch.input import RowCandidate
from fast_agent.batch.structured import (
    StructuredBatchOptions,
    _extract_timing,
    _extract_usage,
    _identity_for_candidate,
    _load_parallel_manifest,
    _merge_timing_key,
    _row_call,
    _summary_int,
    _validate_selected_identities,
    load_json_schema,
    load_pydantic_model,
    load_schema_source,
    normalize_structured_batch_options,
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


def test_id_field_keeps_integer_and_string_identities_distinct() -> None:
    int_identity = _identity_for_candidate(
        RowCandidate(row_number=1, row={"id": 1}),
        "id",
    )
    string_identity = _identity_for_candidate(
        RowCandidate(row_number=2, row={"id": "1"}),
        "id",
    )

    assert int_identity.value == 1
    assert string_identity.value == "1"


@pytest.mark.parametrize("value", [None, True, False, "", [], {}])
def test_id_field_rejects_ambiguous_identity_values(value: object) -> None:
    identity = _identity_for_candidate(
        RowCandidate(row_number=3, row={"id": value}),
        "id",
    )

    assert identity.value == 3
    assert identity.error is not None
    assert identity.error.type == "InvalidIdField"
    assert "ID field 'id'" in identity.error.message


def test_selected_identities_reject_duplicate_valid_id_values() -> None:
    selected = [
        RowCandidate(row_number=1, row={"id": "same"}),
        RowCandidate(row_number=2, row={"id": "same"}),
    ]

    with pytest.raises(ValueError, match="Duplicate id field 'id' value 'same' at rows 1 and 2"):
        _validate_selected_identities(selected, id_field="id")


def test_selected_identities_keep_integer_and_string_id_values_distinct() -> None:
    selected = [
        RowCandidate(row_number=1, row={"id": 1}),
        RowCandidate(row_number=2, row={"id": "1"}),
    ]

    _validate_selected_identities(selected, id_field="id")


def test_selected_identities_preflight_invalid_id_values() -> None:
    selected = [RowCandidate(row_number=4, row={"id": []})]

    with pytest.raises(ValueError, match="Row 4: ID field 'id'"):
        _validate_selected_identities(selected, id_field="id")


def test_parallel_timing_merge_does_not_claim_weighted_shard_median_is_global_median() -> None:
    merged = _merge_timing_key(
        [
            {
                "timing_ms": {
                    "duration": {
                        "count": 1,
                        "min": 1.0,
                        "mean": 1.0,
                        "median": 1.0,
                        "max": 1.0,
                    }
                }
            },
            {
                "timing_ms": {
                    "duration": {
                        "count": 99,
                        "min": 100.0,
                        "mean": 100.0,
                        "median": 100.0,
                        "max": 100.0,
                    }
                }
            },
        ],
        "duration",
    )

    assert "median" not in merged
    assert merged["median_approx"] == 99.01


def test_summary_int_rejects_bool_and_negative_counts() -> None:
    assert _summary_int({"processed_rows": 3}, "processed_rows") == 3
    assert _summary_int({"processed_rows": True}, "processed_rows") == 0
    assert _summary_int({"processed_rows": -1}, "processed_rows") == 0


def test_parallel_timing_merge_rejects_bool_count() -> None:
    merged = _merge_timing_key(
        [
            {
                "timing_ms": {
                    "duration": {
                        "count": True,
                        "min": 1.0,
                        "mean": 1.0,
                        "median": 1.0,
                        "max": 1.0,
                    }
                }
            }
        ],
        "duration",
    )

    assert merged == {"count": 0}


def _set_parallel_manifest_value(
    manifest: dict[str, object], field_path: tuple[str | int, ...], value: object
) -> None:
    if len(field_path) == 1:
        key = field_path[0]
        assert isinstance(key, str)
        manifest[key] = value
        return

    assert len(field_path) == 3
    assert field_path[0] == "shards"
    shard_index = field_path[1]
    shard_key = field_path[2]
    assert isinstance(shard_index, int)
    assert isinstance(shard_key, str)
    shards = manifest["shards"]
    assert isinstance(shards, list)
    shard = shards[shard_index]
    assert isinstance(shard, dict)
    cast("dict[str, object]", shard)[shard_key] = value


@pytest.mark.parametrize(
    ("field_path", "message"),
    [
        (("input_rows",), "input row count"),
        (("selected_rows",), "invalid selected_rows"),
        (("shards", 0, "index"), "invalid shard index"),
        (("shards", 0, "offset"), "invalid shard offset"),
        (("shards", 0, "limit"), "invalid shard limit"),
    ],
)
def test_parallel_manifest_rejects_bool_counts(
    tmp_path, field_path: tuple[str | int, ...], message: str
) -> None:
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text('{"id":"1"}\n', encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    manifest: dict[str, object] = {
        "input": str(input_path),
        "input_rows": 1,
        "selected_rows": 1,
        "shards": [
            {
                "index": 0,
                "offset": 0,
                "limit": 1,
                "output": "out.jsonl",
                "summary_output": "summary.json",
                "error_output": None,
                "telemetry_output": None,
            }
        ],
    }
    _set_parallel_manifest_value(manifest, field_path, True)
    (work_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    options = StructuredBatchOptions(
        input_path=input_path,
        output_path=tmp_path / "out.jsonl",
    )

    with pytest.raises(ValueError, match=message):
        _load_parallel_manifest(options, work_dir, input_rows=1)


@pytest.mark.asyncio
async def test_resume_and_overwrite_are_mutually_exclusive(tmp_path):
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text('{"id":"1"}\n', encoding="utf-8")
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")

    options = StructuredBatchOptions(
        input_path=input_path,
        output_path=tmp_path / "out.jsonl",
        schema_source=schema,
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
        schema_source=schema,
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


def test_normalize_structured_batch_options_strips_hf_dataset_values(tmp_path) -> None:
    options = StructuredBatchOptions(
        input_path=tmp_path / "rows.jsonl",
        output_path=tmp_path / "out.jsonl",
        export_traces_path=tmp_path / "traces",
        hf_dataset=" owner/dataset ",
        hf_dataset_path=" runs/ ",
    )

    normalized = normalize_structured_batch_options(options)

    assert normalized is not options
    assert normalized.hf_dataset == "owner/dataset"
    assert normalized.hf_dataset_path == "runs/"


def test_normalize_structured_batch_options_treats_blank_hf_values_as_missing(
    tmp_path,
) -> None:
    options = StructuredBatchOptions(
        input_path=tmp_path / "rows.jsonl",
        output_path=tmp_path / "out.jsonl",
        export_traces_path=tmp_path / "traces",
        hf_dataset=" ",
        hf_dataset_path="\t",
    )

    normalized = normalize_structured_batch_options(options)

    assert normalized.hf_dataset is None
    assert normalized.hf_dataset_path is None


def test_sql_requires_parquet_input(tmp_path):
    options = StructuredBatchOptions(
        input_path=tmp_path / "rows.jsonl",
        output_path=tmp_path / "out.jsonl",
        sql="SELECT * FROM input",
    )

    with pytest.raises(ValueError, match="only supported for parquet"):
        load_schema_source(options)


@pytest.mark.parametrize("field", ["limit", "offset", "sample"])
def test_sql_rejects_row_selection_options(tmp_path, field):
    options = StructuredBatchOptions(
        input_path=tmp_path / "rows.parquet",
        output_path=tmp_path / "out.jsonl",
        sql="SELECT * FROM input",
    )
    if field == "limit":
        options = StructuredBatchOptions(
            input_path=tmp_path / "rows.parquet",
            output_path=tmp_path / "out.jsonl",
            sql="SELECT * FROM input",
            limit=1,
        )
    elif field == "offset":
        options = StructuredBatchOptions(
            input_path=tmp_path / "rows.parquet",
            output_path=tmp_path / "out.jsonl",
            sql="SELECT * FROM input",
            offset=1,
        )
    else:
        options = StructuredBatchOptions(
            input_path=tmp_path / "rows.parquet",
            output_path=tmp_path / "out.jsonl",
            sql="SELECT * FROM input",
            sample=1,
        )

    with pytest.raises(ValueError, match="cannot be used with --limit, --offset, or --sample"):
        load_schema_source(options)


def test_sql_rejects_parallel(tmp_path):
    options = StructuredBatchOptions(
        input_path=tmp_path / "rows.parquet",
        output_path=tmp_path / "out.jsonl",
        sql="SELECT * FROM input",
        parallel=2,
    )

    with pytest.raises(ValueError, match="cannot be used with --parallel"):
        load_schema_source(options)


def test_load_pydantic_model_from_import_path():
    loaded = load_pydantic_model(f"{__name__}:ImportedResult")

    assert loaded is ImportedResult


def test_extracts_timing_and_usage_channels() -> None:
    response = PromptMessageExtended(
        role="assistant",
        content=[],
        channels={
            FAST_AGENT_TIMING: [text_content('{"duration_ms": 12.5}')],
            FAST_AGENT_USAGE: [
                text_content(
                    '{"turn": {"total_tokens": 42}, "summary": {"cumulative_input_tokens": 40}}'
                )
            ],
        },
    )

    assert _extract_timing(response) == {"duration_ms": 12.5}
    assert _extract_usage(response) == {
        "turn": {"total_tokens": 42},
        "summary": {"cumulative_input_tokens": 40},
    }


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
        schema_source=schema,
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
        schema_source=schema,
        error_output_path=error_output,
        telemetry_output_path=telemetry_link,
    )

    with pytest.raises(ValueError, match=r"--telemetry-output.*--error-output"):
        await run_structured_batch(options)
