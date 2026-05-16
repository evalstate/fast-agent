# Batch Processing

`fast-agent batch run` processes row-oriented inputs and writes one JSONL envelope per row.

## Inputs

Use `--input` with a local `.jsonl`, `.csv`, or `.parquet` file, or with an `hf://` URI for a Hugging Face dataset:

```bash
uv run fast-agent batch run \
  --input hf://datasets/evalstate/my-dataset/data/train.jsonl \
  --output out.jsonl \
  --model passthrough
```

Supported input formats:

| Source | Supported formats | Notes |
| --- | --- | --- |
| Local filesystem | `.jsonl`, `.csv`, `.parquet` | JSONL rows must be JSON objects. CSV rows are dictionaries keyed by header name. Parquet rows are read with DuckDB. |
| `hf://` Hugging Face dataset | `.jsonl`, `.csv`, parquet | Use `hf://datasets/owner/name` to read the dataset viewer parquet files, or point at a specific file such as `hf://datasets/owner/name/path/file.parquet`. If a repo has a single JSONL/CSV file, that file is used before the parquet fallback. |
| DuckDB | Python package or CLI | Parquet input requires either the `duckdb` Python package or a `duckdb` CLI on `PATH`. Install `fast-agent-mcp[batch-parquet]` to add the Python package. |

Dataset-level Hugging Face parquet inputs can be filtered by config and split:

```bash
uv run fast-agent batch run \
  --input 'hf://datasets/evalstate/my-dataset?config=default&split=train' \
  --output out.jsonl \
  --model passthrough
```

Each loaded row becomes the template context. Column names are available as template variables, and `{{row_json}}` renders the complete row:

```text
Classify this record:
{{row_json}}
```

For CSV input, all values are strings because they come from CSV fields. JSONL preserves the JSON value types. Parquet scalar values are normalized for JSON output and templates; dates/times become ISO strings, decimals become strings, and bytes are decoded as UTF-8 with replacement for invalid bytes.

## Hugging Face Output

`--hf-dataset` currently applies to exported trace artifacts, not result JSONL output. Use it with `--export-traces`:

```bash
uv run fast-agent batch run \
  --input rows.jsonl \
  --output out.jsonl \
  --export-traces traces/ \
  --hf-dataset owner/trace-dataset
```

Appending and de-duplicating result rows into a Hugging Face dataset is not implemented yet.
