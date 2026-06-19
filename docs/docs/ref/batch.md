---
social:
  title: Batch Processing (Reference)
  tagline: Command Reference for processing Datasets with LLMs.
  description: Run repeatable batch jobs through fast-agent workflows.
  alt: fast-agent social card — Batch Processing
---

# Batch Processing (Reference)

`fast-agent batch run` processes row-oriented inputs and writes one JSONL envelope per row.

## Inputs

Use `--input` with a local `.jsonl`, `.csv`, or `.parquet` file, or with an `hf://` URI for a Hugging Face dataset:

```bash
uv run fast-agent batch run \
  --input hf://datasets/evalstate/my-dataset/data/train.jsonl \
  --output out.jsonl \
  --model passthrough
```

Use `--prompt` for a short inline row prompt template:

```bash
uv run fast-agent batch run \
  --input rows.jsonl \
  --prompt "Classify this {{product}} into A, B, or C." \
  --output out.jsonl \
  --model passthrough
```

`--prompt` and `--template` are mutually exclusive. Use `--template` when the
row prompt is easier to maintain in a file or URI. `--template`,
`--instruction`, and `--json-schema` accept local paths, HTTP(S) URLs, `file://`
URIs, and `hf://` URIs.

Supported input formats:

| Source                       | Supported formats            | Notes                                                                                                                                                                                                                                            |
| ---------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Local filesystem             | `.jsonl`, `.csv`, `.parquet` | JSONL rows must be JSON objects. CSV rows are dictionaries keyed by header name. Parquet rows are read with DuckDB.                                                                                                                              |
| `hf://` Hugging Face dataset | `.jsonl`, `.csv`, parquet    | Use `hf://datasets/owner/name` to read the dataset viewer parquet files, or point at a specific file such as `hf://datasets/owner/name/path/file.parquet`. If a repo has a single JSONL/CSV file, that file is used before the parquet fallback. |
| DuckDB                       | Python package or CLI        | Parquet input requires either the `duckdb` Python package or a `duckdb` CLI on `PATH`. Install `fast-agent-mcp[batch-parquet]` to add the Python package.                                                                                        |

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

### Parquet SQL selection

For parquet input, `--sql` can define the rows processed by the batch run. The query is a DuckDB `SELECT` query over a view named `input`:

```bash
uv run fast-agent batch run \
  --input rows.parquet \
  --output out.jsonl \
  --sql "SELECT id, text FROM input WHERE split = 'eval'"
```

`--sql` is intentionally limited to parquet input. It cannot be combined with `--limit`, `--offset`, `--sample`, or `--parallel`; put filtering, ordering, and limits directly in the SQL query.

When `--sql` is used, output `row_number` values are result ordinals from the SQL result set, not stable original parquet row positions. Prefer `--id-field` with a stable identifier column when using SQL selection, especially with `--resume`.

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

## Options

### Worker and prompting

| Option                     | Description                                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------------------- |
| `--model`, `-m`            | Model override for direct mode or the selected AgentCard worker.                                      |
| `--instruction PATH_OR_URI` | System instruction file or URI for direct mode. Mutually exclusive with `--agent-card`.               |
| `--agent-card PATH_OR_URI` | AgentCard file, directory, or URI defining the batch worker. Mutually exclusive with `--instruction`. |
| `--agent NAME`             | Agent name to run when `--agent-card` loads multiple runnable agents.                                 |
| `--prompt`, `-p TEXT`      | Inline row prompt template. Mutually exclusive with `--template`.                                     |
| `--template PATH_OR_URI`   | Row prompt template file or URI. Defaults to sending the full row JSON.                               |
| `--json-schema PATH_OR_URI` | JSON Schema file or URI for structured results. Supports local paths, HTTP(S), `file://`, and `hf://`. Mutually exclusive with `--schema-model`. |
| `--schema-model IMPORT`     | Pydantic `BaseModel` import path for structured results. Mutually exclusive with `--json-schema`.     |
| `--var NAME=VALUE`          | AgentCard template variable value. May be repeated.                                                   |
| `--var-file NAME=PATH`      | AgentCard template variable loaded from a file. May be repeated.                                      |
| `--vars-json PATH`          | JSON object containing AgentCard template variables.                                                  |
| `--shell`, `-x`            | Enable a local shell runtime and expose the execute tool.                                             |

### Input selection

| Option                      | Description                                                                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--input`, `-i PATH_OR_URI` | Required. Local `.jsonl`, `.csv`, `.parquet` path or `hf://` Hugging Face dataset URI.                                                   |
| `--limit N`                 | Maximum selected rows to process. Useful while developing prompts and templates.                                                         |
| `--offset N`                | Rows to skip before sampling.                                                                                                            |
| `--sample N`                | Deterministic sample size.                                                                                                               |
| `--seed N`                  | Deterministic sampling seed.                                                                                                             |
| `--sql QUERY`               | DuckDB `SELECT` query over parquet input view named `input`. Cannot be combined with `--limit`, `--offset`, `--sample`, or `--parallel`. |

### Output envelopes

| Option                                   | Description                                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| `--output`, `-o PATH`                    | Required. Output JSONL file.                                                          |
| `--include-input` / `--no-include-input` | Include the source row in each output envelope.                                       |
| `--id-field FIELD`                       | Input field used as the row ID. Prefer this for resumable production jobs.            |
| `--error-output PATH`                    | Additional JSONL file containing failed envelopes.                                    |
| `--telemetry-output PATH`                | JSONL file containing per-attempt normalized telemetry.                               |
| `--summary-output PATH`                  | Write final summary JSON to this path.                                                |
| `--final-summary` / `--no-final-summary` | Print the final summary JSON to stdout. Disable when another process consumes stdout. |

### Resume, overwrite, and failure limits

| Option           | Description                                                                          |
| ---------------- | ------------------------------------------------------------------------------------ |
| `--resume`       | Append missing or retried rows. Successful existing envelopes are skipped by row ID. |
| `--overwrite`    | Replace existing output. Mutually exclusive with `--resume`.                         |
| `--max-errors N` | Stop after this many row-level failures. Cannot be combined with `--parallel`.       |

### Parallel runs

| Option                           | Description                                                                                                |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `--parallel N`                   | Run `N` local workers and merge deterministic chunk outputs.                                               |
| `--work-dir PATH`                | Directory for parallel chunk outputs and resume manifests. Use a stable value for resumable parallel jobs. |
| `--keep-temp` / `--no-keep-temp` | Keep parallel chunk outputs after a successful merge.                                                      |
| `--progress-every N`             | Print progress every `N` processed rows per worker.                                                        |
| `--progress` / `--no-progress`   | Print batch progress messages to stderr.                                                                   |

`--parallel` cannot be combined with `--sql`, `--sample`, `--max-errors`, or
`--export-traces`.

### Trackio monitoring

Trackio is optional and explicit opt-in. Install `fast-agent-mcp[trackio]` or
`fast-agent-mcp[gepa]`, then pass `--project` or `--trackio-project`.

| Option                         | Description                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------- |
| `--project PROJECT`            | Enable Trackio monitoring and set the Trackio project. Alias: `--trackio-project`.          |
| `--run-name NAME`              | Trackio run name. Alias: `--trackio-name`.                                                  |
| `--group GROUP`                | Trackio group for repeated runs or phases. Alias: `--trackio-group`.                        |
| `--trackio-space-id SPACE`     | Optional Trackio / Hugging Face Space id.                                                   |
| `--trackio-server-url URL`     | Optional Trackio server URL.                                                                |
| `--trackio-every N`            | Log aggregate progress every `N` processed rows. Defaults to `--progress-every`, then `10`. |
| `--trackio-config-json PATH`   | Extra JSON object merged into `trackio.init(config=...)`.                                   |
| `--no-trackio`                 | Explicitly disable Trackio monitoring.                                                      |

Trackio logs aggregate `batch/` metrics only: progress, timing, usage, and
cache behavior. Row contents, prompts, full outputs, and full errors are not
logged by default.

### Trace export

| Option                   | Description                                                                                               |
| ------------------------ | --------------------------------------------------------------------------------------------------------- |
| `--export-traces PATH`   | Directory for per-row Codex trace JSONL files and `manifest.jsonl`. Cannot be combined with `--parallel`. |
| `--hf-dataset REPO`      | Upload exported traces to a Hugging Face dataset repository. Requires `--export-traces`.                  |
| `--hf-dataset-path PATH` | Path or prefix inside the Hugging Face dataset for exported traces. Requires `--hf-dataset`.              |
