---
title: Batch Processing
description: A practical guide to running repeatable batch jobs with fast-agent.
social:
  title: Batch Processing
  tagline: Turn rows into prompts and run them efficiently through fast-agent.
  description: Turn rows into prompts and run them efficiently through fast-agent.
  alt: fast-agent social card — Batch Processing with fast-agent
---

# Batch Processing with fast-agent

Use batch processing when you need to conduct the same activity over many input
records: classification, extraction, evaluation, enrichment, rewriting, or any
other repeatable LLM workflow.

With **fast-agent**, each row is turned into a prompt, sent to a model or
AgentCard worker, and written as one JSONL result envelope. It is designed for
repeatable jobs where you want stable inputs, stable prompts, observable
outputs, and a clear way to retry incomplete work.

```bash
fast-agent batch run \
  --input rows.jsonl \
  --output results.jsonl \
  --model "responses.gpt-5.5"
```

If you do not provide a template, fast-agent sends the whole input row as JSON:

```text
Input record:

{{row_json}}
```

That makes the first run simple: prepare row-oriented data, choose a model, and
start the batch.

## Why use fast-agent for batches?

Batch jobs benefit from the same runtime features as interactive fast-agent
sessions:

- **Prompt caching** where providers support it. Repeated system prompts, tools,
  and template content can be reused efficiently across rows.
- **WebSocket transport** for supported providers. OpenAI Responses models can
  use `transport=ws` or `transport=auto` to reduce request overhead.
- **OpenAI service tiers**. For supported OpenAI API keys and models, use
  `service_tier=flex` when cost-sensitive throughput is more important than
  lowest latency.
- **Agent reuse**. The batch worker can be a direct model call, a custom system
  prompt, or a full AgentCard with tools and workflow configuration.
- **Parallel local workers**. Use `--parallel` to shard the selected input rows,
  run several workers concurrently, and merge the shard outputs into the final
  JSONL file.
- **Resumable outputs**. Use `--resume` to append only rows whose successful
  result is not already present, so interrupted or partially failed jobs can be
  continued instead of started from scratch.
- **Operational visibility**. Optional progress, telemetry JSONL, error JSONL,
  and summary JSON make long-running jobs easier to monitor and audit.

For example, a cost-sensitive OpenAI Responses batch can use:

```bash
fast-agent batch run \
  --input rows.jsonl \
  --output results.jsonl \
  --model "responses.gpt-5.5?service_tier=flex&transport=auto"
```

## 1. Create input data

Use a local `.jsonl`, `.csv`, or `.parquet` file. JSONL rows must be JSON
objects:

```json title="reviews.jsonl"
{"id": "r1", "review": "The battery lasts all day.", "product": "phone"}
{"id": "r2", "review": "Arrived late and the box was damaged.", "product": "speaker"}
```

Run a quick batch:

```bash
fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --model "responses.gpt-5.5"
```

Each output line is a JSON envelope for one input row. Add `--include-input` if
you want the source row copied into each result envelope.

```bash
fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --include-input \
  --id-field id \
  --model "responses.gpt-5.5"
```

With `--id-field id`, successful output records look like:

```json
{"id":"r1","row_number":1,"ok":true,"result":{"sentiment":"positive","reason":"The review praises battery life."},"error":null}
```

Failed rows are written to the main output as `ok: false` envelopes, and can
also be copied to a separate error file with `--error-output`.

## 2. Add a system prompt

Use `--instruction` to provide the system prompt for the batch worker:

```text title="sentiment-instructions.md"
You classify customer reviews.

Return a concise answer with:
- sentiment: positive, neutral, or negative
- reason: one short sentence
```

```bash
fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --instruction sentiment-instructions.md \
  --model "responses.gpt-5.5"
```

This instruction is stable across every row, which makes it a good candidate for
provider prompt caching when the selected provider supports caching.

For more complex workers, use an AgentCard instead:

```bash
fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --agent-card ./review-worker.md \
  --agent reviewer \
  --model "responses.gpt-5.5"
```

AgentCards are useful when the batch worker needs tools, MCP servers, skills, or
workflow definitions.

## 3. Shape each row with a template

Templates control the user prompt sent for each row.

```text title="review-template.md"
Classify this review.

Product: {{product}}
Review: {{review}}
```

```bash
fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --instruction sentiment-instructions.md \
  --template review-template.md \
  --model "responses.gpt-5.5"
```

For short templates, use `--prompt` instead of a template file:

```bash
fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --instruction sentiment-instructions.md \
  --prompt "Classify this {{product}} review into positive, neutral, or negative: {{review}}" \
  --model "responses.gpt-5.5"
```

`--prompt` and `--template` are mutually exclusive. Both use the same
placeholder syntax and both require an input source.

Template variables come from the top-level row fields. The full row is also
available as `{{row_json}}`:

```text title="record-template.md"
Analyze the complete source record:

{{row_json}}
```

You can mix specific fields with the full record:

```text
Review text:
{{review}}

Use the complete source record for context:
{{row_json}}
```

Template details:

- `{{field_name}}` inserts a top-level field from the row.
- `{{row_json}}` inserts the complete row as pretty-printed JSON.
- String values are inserted as-is.
- Non-string values are JSON encoded before insertion.
- Missing fields produce a row-level `MissingTemplateField` error.
- The syntax is simple placeholder replacement, not Jinja-style logic.

## 4. Ask for structured results

For extraction, evaluation, or repeatable classification, add a JSON Schema or a
Pydantic model so outputs are machine-readable.

```json title="sentiment.schema.json"
{
  "type": "object",
  "properties": {
    "sentiment": {
      "type": "string",
      "enum": ["positive", "neutral", "negative"]
    },
    "reason": {
      "type": "string"
    }
  },
  "required": ["sentiment", "reason"],
  "additionalProperties": false
}
```

```bash
fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --instruction sentiment-instructions.md \
  --template review-template.md \
  --schema sentiment.schema.json \
  --model "responses.gpt-5.5"
```

See [Structured Outputs](structured-outputs.md) for more schema options.

## 5. Parallelize the run

Use `--parallel` to run multiple local shard workers and merge the results:

```bash
fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --instruction sentiment-instructions.md \
  --template review-template.md \
  --schema sentiment.schema.json \
  --parallel 4 \
  --model "responses.gpt-5.5?transport=auto"
```

This is useful when the provider and account limits can support more concurrent
requests. Increase gradually and watch provider rate limits, latency, and cost.

When you want a parallel job to be resumable, provide a stable work directory
from the first run:

```bash
fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --parallel 4 \
  --work-dir .batch/reviews \
  --model "responses.gpt-5.5"
```

Notes:

- `--parallel` splits the selected rows into local shards.
- Shard outputs are merged into the final `--output` file.
- `--parallel` cannot be combined with `--sql`, `--sample`,
  `--max-errors`, or `--export-traces`.
- Use `--progress-every N` to print progress every `N` processed rows per
  worker.

## 6. Resume interrupted work

Use `--resume` when a run was interrupted or when you want to retry only the
rows that did not complete successfully:

```bash
fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --instruction sentiment-instructions.md \
  --template review-template.md \
  --schema sentiment.schema.json \
  --resume \
  --id-field id \
  --model "responses.gpt-5.5"
```

On resume, fast-agent reads the existing output file and builds a set of
completed row IDs from records where `ok` is `true`. Rows with completed IDs are
skipped. Missing rows, previous failures, and rows without a successful output
record are processed and appended.

ID semantics:

- The output envelope always has `id` and `row_number`.
- By default, `id` is the 1-based `row_number` from the loaded input stream.
- With `--id-field FIELD`, `id` is the string value of `FIELD` from each input
  row. Prefer this for resumable production jobs because it stays stable when
  input row order changes.
- `row_number` is useful for debugging and trace correlation, but it is not a
  durable business identifier unless your input ordering is immutable.
- If `--id-field` is set and a row is missing that field, the row is emitted as
  a `MissingIdField` error.

For parallel jobs, resumption is based on the shard work directory rather than
an existing final output file. Start the first run with a stable `--work-dir`,
then resume with the same directory:

```bash
fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --parallel 4 \
  --work-dir .batch/reviews \
  --resume \
  --id-field id \
  --model "responses.gpt-5.5"
```

Parallel resume validates that the input source and input row count match the
saved `manifest.json`. Move or remove any already-merged final output before
resuming a parallel run, or use `--overwrite` for the final merged output.

## 7. Capture telemetry and summaries

For long-running or repeated jobs, write machine-readable telemetry and a final
summary:

```bash
fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --instruction sentiment-instructions.md \
  --template review-template.md \
  --schema sentiment.schema.json \
  --telemetry-output review-telemetry.jsonl \
  --summary-output review-summary.json \
  --error-output review-errors.jsonl \
  --progress-every 100 \
  --model "responses.gpt-5.5"
```

Telemetry is JSONL with one record per attempted row. Each record includes the
row `id`, `row_number`, success flag, normalized timing values when available,
and usage information when the provider reports it:

```json
{"id":"r1","row_number":1,"ok":true,"timing":{"duration_ms":1260.4,"ttft_ms":210.2,"time_to_response_ms":1259.8},"usage":{"turn":{"input_tokens":120,"output_tokens":36}}}
```

Summary output is a JSON object describing the whole run: selected row counts,
processed/skipped/failed counts, model and input metadata, duration, and timing
aggregates for duration, time to first token, and time to response. The same
summary is printed to stdout by default; use `--no-final-summary` when another
process is consuming stdout.

Use these outputs together:

- `--output`: canonical per-row result envelopes.
- `--error-output`: optional copy of failed row envelopes for triage.
- `--telemetry-output`: per-attempt operational data for dashboards or cost and
  latency analysis.
- `--summary-output`: final run metadata for audit logs, CI artifacts, or
  regression comparisons.

## 8. Use Hugging Face datasets as input

Use `hf://` URIs with `--input` to read from Hugging Face datasets:

```bash
fast-agent batch run \
  --input 'hf://datasets/evalstate/my-dataset?config=default&split=train' \
  --output results.jsonl \
  --template record-template.md \
  --model "responses.gpt-5.5?service_tier=flex"
```

You can also point at a specific file in a dataset repository:

```bash
fast-agent batch run \
  --input hf://datasets/evalstate/my-dataset/data/train.jsonl \
  --output results.jsonl \
  --model "responses.gpt-5.5"
```

Supported local and Hugging Face input formats are `.jsonl`, `.csv`, and
`.parquet`. Parquet dataset inputs can be filtered by config and split, and
local parquet files can also use DuckDB SQL selection:

```bash
fast-agent batch run \
  --input rows.parquet \
  --output results.jsonl \
  --sql "SELECT id, text FROM input WHERE split = 'eval'" \
  --model "responses.gpt-5.5"
```

## A complete small example

```bash
cat > reviews.jsonl <<'EOF'
{"id": "r1", "review": "The battery lasts all day.", "product": "phone"}
{"id": "r2", "review": "Arrived late and the box was damaged.", "product": "speaker"}
EOF

cat > instructions.md <<'EOF'
You classify customer reviews.
Return JSON that matches the supplied schema.
EOF

cat > template.md <<'EOF'
Classify this customer review.

Product: {{product}}
Review: {{review}}
EOF

cat > sentiment.schema.json <<'EOF'
{
  "type": "object",
  "properties": {
    "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
    "reason": {"type": "string"}
  },
  "required": ["sentiment", "reason"],
  "additionalProperties": false
}
EOF

fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --instruction instructions.md \
  --template template.md \
  --schema sentiment.schema.json \
  --parallel 2 \
  --model "responses.gpt-5.5?service_tier=flex&transport=auto"
```

For the full option reference, see [Batch Processing](../ref/batch.md).
