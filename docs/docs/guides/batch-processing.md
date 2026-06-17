---
title: Batch Processing
description: Run repeatable, resumable LLM workflows over datasets with fast-agent.
social:
  title: Batch Processing
  tagline: Efficient LLM processing of datasets.
  description: Efficient LLM processing of datasets.
  alt: fast-agent social card — Batch Processing
---

# Batch Processing

Use batch processing when you have a dataset of records and want to run the
same LLM workflow over each one: classify examples, extract fields, evaluate
outputs, enrich metadata, rewrite text, or run experiments at scale.

With **fast-agent**, each row is rendered into a prompt, combined with a stable
system instruction, sent to an Agent, and written as one
JSONL result envelope. Runs are built for practical iteration: stable inputs,
reusable prompts, efficient provider features, observable outputs, and resumable
retries when work is interrupted.

```bash
fast-agent batch run \
  --input rows.jsonl \
  --output results.jsonl \
  --limit 10 \
  --model "responses.gpt-5.4-mini?reasoning=low"
```

Use `--limit` to restrict the size of the run while developing 
your instruction and template, then remove it when you are ready
to run against the full input.

If you do not provide a template, fast-agent sends the whole input row as JSON:

```text
Input record:

{{row_json}}
```

That makes the first run simple: prepare row-oriented data, choose a model, and
start the batch.

## Why use fast-agent for batches?

Batch jobs benefit from the same runtime features as other **fast-agent**
sessions:

- **Parallel local workers**. Use `--parallel` to shard the selected input rows,
  run several workers concurrently, and merge the shard outputs into the final
  JSONL file.
- **Efficient provider execution**. Stable instructions, tools, schemas, and
  templates can benefit from provider prompt caching where supported; OpenAI
  Responses models can use `service_tier=flex` for cost-sensitive throughput;
  and Responses-family models can use WebSocket transport to reduce
  per-request connection overhead.
- **Resumable outputs**. Use `--resume` to append only rows whose successful
  result is not already present, so interrupted or partially failed jobs can be
  continued instead of started from scratch.
- **Operational visibility**. Optional progress, telemetry JSONL, error JSONL,
  and summary JSON make long-running jobs easier to monitor and audit.
- **Agent reuse**. The batch worker can be a direct model call, a custom system
  prompt, or a full AgentCard with tools and workflow configuration. Inspect agent
  behaviour interactively before committing to large runs.
- **Tool Routing**. **`fast-agent`** makes it simple to simulate or stub tool calls,
  or bypass LLM processing of results, making it ideal for GEPA optimization scenarios.

For example, a cost-sensitive OpenAI Responses batch can use the flex service
tier:

```bash
fast-agent batch run \
  --input rows.jsonl \
  --output results.jsonl \
  --model "responses.gpt-5.4-mini?service_tier=flex"
```

## 1. Source input data

Start with row-oriented data from a local `.jsonl`, `.csv`, or `.parquet` file,
or from a Hugging Face dataset URI. JSONL rows must be JSON objects:

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

## 2. Configure the system prompt

Use `--instruction` to provide the [system prompt](../agents/instructions.md)
for the batch worker:

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

AgentCards can also declare instruction variables for candidate-specific text,
such as a policy you are optimizing in a GEPA loop:

```markdown title="review-worker.md"
---
type: agent
name: reviewer
model: "$system.default"
variables:
  policy: ""
---

Classify reviews according to this policy:

{{policy}}
```

Populate those variables at run time:

```bash
fast-agent batch run \
  --input reviews.jsonl \
  --output review-results.jsonl \
  --agent-card ./review-worker.md \
  --agent reviewer \
  --var-file policy=seed/review-policy.md \
  --model "responses.gpt-5.5"
```

Use `--var NAME=VALUE` for short values, `--var-file NAME=PATH` for longer
prompt/policy files, or `--vars-json PATH` for a JSON object of variable
values.

## 3. Customise your Prompt with a template

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

## 4. Return structured results

For extraction, evaluation, or repeatable classification, add a JSON Schema or a
Pydantic model so outputs are machine-readable. `--json-schema` accepts a local
path, HTTP(S) URL, `file://` URI, or `hf://` URI.

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
  --json-schema sentiment.schema.json \
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
  --json-schema sentiment.schema.json \
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
  --json-schema sentiment.schema.json \
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
  --json-schema sentiment.schema.json \
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
processed/skipped/failed counts, model and input metadata, duration, timing
aggregates, and aggregate `usage` / `cache` blocks. The same summary is printed
to stdout by default; use `--no-final-summary` when another process is consuming
stdout.

Use these outputs together:

- `--output`: canonical per-row result envelopes.
- `--error-output`: optional copy of failed row envelopes for triage.
- `--telemetry-output`: per-attempt operational data for dashboards or cost and
  latency analysis.
- `--summary-output`: final run metadata for audit logs, CI artifacts, or
  regression comparisons.

## 8. Monitor expensive batches with Trackio

Trackio monitoring is explicit opt-in. Install `fast-agent-mcp[trackio]` or
`fast-agent-mcp[gepa]`, then pass a project:

```bash
fast-agent batch run \
  --input eval/input.jsonl \
  --output runs/labels/results.jsonl \
  --agent-card eval/teacher-card.md \
  --agent teacher \
  --template eval/template.md \
  --json-schema eval/schema.json \
  --model "codexresponses.gpt-5.5?reasoning=high" \
  --parallel 4 \
  --summary-output runs/labels/summary.json \
  --telemetry-output runs/labels/telemetry.jsonl \
  --project easy-v6-label-build \
  --run-name v6-gpt55-anchor-free-repeat-01 \
  --group v6-anchor-free-revalidation \
  --trackio-every 10
```

Batch Trackio logs aggregate progress, timing, token usage, and provider cache
behavior under `batch/` metric names. Charts use processed rows as the natural
step, so cumulative token counters and cache/rate percentages can be compared
across repeated runs.

By default fast-agent does **not** send row contents, rendered prompts, full
model outputs, or full errors to Trackio. Keep full-fidelity artifacts in local
`--output`, `--telemetry-output`, `--error-output`, and `--summary-output` files.

Batch Trackio is lower-level than GEPA Trackio: use it for row-processing health
and cache diagnostics; use GEPA Trackio for optimizer state, candidate scores,
frontier metrics, and proposals. They can be combined because batch metrics are
namespaced under `batch/`.

## 9. Use Hugging Face datasets as input

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
uv tool install 'fast-agent-mcp[batch-parquet]'
```

```bash
fast-agent batch run \
  --input rows.parquet \
  --output results.jsonl \
  --sql "SELECT id, text FROM input WHERE split = 'eval'" \
  --model "responses.gpt-5.5"
```

## Example 1: Zero Install Hugging Face Analysis

You can also run a small demo directly from a Hugging Face dataset repository.
This uses `uvx`, a JSONL input file, a row template, and an AgentCard stored in
the same dataset repo. The card connects the worker to the Hugging Face MCP
server so a model such as Kimi can answer with current Hugging Face context
rather than relying only on its training data.

```bash
uvx fast-agent-mcp@latest batch run \
  --input hf://datasets/evalstate/fast-agent-batch-demo/hf-research-questions.jsonl \
  --output hf-research-results.jsonl \
  --agent-card hf://datasets/evalstate/fast-agent-batch-demo/hf-research-agent.md \
  --template hf://datasets/evalstate/fast-agent-batch-demo/hf-research-template.md \
  --limit 3 \
  --id-field id \
  --progress-every 1 \
  --model kimi26instant
```

The dataset repo contains ordinary text artifacts:

```json title="hf-research-questions.jsonl"
{"id":"hf1","task":"Find a compact sentiment-analysis dataset suitable for a quick batch classification demo."}
{"id":"hf2","task":"Find a small text-generation model with recent activity and summarize why it is a useful baseline."}
{"id":"hf3","task":"Find a dataset for evaluating retrieval-augmented question answering and note the likely split to try first."}
```

```md title="hf-research-template.md"
Research this Hugging Face task:

{{task}}

Return a concise recommendation with the repository id, what you found, and one
practical next step.
```

The AgentCard declares the Hugging Face MCP server as a runtime connection:

```yaml title="hf-research-agent.md"
---
name: hf_researcher
description: Research Hugging Face models and datasets for batch rows.
mcp_connect:
  - target: "https://huggingface.co/mcp?login"
    name: huggingface
---
You are a concise Hugging Face research assistant.

Use the Hugging Face MCP server when you need current model or dataset
information. Prefer concrete repository ids and keep each answer short enough to
fit in a JSONL result record.
```

`hf://` sources work for inputs, AgentCards, prompt templates, instructions, and
JSON Schemas. Use JSONL or CSV for no-install demos; dataset-level parquet input
can require DuckDB.


## Example 2: Competitive Analysis with Web Search

This example asks the model to search the web for three competitive products
for each input product, then returns structured manufacturer and product names.

=== "bash"

    ```bash
    fast-agent batch run \
      --input products.jsonl \
      --output competitive-products.jsonl \
      --instruction instructions.md \
      --template template.md \
      --json-schema competitive-products.schema.json \
      --id-field id \
      --parallel 2 \
      --model "responses.gpt-5.4-mini?web_search=on&service_tier=flex"
    ```

=== "products.jsonl"

    ```json title="products.jsonl"
    {"id": "p1", "manufacturer": "Anker", "product": "PowerCore 10000 portable charger"}
    {"id": "p2", "manufacturer": "Sony", "product": "WH-1000XM5 headphones"}
    ```

=== "instructions.md"

    ```text title="instructions.md"
    You are a product research assistant.

    Use web search to identify competitive products.
    Return plain manufacturer and product names without citations or links.
    Return JSON that matches the supplied schema.
    ```

=== "template.md"

    ```text title="template.md"
    Find 3 competitive products for this product.

    Manufacturer: {{manufacturer}}
    Product: {{product}}

    For each competitor, capture the manufacturer and product name.
    ```

=== "competitive-products.schema.json"

    ```json title="competitive-products.schema.json"
    {
      "type": "object",
      "properties": {
        "competitors": {
          "type": "array",
          "minItems": 3,
          "maxItems": 3,
          "items": {
            "type": "object",
            "properties": {
              "manufacturer": {
                "type": "string"
              },
              "product": {
                "type": "string"
              }
            },
            "required": ["manufacturer", "product"],
            "additionalProperties": false
          }
        }
      },
      "required": ["competitors"],
      "additionalProperties": false
    }
    ```

The output is one JSONL envelope per input row:

```json title="competitive-products.jsonl"
{"id":"p1","row_number":1,"ok":true,"result":{"competitors":[{"manufacturer":"Belkin","product":"BoostCharge Power Bank 10K with Display"},{"manufacturer":"UGREEN","product":"Uno Power Bank 10000mAh 30W"},{"manufacturer":"Baseus","product":"Airpow Power Bank 20W 10000mAh"}]},"error":null}
{"id":"p2","row_number":2,"ok":true,"result":{"competitors":[{"manufacturer":"Bose","product":"QuietComfort Ultra Headphones"},{"manufacturer":"Apple","product":"AirPods Max"},{"manufacturer":"Sennheiser","product":"MOMENTUM 4 Wireless"}]},"error":null}
```

For the full option reference, see [Batch Processing](../ref/batch.md).
