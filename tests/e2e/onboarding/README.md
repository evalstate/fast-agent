# Model onboarding conformance

Opt-in live tests of **one** catalogued model/alias through `FastAgent.run()`, not
a substitute endpoint or a direct SDK smoke test. Run from the repository root:

```bash
ONBOARD_MODEL=<alias> uv run pytest tests/e2e/onboarding -m e2e -q
ONBOARD_MODEL=<alias> ONBOARD_TRACK=lite \
  ONBOARD_EVIDENCE=tests/e2e/onboarding/.evidence/lite.json \
  uv run pytest tests/e2e/onboarding -m e2e -q
```

`ONBOARD_MODEL` is required: without it **every test skips**, before credentials,
workspace creation or inference. `ONBOARD_TRACK` defaults to `full`; other values
besides `full`/`lite` are rejected. Use `-rs` to see skip reasons. No model matrix,
provider fallback, xdist, or reruns: run serially and choose the exact route.

Set the chosen provider's normal environment credentials. For HF, without
printing the token:

```bash
HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
  ONBOARD_MODEL=hf.zai-org/GLM-5.3-Flash:novita ONBOARD_TRACK=lite \
  FAST_AGENT_RETRIES=0 uv run pytest tests/e2e/onboarding -m e2e -q
```

The default evidence path is `.evidence/<sanitised-model>.json` in this directory.
Use different `ONBOARD_EVIDENCE` paths to retain full/lite and repeated runs;
the same path is atomically replaced, not merged. Relative paths resolve against
the invocation directory. `.evidence/` is ignored. No evidence is overwritten
when the suite is disabled.

## Scope and cost

| Check | Evidence / limitation |
|---|---|
| `text-stream` | Nonempty text, normal completion, observed stream chunks. Not latency or throughput benchmarking. |
| `reasoning-efforts` | Catalog ladder through model queries, known sum, requested/configured/effective settings, outgoing reasoning fields, reasoning and usage counts. Full tests every named effort (or catalog budget presets/bounds); lite tests lowest/highest plus the unmodified selection's default. Continuous token budgets are not exhaustively enumerated. Reasoning at off/none/zero is recorded as a warning; the provider contract determines hard versus soft disable semantics. This does not establish effort quality or monotonic token growth. |
| `tool-loop` | Real deterministic Python function, two independent calls, and real shell `printf` in a temporary directory; execution and final values checked. Parallel issuance is informational, not required. |
| `multi-turn` | Three turns, first-turn memory plus middle-turn tool result used in the last sum. Full only; not a long-context stress test. |
| `structured` | Pydantic validation, alone and with a tool registered. Uses `structured_schema()` for the tool loop, following the existing live suite. `json_mode=None` skips; `no_tools` checks schema output with the tool registered but not called; other policies require execution. Object mode proves schema-valid JSON, not provider-side schema enforcement. |
| `media` | Existing `../multimodal/image.png` username read from pixels, and `sample.pdf` product/company. Positive expectations come from catalog `tokenizes`; unsupported PDF/image and an opaque unsupported MIME must produce a nonempty response without raising. A returned harness error is acceptable for a negative case and recorded. Not a claim of understanding unsupported media. |
| `usage` | Nonzero input/output totals and populated context percentage; caching/reasoning subsets recorded when exposed. Not a billing reconciliation. Shares the plain streamed turn to save cost. |
| `ptc/uptake` | Full only. Explicit PTC on, two function results combined in code, nested-call metadata and exposed Python names recorded. Maps to **stage `ptc`, check `uptake`**, not the other PTC stage checks. |
| `trace-review`, `interactive` | Always `manual`: machine extraction of reasoning fields is not human trace review or interactive surface validation. |

Models resolve through `ModelFactory.resolve_model_spec()` and its
`ModelDatabase` parameters. No provider-specific capability allowlists.
The requested model is preserved; direct-tool probes explicitly set `ptc=off`
even when the catalog default is on; uptake sets `ptc=on`.

Prompts are short. Each inference has a 2,048-token output allowance and tool
loops have at most five iterations. Explicit integer reasoning budgets raise
the output allowance to budget + 1,024, bounded by the catalog output maximum.
Full can therefore be costly on budget-based models: obtain authorization first.
These are smoke-test bounds, not tests of maximum output/context limits.
Truncation fails rather than silently raising the allowance and retrying.

There are no suite retries or mocks. The isolated YAML sets `llm_retries: 0`
(`FAST_AGENT_RETRIES` can override it). Existing provider SDK retries and
empty-completion fallback paths are **not** replaced: for example the OpenAI
SDK client constructor does not expose a retry setting through this harness.
This is a harness behavior, not an assertion that only one HTTP attempt occurs.

## Evidence

At session end the plugin writes:

```text
{
  model, resolved: {
    provider, model_name, context_window, max_output_tokens, tokenizes,
    json_mode, structured_tool_policy, reasoning_effort_spec, ptc_enabled
  },
  track, fast_agent_version, git_rev, started, finished,
  checks: [{
    id, stage, manifest_check_id, result, summary,
    tests: [{name, outcome, duration_s, details}]
  }],
  usage_total
}
```

`reasoning_effort_spec` includes kind, allowed values/budget bounds, default,
auto and disable support. `configured_reasoning` is the live LLM's property;
`effective_reasoning` includes the catalog default when that property is unset,
with its source recorded. **`wire_reasoning` is the actual request capture**, not
an inferred translation. Missing capture produces a `warn`, never invented wire
evidence; an empty captured object means no whitelisted reasoning fields were
present. Providers imported before trace enabling may not capture.

The suite enables the harness's existing `FAST_AGENT_LLM_TRACE` facility before
provider loading. Trace files live only in a system temporary workspace (set `TMPDIR` to choose
its parent directory) and are deleted on teardown. Only whitelisted reasoning fields enter JSON, never
credentials, request bodies, answers or long transcripts. Failure evidence
contains exception type/phase rather than possibly sensitive traceback text.
An interrupted/kill -9 process may leave a temporary workspace: remove it before
sharing artifacts. The suite does not retain traces for human review.

Pytest setup/call/teardown outcomes determine results; teardown failures count.
Aggregation is conservative: `fail > warn > skip > pass`. Thus one failed
subtest cannot be hidden by passes, and a mixed applicable/skipped check remains
`skip` with individual outcomes visible. Unselected/unrun tests say `Not run`;
a partial `-k` run is not a complete conformance result. `manual` is reserved for
the two human checks. Durations include fixture phases. Usage totals aggregate
each harness once (not once per check), retaining unknown subsets as null.
Provider attempts without reported usage cannot be costed from this file.

## PTC arithmetic fixture

The PTC uptake probe uses FastMCP's structured return for the integer lookup:
`{"result": int}` with a generated output schema. The supported PTC `returns`
metadata exposes that shape in the function stub (`dict[str, Any]`,
`Returns: {result: int}`). The prompt and final sum assertion are unchanged.
Other function-tool conformance checks retain the ordinary text-only decorator.

A plain `@fast.tool` function annotated `-> int` still returns scalar **text**
to PTC, behind an `Any` stub. Adding two such results concatenates strings unless
the model converts them. That is a separate product-boundary finding, not proof
that the model cannot call functions from code. Keep earlier observations when
comparing runs from before and after this fixture correction.

## Feeding the onboarding tracker

The stage reports, records and tracker live in the onboarding tracker in the
fast-agent-model-lab repository (its `onboarding/README.md`), not here. This
JSON is evidence, **not** a stage report or an automatic stage approval.

1. Keep/copy the evidence into the onboarding record's evidence directory in
   the model-lab checkout (an absolute `ONBOARD_EVIDENCE` path can write it
   there directly).
2. Write the conformance Markdown report described in the model-lab's
   `onboarding/README.md`.
   Cite its evidence path and invocation in the body; set `status: blocked` if
   any failure or required human review remains.
3. Copy each automated conformance check's `id`, `result`, and `summary` into
   the YAML front matter's `checks` list. Do not convert failures to warnings.
   The `ptc/uptake` entry belongs in the separate PTC report as `id: uptake`.
4. Complete `trace-review` and `interactive` manually and record their actual
   results, citing separate evidence. The report parser accepts
   `pass|warn|fail|skip`, **not `manual`**: do not blindly paste `manual` or call
   it a pass. Pending human work keeps the stage blocked; a skip needs a reason
   and the manifest/track's applicability rules.
5. Register the Markdown, not this JSON, from the model-lab checkout:

```bash
uv run onboard report <slug> conformance <path-to-conformance.md>
uv run onboard report <slug> ptc <path-to-ptc.md>
```

Uptake alone does not satisfy the PTC stage's evals, naming, benchmark or
recommendation checks. The supervisor still obtains the required independent
review.

## Validation finding (2026-11-14)

HF `zai-org/GLM-5.3-Flash:novita`, lite then full: all applicable automated
checks passed **except schema-with-tool** under the resolved `always` policy.
It returned JSON with an `answer` key but no schema-valid integer and no tool
execution. This remains a failure, not a suite workaround. Plain schema,
image comprehension, unsupported attachments, history replay and PTC uptake
passed; the model issued two independent direct calls in one turn. Full
reasoning requests sent low/high/max and the default sent max. The default LLM
reasoning property was unset, despite max appearing on the wire.

Evidence is in ignored `.evidence/glm53flash-novita-{lite,full}.json`; the
initial development run is also retained. These findings are route- and
date-specific, not promises about other backends or future versions.
