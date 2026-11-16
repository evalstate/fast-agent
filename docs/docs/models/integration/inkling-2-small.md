# Inkling 2 Small — integration report

How fast-agent onboarded Inkling 2 Small: stage status, checks, caveats, and the reasoning-effort and structured-output findings. The full record and its evidence are in the fast-agent-model-lab repository.

## Model card

```yaml
name: Inkling 2 Small
vendor: Thinking Machines
id: thinkingmachines/Inkling-2-Small
route: hf.thinkingmachines/Inkling-2-Small:deepinfra
class: flagship
source: https://huggingface.co/thinkingmachines/Inkling-2-Small
```

Track: full; class: flagship

## Stages

| Stage | Status |
| --- | --- |
| Documentation and access | passed |
| Wire contract | passed |
| Catalog and provider integration | passed |
| Harness conformance | passed |
| Tool preference | passed |
| Programmatic tool calling | passed |
| Documentation and product surfaces | passed |
| Release readiness | passed |
| Release | pending |
| Benchmark | pending |

## Check matrix

| Stage | Check | Result | Summary |
| --- | --- | --- | --- |
| dossier | Primary documentation snapshotted with URLs and fetch dates | pass | Primary snapshots retained with URLs, fetch dates and SHA-256 hashes |
| dossier | Canonical model id, wire id casing and route | pass | thinkingmachines/Inkling-2-Small pinned to HF :together |
| dossier | Documented context window and output limit | warn | 1,048,576 context documented; separate output maximum unpublished |
| dossier | Documented input and output modalities | pass | Text and image inputs, text output; no native audio or PDF contract |
| dossier | Documented reasoning ladder and default | pass | Seven template names and numeric 0–0.99; documented default high |
| dossier | Documented tools, structured output, caching, sampling | warn | Tools, vision and schema advertised; exact routed combinations require contract probes |
| dossier | Pricing and rate limits | warn | $0.30/M input, $1/M output, $0.03/M cached input at Together; routed discounts and numeric rate limits unconfirmed |
| dossier | Conflicts between sources listed | warn | Together missing from router listing despite live Hub mapping and successful pinned access |
| dossier | Authenticated request to the exact model on the exact route | pass | User-supplied pinned HTTP 200 with x-inference-provider together; independently corroborated in contract evidence |
| contract | Each effort accepted, with observable effect; default and off behaviour | warn | 66 valid-setting runs correct; adjacent efforts overlap, none remains soft, historical default outlier reproduces at 4x allowance |
| contract | Reasoning field, streaming shape and replay requirement | pass | reasoning_content identified; shared replay succeeds with reasoning, reasoning_content and omission |
| contract | Default and maximum output tokens; truncation finish reason | warn | Omission generated 60014 tokens; both limit fields truncate at 64; separate output ceiling unknown |
| contract | Enforced context window and request size limits | warn | Input rejection confirms 1048576 window; recognized 4.6 MiB reaches Together, 5/6 MiB get router 413 |
| contract | Chunk shape, finish reasons, terminal usage | pass | Standard finishes and DONE; terminal usage with include_usage, absent without it |
| contract | Single, parallel and forced tool calls; streamed arguments | pass | Complete streamed get_weather matrix including forced none and real-call continuation; fragment shapes recorded |
| contract | JSON schema, JSON object, and schema with tools | warn | City schema validates via legacy envelope; standard schema rejected and legacy auto-tool cases skip the tool |
| contract | Each documented modality; unsupported types rejected cleanly | pass | Required image fixture passes inline, HTTPS and tool role; sample PDF, MP4 and audio cleanly rejected |
| contract | Accepted and rejected sampling parameters | pass | All nine parameters sent individually; ranges and recommendations recorded including null unknowns |
| contract | Usage fields and cached-token reporting | pass | Usage fields recorded; corrected roughly 5K prefix with different final lines reports 4864 cached tokens on repeat |
| contract | Proposed model parameters and route profile | pass | Evidence-to-field table complete; product recommendations provisional under D1-D4/D7-D8 |
| integration | Model parameters and catalog key | pass | INKLING_2_SMALL_HF under key thinkingmachines/inkling-2-small with contract values; 1,048,576 context, 128,000 nominal output (D1, provisional); mixed-case wire name mapped |
| integration | Provider or router profile | pass | Together rule: top-level effort default high, raw reasoning replay, legacy schema envelope, omitted default max_tokens. Deviation: reasoning=off maps to none (was silently high) |
| integration | Aliases and presets; superseded aliases pinned | pass | alias inkling -> hf.thinkingmachines/Inkling-2-Small:together with no sampling defaults (D5, provisional); nothing superseded or pinned |
| integration | Model selector entry; current flags | pass | HF picker entry "Inkling 2 Small (together)", current; no entry superseded; provider default unchanged |
| integration | Hard-coded model id sets and heuristics updated | pass | All known sites reviewed; none need the new id; mcp_agent name regexes do not match |
| integration | Offline wire-request contract tests | pass | 5 tests (10 cases) added; each fails without its production line; focused llm/picker/ptc run 3179 passed |
| integration | Format, lint, typecheck, unit tests | warn | lint, typecheck, cpd, unit (10673 passed) and diff-check pass; format --check fails only on an untouched file owned by other work (onboarding/notes/ptc-function-naming.md) |
| conformance | Streamed text turn | pass | 1 pass |
| conformance | Every ladder value through the alias; effective wire value recorded | warn | 7 pass, 1 warn |
| conformance | Tool call, execution, continuation, final answer | pass | 3 pass |
| conformance | History and reasoning replay across turns | pass | 1 pass |
| conformance | Structured output validated against the schema | pass | 2 pass |
| conformance | Each supported modality understood; unsupported types handled | pass | 3 pass |
| conformance | Usage and context accounting | pass | 1 pass |
| conformance | Raw request and stream traces inspected | pass | Requests/chunks inspected; route, reasoning order/replay, tool IDs/fragments, finishes and usage verified; raw traces deleted |
| conformance | Picker, effort cycling and check commands | pass | Picker, all seven efforts, off, invalid-value rejection and full F6 wrap verified; seven distinct renderer styles; 106 unit tests pass |
| preference | Shell, process and edit tool names and arguments used with none declared | warn | Reused Together panel; act/declare differ, 28/4 unparsed rows, native protocol syntax; invented-result count unavailable. |
| preference | First choice among declared equivalent tools | warn | Reused Together panel: 763/768 valid first choices; structural preferences and position bias; standalone forced-process panel absent. |
| preference | Background start, poll, wait and stop with the default contract | warn | 10/10 scratch tasks complete, no leaks; two retained-output errors preserved; Docker phase explicitly skipped. |
| preference | Edit tool reliability with the default contract | warn | 8/8 fixtures correct: five edit_file changes, three bash changes; write_text_file unused. |
| preference | Shell profile, shell name and edit tool recommendation | pass | Keep maintainer-selected common contract and PTC off; preference alone does not support a change. |
| preference | Resolved recommendation applied to the catalog; affected conformance checks rerun | pass | Read-only catalog confirms DeepInfra alias, unset contract overrides and PTC off; no code change or affected conformance delta. |
| ptc | PTC evals pass with and without PTC | pass | Real CLI: all 7 PTC answer/path checks and all 7 direct answer checks pass. |
| ptc | Functions are called from code; no direct-call confusion | warn | Corrected structured fixture 5/5 with nested calls; original scalar fixture 1/5 retained as product-boundary finding. Shell ledger 2/3; recovered unsupported imports. Supervisor-approved interpretation. |
| ptc | Canonical function names; no mangled or model-specific names | pass | Live traces and offline render agree: common bash in both modes; no mangled or model-specific names. |
| ptc | Matched direct and PTC benchmark probe | warn | Together direct 15/15 vs PTC 13/15; DeepInfra 15/15 each, sequential. Real nested uptake; no equivalence claim. |
| ptc | Sandbox-sensitive probe before PTC becomes the default | warn | Together PTC 35/80 as run, 14 unreplaced setup failures; direct 46/80 after 16 replacements. Original executed 37/64 vs 35/66; mteb-retrieve 6/8 vs 0/8. PTC remains off. |
| ptc | PTC default recommendation | pass | D11 resolved: DeepInfra default, PTC off; results do not justify enabling PTC. |
| ptc | Resolved PTC default applied to the catalog; affected conformance checks rerun | pass | Existing catalog False retained per D11; offline alias/default checks and 75 policy tests pass. Earlier live conformance remains historical. |
| surfaces | Provider page section with dated verification note | pass | New "Inkling 2 Small via Together" section on huggingface.md with alias, route, limits, modalities, reasoning ladder, structured stance, PTC off, caveats and dated verification note; provisional on D1-D5, D7 |
| surfaces | Generated references regenerated without unrelated drift | pass | docs.py generate changed only model_aliases_hf.md and models_reference.md; no generation warnings; no unrelated drift |
| surfaces | Provider default, setup template and examples | pass | HF default unchanged (no decision moves it); setup template names no HF current models; check models hf lists inkling; no predecessor to replace |
| surfaces | PTC default lists in docs and examples | skip | PTC stage still running; left for a follow-up run. Today inkling resolves PTC off and appears in no list; all copies match resolution for kimi/glm53flash/astra/glimmer except the pre-existing stale setup-template comment |
| surfaces | Strict docs build | pass | uv run scripts/docs.py build exit 0 (social cards + zensical build --strict) |
| review | Independent audit of the whole change; blockers resolved | pass | Independent GPT-family audit of Claude-authored Inkling change; zero open blockers; output-length wording corrected; one nonblocking generic test nit. |
| review | Final format, lint, typecheck and unit tests | pass | Final head format/lint/typecheck/CPD/resources green; unit 10709 passed, 9 skipped; integration 446 passed, 16 skipped; focused 547 passed. Diff check uses ignore-submodules for pre-existing broken submodule pointer. |
| review | Live smoke through the public alias | pass | Both public aliases: pinned upstream, one real bash call, exit 0, result/final marker and endTurn; four provider turns, no retries; reported raw usage cost $0.001711768. |
| review | No traces, credentials or unrelated changes | pass | No trace files, credential or signed-URL matches; sources ignored. Empty capture directory from offline unit tests disclosed. Worker changes only published report pages in fast-agent. |
| review | Integration report compiled and published to the docs site | pass | Compiled and published integration Markdown locally; strict docs build green, existing social cards present. PR drafted only; no push, PR opening or release. |
| release | Pull request raised with green checks | missing |  |
| release | Version published | missing |  |
| benchmark | Benchmark, route, reasoning, PTC mode and published version stated | missing |  |
| benchmark | Cohort complete | missing |  |
| benchmark | Results reviewed and published to the docs site | missing |  |

## Caveats

- dossier/limits: 1,048,576 context documented; separate output maximum unpublished
- dossier/capabilities: Tools, vision and schema advertised; exact routed combinations require contract probes
- dossier/pricing: $0.30/M input, $1/M output, $0.03/M cached input at Together; routed discounts and numeric rate limits unconfirmed
- dossier/discrepancies: Together missing from router listing despite live Hub mapping and successful pinned access
- contract/reasoning-ladder: 66 valid-setting runs correct; adjacent efforts overlap, none remains soft, historical default outlier reproduces at 4x allowance
- contract/output-limit: Omission generated 60014 tokens; both limit fields truncate at 64; separate output ceiling unknown
- contract/context-limit: Input rejection confirms 1048576 window; recognized 4.6 MiB reaches Together, 5/6 MiB get router 413
- contract/structured: City schema validates via legacy envelope; standard schema rejected and legacy auto-tool cases skip the tool
- integration/quality-gate: lint, typecheck, cpd, unit (10673 passed) and diff-check pass; format --check fails only on an untouched file owned by other work (onboarding/notes/ptc-function-naming.md)
- conformance/reasoning-efforts: 7 pass, 1 warn
- preference/unprompted-prior: Reused Together panel; act/declare differ, 28/4 unparsed rows, native protocol syntax; invented-result count unavailable.
- preference/matched-choice: Reused Together panel: 763/768 valid first choices; structural preferences and position bias; standalone forced-process panel absent.
- preference/lifecycle: 10/10 scratch tasks complete, no leaks; two retained-output errors preserved; Docker phase explicitly skipped.
- preference/edit-tool: 8/8 fixtures correct: five edit_file changes, three bash changes; write_text_file unused.
- ptc/uptake: Corrected structured fixture 5/5 with nested calls; original scalar fixture 1/5 retained as product-boundary finding. Shell ledger 2/3; recovered unsupported imports. Supervisor-approved interpretation.
- ptc/bench-probe: Together direct 15/15 vs PTC 13/15; DeepInfra 15/15 each, sequential. Real nested uptake; no equivalence claim.
- ptc/bench-sandbox: Together PTC 35/80 as run, 14 unreplaced setup failures; direct 46/80 after 16 replacements. Original executed 37/64 vs 35/66; mteb-retrieve 6/8 vs 0/8. PTC remains off.

## Not tested

- surfaces/ptc-lists: PTC stage still running; left for a follow-up run. Today inkling resolves PTC off and appears in no list; all copies match resolution for kimi/glm53flash/astra/glimmer except the pre-existing stale setup-template comment

## Reasoning effort sweep — Wire contract

**Evidence:** `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-reasoning-ladder.json`, [chat template](https://huggingface.co/thinkingmachines/Inkling-2-Small/raw/main/chat_template.jinja).

Prompt: `How many positive divisors does 7560 have? Reply with only the number.` Answer **64**, since 7560 = 2³·3³·5·7 and (3+1)(3+1)(1+1)(1+1)=64. All ladder bodies streamed with `include_usage: true`, `max_tokens: 32768`; no sampling override. Three repeats were used for every valid setting because adjacent distributions overlap. Every valid-setting run stopped and had final numeric answer 64. The shared summarizer's substring count is supplemented by checking the last standalone integer in the complete visible answer.

| Top-level setting | HTTP | Reasoning tokens (3 runs) | Correct |
|---|---|---|---|
| none | 200 | 257 / 193 / 238 | 3/3 |
| minimal | 200 | 137 / 149 / 175 | 3/3 |
| low | 200 | 203 / 165 / 187 | 3/3 |
| medium | 200 | 193 / 136 / 125 | 3/3 |
| high | 200 | 178 / 180 / 140 | 3/3 |
| xhigh | 200 | 143 / 226 / 168 | 3/3 |
| max | 200 | 183 / 151 / 177 | 3/3 |
| omitted | 200 | 164 / 255 / 154 | 3/3 |
| toggle-off | 200 | 170 / 174 / 145 | 3/3 |
"
The same seven names were swept through `chat_template_kwargs.reasoning_effort` (all 21 answers correct), along with top-level numeric 0, 0.1, 0.2, 0.7, 0.9, 0.99 (all 18 correct). **`bogus` was rejected with HTTP 400 twice through each field**, with an unknown-reasoning-effort template error. Earlier out-of-range numeric rejection evidence remains retained.

**Warn, not a monotonic reasoning-budget claim.** Effort distributions overlap, including documented xhigh/max aliases. `none`, template none and numeric zero all sometimes produce reasoning; even this short divisor prompt disproves hard-off semantics. `reasoning.enabled=false` likewise does not disable reasoning. D4 must describe none as a soft minimum, not “off.”

##### Default and the historical outlier

Omission answered the divisor prompt correctly 3/3. The template documents default high/0.9; observed token distributions cannot uniquely identify that default.

The original CRT prompt (known answer **863**), temperature 0, seed 17, with effort omitted, was rerun at **32768**, four times its prior 8192 allowance:

| Repeat | Completion / reasoning tokens | Finish | Visible answer |
|---|---|---|---|
| 1 | 32768 / 32766 | length | none |
| 2 | 525 / 423 | stop | correct, 863 |

This satisfies the required raised-allowance rerun and leaves an **intermittent default-effort looping caveat**, not a finding that omission never works. No further 131072-token escalation is needed to make that limited claim. D3's recommendation remains high because it is the documented default; **the new evidence does not prove explicit high eliminates looping**. Original non-streamed CRT correctness was not fully retained and is not counted as verified here.

## Reasoning effort sweep — Wire contract: deepinfra

**Evidence:** `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-deepinfra-reasoning-ladder.json`.

Divisor prompt (answer 64), streamed, `include_usage`, `max_tokens: 32768`.

| Setting (top-level `reasoning_effort` unless noted) | HTTP | Reasoning tokens | Final answer 64 | Working in visible content |
|---|---|---|---|---|
| none (streamed) | 200 | 0 ×7 | 7/7 | 7/7 |
| none (non-streamed) | 200 | 0 ×3 | 3/3 | 3/3 |
| minimal | 200 | 195 / 195 / 190 | 3/3 | 0/3 |
| low | 200 | 226 / 178 / 184 | 3/3 | 0/3 |
| medium | 200 | 217 / 180 / 222 | 3/3 | 0/3 |
| high | 200 | 169 / 172 / 175 | 3/3 | 0/3 |
| xhigh | 200 | 186 / 133 / 175 | 3/3 | 0/3 |
| max | 200 | 177 / 160 / 189 | 3/3 | 0/3 |
| omitted | 200 | 179 / 182 / 190 | 3/3 | 0/3 |
| `reasoning: {enabled: false}` | 200 | 193 / 193 / 0 / 0 / 0 / 0 | 6/6 | 4/6 |
| `chat_template_kwargs.reasoning_effort: none` | 200 | 115 / 185 / 0 / 0 | 4/4 | 2/4 |
| `chat_template_kwargs.reasoning_effort: max` | 200 | 185 / 168 | 2/2 | 0/2 |
| bogus | **422** | – | – | – |

- Every documented effort and omission is accepted and answered. Adjacent
  efforts overlap on this prompt; no monotonic budget claim.
- **`none` behaves differently from Together.** On Together `none` was a soft
  minimum that still filled `reasoning_content`. On DeepInfra top-level `none`
  reported 0 reasoning tokens in 10/10 divisor runs and the model wrote its
  working ("Factor 7560 … Need final only number.64") into `content`. The
  prompt renders 2 tokens shorter at `none` (29 vs 31), so the template is
  applied; the reasoning parser then finds no reasoning block. On a prose
  prompt `none` gave a clean answer (2/2), with tools it still produced a
  short `reasoning_content` and a correct call (2/2), and in the 6K cache pair
  one `none` request leaked and one did not. `reasoning.enabled=false` renders
  like `none` (29 prompt tokens) and leaked 4/6. This is a **warn**: the answer
  is still right, but users who pick `none`/`off` on DeepInfra should expect
  visible working on problems that need it.
- The unknown value is rejected by DeepInfra's request validation (`422`,
  "Input should be 'none', 'minimal', … or 'max'"), not by the template.
  fast-agent validates locally anyway.
- Default: omission behaves like the explicit efforts (prompt 31 tokens,
  reasoning in `reasoning_content`); the template default is `high`.

## Structured output — Wire contract

**Evidence:** `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-structured.json`.

The prescribed schema requires `city: string`, `population_millions: number`, and disallows extra properties. Prompt: `Give the largest city in Japan and its population in millions.` Complete visible content was parsed and checked using `jsonschema.validate`, not just HTTP status or excerpts.

| Mode (two repeats each) | Status | Valid | Tool first? |
|---|---|---|---|
| Standard json_schema, strict | 400 / 400 | false / false | n/a |
| json_object, schema in prompt | 200 / 200 | true / true | n/a |
| Legacy json_object + schema | 200 / 200 | true / true | n/a |
| Standard schema + get_weather | 400 / 400 | false / false | no response |
| Standard schema at max effort | 400 / 400 | false / false | n/a |
| Legacy schema + get_weather | 200 / 200 | true / true | **no / no** |
| Legacy schema at max effort | 200 / 200 | true / true | n/a |

Standard envelope errors report missing required `response_format.json_schema`; legacy succeeds. The prior conflicting enum/prompt probes still support actual schema enforcement beyond accidental compliance. **Warn:** legacy auto tools are skipped despite the request to use get_weather first. D2 remains legacy schema + defer, provisional; later harness conformance must prove tool-then-final-schema behavior. This stage does not establish strict-envelope equivalence or population factuality.

## Structured output — Wire contract: deepinfra

**Evidence:** `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-deepinfra-structured.json`.

| Mode | Valid | Tool called |
|---|---|---|
| `json_schema` strict | 2/2 | – |
| `json_schema` non-strict | 2/2 | – |
| `json_schema` strict, streamed | 2/2 | – |
| `json_schema` at `max` | 2/2 | – |
| `json_object` + schema in prompt | 2/2 | – |
| legacy `{"type": "json_object", "schema": …}` | 1/2 (plus 0/1 in the comparison) | – |
| `json_schema` + get_weather | 2/2 | **0/2** |
| `json_schema` + get_weather, streamed | 2/2 | **0/2** |

The standard nested envelope is the right form here; the legacy envelope is
accepted but does not enforce the schema (one answer was prose). With a
schema and an auto tool the model answers from memory without calling the
tool, as on Together: keep `structured_tool_policy="defer"`. Graded **warn**
per the contract playbook because of the skipped tool; the harness `defer`
policy is assessed separately in conformance.
