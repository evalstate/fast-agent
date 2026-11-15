# Inkling 2 Small — integration report

How fast-agent onboarded Inkling 2 Small: stage status, checks, decisions and the stage reports. Paths starting `fast-agent-model-lab/` are files in the fast-agent-model-lab repository, which holds the onboarding record and its evidence.

## Model card

```yaml
name: Inkling 2 Small
vendor: Thinking Machines
id: thinkingmachines/Inkling-2-Small
route: hf.thinkingmachines/Inkling-2-Small:together
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
| Tool preference | pending |
| Programmatic tool calling | blocked |
| Documentation and product surfaces | passed |
| Release readiness | pending |
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
| preference | Shell, process and edit tool names and arguments used with none declared | missing |  |
| preference | First choice among declared equivalent tools | missing |  |
| preference | Background start, poll, wait and stop with the default contract | missing |  |
| preference | Edit tool reliability with the default contract | missing |  |
| preference | Shell profile, shell name and edit tool recommendation | missing |  |
| preference | Resolved recommendation applied to the catalog; affected conformance checks rerun | missing |  |
| ptc | PTC evals pass with and without PTC | pass | Real CLI: all 7 PTC answer/path checks and all 7 direct answer checks pass. |
| ptc | Functions are called from code; no direct-call confusion | warn | Corrected structured fixture 5/5 with nested calls; original scalar fixture 1/5 retained as product-boundary finding. Shell ledger 2/3; recovered unsupported imports. Supervisor-approved interpretation. |
| ptc | Canonical function names; no mangled or model-specific names | pass | Live traces and offline render agree: common bash in both modes; no mangled or model-specific names. |
| ptc | Matched direct and PTC benchmark probe | skip | prepared, awaiting supervisor launch |
| ptc | Sandbox-sensitive probe before PTC becomes the default | skip | prepared, awaiting supervisor launch |
| ptc | PTC default recommendation | pass | D9 resolved: retain PTC off pending the authorized matched probes and uptake follow-up. |
| ptc | Resolved PTC default applied to the catalog; affected conformance checks rerun | pass | Existing catalog default False retained per resolved D9; direct tool-loop 3/3, corrected PTC 5/5 and related unit tests 221/221 pass. |
| surfaces | Provider page section with dated verification note | pass | New "Inkling 2 Small via Together" section on huggingface.md with alias, route, limits, modalities, reasoning ladder, structured stance, PTC off, caveats and dated verification note; provisional on D1-D5, D7 |
| surfaces | Generated references regenerated without unrelated drift | pass | docs.py generate changed only model_aliases_hf.md and models_reference.md; no generation warnings; no unrelated drift |
| surfaces | Provider default, setup template and examples | pass | HF default unchanged (no decision moves it); setup template names no HF current models; check models hf lists inkling; no predecessor to replace |
| surfaces | PTC default lists in docs and examples | skip | PTC stage still running; left for a follow-up run. Today inkling resolves PTC off and appears in no list; all copies match resolution for kimi/glm53flash/astra/glimmer except the pre-existing stale setup-template comment |
| surfaces | Strict docs build | pass | uv run scripts/docs.py build exit 0 (social cards + zensical build --strict) |
| review | Independent audit of the whole change; blockers resolved | missing |  |
| review | Final format, lint, typecheck and unit tests | missing |  |
| review | Live smoke through the public alias | missing |  |
| review | No traces, credentials or unrelated changes | missing |  |
| review | Integration report compiled and published to the docs site | missing |  |
| release | Pull request raised with green checks | missing |  |
| release | Version published | missing |  |
| benchmark | Benchmark, route, reasoning, PTC mode and published version stated | missing |  |
| benchmark | Cohort complete | missing |  |
| benchmark | Results reviewed and published to the docs site | missing |  |

## Decisions

### D1: Nominal max_output_tokens? No vendor output ceiling is published; the route omits max_tokens by default (omit_default_max_tokens), and an omitted-limit generation reached 40,011 tokens. ModelParameters still needs an integer (it also sizes shell output).
Resolution: 128000
Note: none
Why: Same nominal budget as Muse Glimmer on HF/Together; provider default is still used on the wire. Evidence: reports/02-contract.md output-limit

### D2: Structured output on HF/Together: standard json_schema is rejected; the legacy json_object+schema envelope validates but can suppress an auto tool call.
Resolution: legacy schema + defer
Note: none
Why: Enforces the schema and keeps tools usable; same adapter as Kimi K3.1 on Together. Conformance must show tool-then-schema works. Evidence: reports/02-contract.md structured

### D3: Default reasoning effort?
Resolution: high
Note: none
Why: Chat template default is high (0.9); explicit value avoids the omitted-effort outlier seen once. Evidence: reports/02-contract.md reasoning-ladder

### D4: Expose reasoning 'none'? It is a soft minimum: zero reasoning tokens on short prompts, but reasoning still appears on long prefixes, and reasoning.enabled=false is ignored.
Resolution: expose none as lowest effort
Note: none
Why: Matches the vendor ladder; docs must not promise reasoning off. Evidence: reports/02-contract.md reasoning-ladder, usage-cache

### D5: Curated selector and alias: add alias 'inkling' and list Inkling 2 Small in the Hugging Face picker as a current model?
Resolution: alias inkling + picker entry
Note: none
Why: Same treatment as Muse Glimmer (alias glimmer, picker entry)

### D6: Model class for optional stages? 'Small' in name, but 1M context, agentic positioning, 500 GB VRAM. mid = preference prior + lifecycle and PTC probe panel; small = neither; flagship = full preference panel, sandbox probe and a post-release benchmark.
Resolution: flagship
Note: New model family: treat as flagship
Why: Vendor positions it for agentic and coding use; cost is low enough for the probe panels

### D7: Reasoning replay policy for Inkling on HF/Together? All shared replay variants (reasoning, reasoning_content, omitted) answer correctly; preservation is supported, not proven mandatory.
Resolution: preserve raw reasoning
Note: none
Why: Template and provider guidance retain reasoning_content; preserve_raw_reasoning follows that contract without claiming omission fails. Evidence: evidence/contract-reasoning-transport.json and reports/02-contract.md.

### D8: Change the existing 4,718,592-byte HF request budget for Inkling? A recognized-message body around 4.6 MiB reaches Together; 5 and 6 MiB receive router 413. Exact cutoff and media-specific limits remain unmeasured.
Resolution: retain existing 4718592-byte budget
Note: none
Why: The default is conservative, not an asserted measured ceiling; this contract does not justify a global router budget change. Evidence: evidence/contract-context-limit.json.

### D9: Keep ptc_enabled=False pending an authorized matched probe? CLI evals pass, but scalar-result concatenation recurs and one shell run records no nested calls.
Resolution: off pending authorized probe
Note: Probe work authorized; revisit the default on its result
Why: Campaign not authorized; exact PTC conformance repeats pass only 1/5 while nested functions execute. Catalog already defaults off; evidence/ptc-uptake.json and plans/2026-11-14-ptc-bench-probe.md.

### D10: Authorize the planned matched tb21-probe3 PTC off/on campaign after artifact, pricing, retry and timeout prerequisites are pinned?
Resolution: authorize pinned campaign plan
Note: User authorized D9/D10 work, 14 Nov
Why: No campaign authorization granted. Prepared plan is plans/2026-11-14-ptc-bench-probe.md; 15 trials per arm diagnose uptake, not equivalence.

### D11: Default backend and PTC default for Inkling 2 Small?
Resolution: DeepInfra default, PTC off
Note: Human decision 14 Nov: alias inkling points at DeepInfra; ptc_enabled stays False
Why: tb21-probe3 observations from small separate probes (15 trials per mode per backend; different wheels, scheduling and times, so not a matched comparison): DeepInfra 30/30 with no empty completions and faster observed decoding; Together PTC 13/15 (0/30 vs 2/30 not statistically distinguishable). PTC cost 1.4x on DeepInfra and shows sandbox friction on both


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
- ptc/uptake: Corrected structured fixture 5/5 with nested calls; original scalar fixture 1/5 retained as product-boundary finding. Shell ledger 2/3; recovered unsupported imports. Supervisor-approved interpretation.

## Not tested

- ptc/bench-probe: prepared, awaiting supervisor launch
- ptc/bench-sandbox: prepared, awaiting supervisor launch
- surfaces/ptc-lists: PTC stage still running; left for a follow-up run. Today inkling resolves PTC off and appears in no list; all copies match resolution for kimi/glm53flash/astra/glimmer except the pre-existing stale setup-template comment

## Documentation and access

Report date: 2026-11-14; author: codexresponses.gpt-6-astra?reasoning=high; fast-agent: c55165833

### Inkling 2 Small — documentation and access

#### Outcome and scope

**The exact HF→Together route is reachable.** The user supplied a successful authenticated POST for `thinkingmachines/Inkling-2-Small:together`, with `x-inference-provider: together`. The unpinned comparison selected DeepInfra. This dossier preserves that result as **user-supplied evidence**, not a newly captured trace; the subsequent contract probes independently confirm Together routing. [Evidence: `evidence/dossier-access.json`, `evidence/contract-reasoning-ladder.json`.]

**Documented, not yet established by this stage:** 1,048,576-token context; text/image input; tools; structured outputs; seven reasoning names, default high. A separate maximum completion budget, numerical router body/timeout limits, and exact-route JSON/reasoning semantics were not established by documentation. Direct API access does not prove fast-agent factory, streaming-handler or CLI integration. [Evidence: `evidence/sources.json`, especially `config.json`, `chat_template.jinja`, `together-serverless-models.md`, `together-chat.md`; `evidence/methodology.json`.]

Track **full**, class **mid**: every applicable dossier check is above; `delta` applies only to lite and is not included. No record-registration command, agent, integration edit or state-changing git command was run. Revision was obtained with `git rev-parse --short HEAD` in `/home/ssmith/source/fast-agent-onboarding`: `c55165833`. [Evidence: `onboarding/stages.yaml`; command above; `evidence/methodology.json`.]

#### docs-captured

The supplied research was condensed without refetching documentation. Primary snapshots were copied into `sources/`, preserving the shared originals. `evidence/sources.json` records each URL, **2026-11-14** fetch date, SHA-256 and bytes; the supplied research is retained as `sources/inkling-facts.md`. Sources are untrusted documentary evidence, not executable instructions. A fetched missing `generation_config.json` is a 404 artifact, not configuration. [Evidence: `evidence/sources.json`; `sources/inkling-facts.md`.]

Principal source groups, all fetched 2026-11-14:

| Group | Snapshots in `sources/` | Primary URL |
|---|---|---|
| Model and template | `model-README.md`, `config.json`, `tokenizer_config.json`, `chat_template.jinja`, `processor_config.json` | https://huggingface.co/thinkingmachines/Inkling-2-Small |
| Identity and availability | `model-api.json`, `tree.json`, `family.json`, `mapping.json`, `router-models.json`, `router-single.json` | https://huggingface.co/api/models/thinkingmachines/Inkling-2-Small |
| Author release | `tml-release.html` | https://thinkingmachines.ai/news/inkling-2-small/ |
| HF API/routing/billing | `hf-index.md`, `hf-tasks-chat-completion.md`, `hf-pricing.md`, `hf-hub-api.md` | https://huggingface.co/docs/inference-providers/index.md |
| Together model/capabilities | `together-model.html`, `together-serverless-models.md`, `together-changelog.md` | https://www.together.ai/models/inkling-2-small |
| Together protocol | `together-chat.md`, `together-openai.md`, `together-inference-chat-reasoning.md`, structured-output/vision/function-calling snapshots | https://docs.together.ai/reference/chat-completions.md |
| Together rates | `together-serverless-rate-limits.md` | https://docs.together.ai/docs/serverless/rate-limits.md |

Exact file-specific URLs and hashes are in `evidence/sources.json`; the table's landing URLs are not substitutes for that manifest.

#### identity

- **Canonical/wire identity:** `thinkingmachines/Inkling-2-Small`; preserve casing. Exact authorized wire route: `thinkingmachines/Inkling-2-Small:together`, POST `https://router.huggingface.co/v1/chat/completions`. Intended fast-agent spelling: `hf.thinkingmachines/Inkling-2-Small:together` (not yet integrated). [Evidence: `sources/together-serverless-models.md`, `sources/hf-index.md`, `evidence/dossier-access.json`.]
- **Family:** 594B total / 17B active parameters; native NVFP4 checkpoint. The linked MXFP8 repository is a quantization sibling, not a different-sized second-generation model. Public family search did not confirm another Inkling 2 size. Inspected Hub revision: `8361e7a31ae7e9966a49a91d3994fde618936c2b`. [Evidence: `sources/model-README.md`, `sources/family.json`, `sources/model-api.json`.]
- **Dates have different meanings:** November 4 repository creation / Together marketing date; November 5 author announcement and initial Tinker availability; November 12 open weights / Together serverless launch. [Evidence: `sources/model-api.json`, `sources/together-model.html`, `sources/tml-release.html`, `sources/together-changelog.md`.]

#### limits

| Limit | Documentary finding | Evidence |
|---|---|---|
| Context | **1,048,576 tokens**, agreed by model/tokenizer configuration and Together catalog | `sources/config.json`, `sources/tokenizer_config.json`, `sources/together-serverless-models.md` |
| Maximum completion | **Unknown**; no separate published model-specific ceiling | `sources/tree.json`, `sources/generation_config.json` (404), `sources/together-chat.md`, `sources/together-serverless-models.md` |
| Local-serving 262,144 default | Hardware/deployment recipe, **not** a Together API ceiling | `sources/vllm.html` |
| Image dimensions | Each dimension *ideally* 40–4096 px, not a hard upload limit | `sources/model-README.md` |
| HTTP request/image bytes, image count, gateway timeout | No numerical exact-route contract in reviewed docs; API mentions 504 without a duration | `sources/hf-tasks-chat-completion.md`, `sources/together-chat.md`, `sources/together-inference-vision-inputs.md` |

The evaluation's 1M-token trajectory is not a maximum completion budget. The processor's 40×40 configuration is not a maximum image size. [Evidence: `sources/tml-release.html`, `sources/processor_config.json`.]

#### modalities

The card specifies **UTF-8 text + images → text**. Native audio is explicitly unsupported; the template accepts text/image parts and rejects unsupported content types. Video training, rendered document understanding and evaluation resizing do not establish native video/PDF upload support. Tool-role images are representable in the template but were not access-tested in this stage. [Evidence: `sources/model-README.md`, `sources/chat_template.jinja`, `sources/tml-release.html`.]

Together's generic vision documentation shows both remote and base64 data URLs; endpoint-specific formats/size handling remain contract work. [Evidence: `sources/together-inference-vision-inputs.md`, `sources/together-inference-vision-overview.md`.]

#### reasoning-controls

| Template name | Value |
|---|---:|
| none | 0 |
| minimal | 0.1 |
| low | 0.2 |
| medium | 0.7 |
| high | 0.9 |
| xhigh | 0.99 |
| max | 0.99 |

Omission/null defaults to **high/0.9**. The template accepts numeric values in `[0, 0.99]`; names are case-sensitive after trimming. This is a template contract, not by itself router acceptance. [Evidence: `sources/chat_template.jinja`.]

The tokenizer/template use **`reasoning_content`**, preserve assistant reasoning in history, and represent thinking separately from text. Together's current docs allow model-dependent `reasoning` or `reasoning_content`; they say to replay the returned field and count reasoning in completion usage/budget. The generic API's narrower `low/medium/high` enum and optional `reasoning.enabled` do not settle Inkling's actual controls. [Evidence: `sources/tokenizer_config.json`, `sources/chat_template.jinja`, `sources/together-inference-chat-reasoning.md`, `sources/together-chat.md`, `sources/together-openai.md`.]

#### capabilities

| Capability | What documentation supports; what remains unverified here | Evidence |
|---|---|---|
| Tools | Model-specific function-calling support; generic auto/none/required/named selection; template represents multiple calls | `sources/together-serverless-models.md`, `sources/together-inference-function-calling-best-practices.md`, `sources/chat_template.jinja` |
| Parallel tools | Generic multiple-call guide; exact route and flag not established by docs | `sources/together-inference-function-calling-parallel.md` |
| Tool history | IDs/names associate results; template expects parsed arguments internally, not proof wire arguments should be objects | `sources/chat_template.jinja` |
| Structured output | Catalog advertises support; API describes `json_schema` and `json_object`; strict enforcement/combinations require probing | `sources/together-serverless-models.md`, `sources/together-chat.md`, `sources/together-inference-chat-structured-outputs.md` |
| Streaming/usage | SSE, optional terminal usage; model-dependent reasoning field | `sources/hf-tasks-chat-completion.md`, `sources/together-inference-chat-reasoning.md` |
| Cache | Discount advertised; HF-routed cache reporting/discount unverified | `sources/together-serverless-models.md`, `sources/hf-pricing.md` |
| Sampling | Temperature/top-p/seed/penalties documented; no universal model-specific default. Evaluations used temperature 1.0 and xhigh | `sources/together-chat.md`, `sources/tml-release.html` |

The author warns of long multi-turn degradation, verbosity and imperfect tool selection in skill-heavy codebases; these motivate later harness/conformance/preference work, not model-specific tool changes in this stage. [Evidence: `sources/model-README.md`, `sources/tml-release.html`.]

#### pricing

Together lists **$0.30/M input, $1.00/M output, $0.03/M cached input**. Reasoning is completion output for billing. HF documents provider pass-through pricing, but its router record lacks a Together pricing entry here; do not substitute DeepInfra's flags or promise a routed cache discount from that record. [Evidence: `sources/together-serverless-models.md`, `sources/together-inference-chat-reasoning.md`, `sources/hf-pricing.md`, `sources/router-single.json`.]

No fixed Inkling RPM/TPM or account-specific guarantee was established. Together describes best-effort service, potential 429/503 responses and contractual exceptions; no rate/load test is justified by these docs. [Evidence: `sources/together-serverless-rate-limits.md`.]

#### discrepancies

| Topic | Conflict/gap | Consequence | Evidence |
|---|---|---|---|
| Availability | Hub says Together and DeepInfra live; router listing/single-model expose only DeepInfra; pinned Together nevertheless succeeds | Pin `:together`; listing absence is not route failure | `sources/mapping.json`, `sources/router-single.json`, `evidence/dossier-access.json` |
| Capabilities | No Together router provider record | Use Together documentation plus exact-route tests, not DeepInfra capability flags | `sources/router-single.json`, `sources/together-serverless-models.md` |
| Reasoning names | Seven template names vs narrower generic API enum | Probe all names/numerics/toggle/forwarding | `sources/chat_template.jinja`, `sources/together-chat.md` |
| Reasoning field | Older generalized `reasoning` guidance vs current model-dependent guidance/template `reasoning_content` | Inspect final messages, deltas and replay | `sources/together-changelog.md`, `sources/together-openai.md`, `sources/tokenizer_config.json` |
| Output/context | Context documented; output maximum missing | Do not label 1M, 32K or 64K as vendor output maximum | `sources/config.json`, `sources/together-chat.md` |
| Temperature | HF describes 0–2; Together describes 0–1; evals use 1.0 | Do not turn evaluation settings into universal defaults | `sources/hf-tasks-chat-completion.md`, `sources/together-chat.md`, `sources/tml-release.html` |
| Release dates | November 4/5/12 label distinct events | Retain event-specific chronology | Identity sources above |
| Media | Broad pixel-format card language / video training vs restricted API content types | Advertise text/image only; test MIME transport | `sources/model-README.md`, `sources/chat_template.jinja` |
| Router architecture/limits | No confirmation of an OpenRouter migration or larger byte allowance | No architecture/limit change justified by reviewed docs; internal hops remain unknown | `sources/hf-index.md`, `sources/hf-pricing.md`, `sources/inkling-facts.md` |
| Cached pricing | Together publishes discounted cache input; HF Together record absent | Verify cache usage separately from actual billing discount | `sources/together-serverless-models.md`, `sources/router-single.json` |

#### access and spend

The supplied pinned access returned HTTP 200, reasoning in `message.reasoning_content`, and `usage.completion_tokens_details.reasoning_tokens`. The supplied unsuffixed comparison selected DeepInfra. Raw responses, usage quantities and dollar charges for these prior probes were not supplied and remain **unknown**. [Evidence: `evidence/dossier-access.json`.]

**New dossier inference requests: 0; incremental dossier spend: $0.** Two prior requests are described by the user (one pinned, one comparison); their spend is not counted as zero. This worker's newly measured requests/spend are all allocated to the contract report and `evidence/spend.json`.

#### Decisions to raise

| Question | Options | Recommendation and why | Blocks |
|---|---|---|---|
| What separate output ceiling belongs in the catalog? | Published maximum if obtained; explicit product budget; omit provider default | Do not invent a vendor ceiling; prefer omission pending contract evidence and catalog representation decision | integration |
| Which reasoning default/off promise should users get? | Provider omission; explicit high; cheaper effort; hard-off promise vs soft control | High is the documented default; probe before promising an off mode or deterministic effort ordering | integration, surfaces |
| Should router listing omission prevent onboarding? | Block on listing repair; require pinned live verification | Pinned verification is the relevant access gate; keep listing discrepancy visible and never silently select DeepInfra | integration, review |
| Should advertised JSON/tools be enabled unchanged? | Direct schema; route adaptation; object-only; restricted coexistence | Decide from the exact-route contract, not generic flags; the contract report supplies the resulting adaptation/policy questions | integration, conformance |

## Wire contract

Report date: 2026-11-14; author: codexresponses.gpt-6-astra?reasoning=high; fast-agent: c55165833

### Outcome

**Contract rework complete: 7 pass, 4 warn, 0 fail, 0 skip.** This is direct-wire evidence for **`hf.thinkingmachines/Inkling-2-Small:together`**, not a fast-agent harness/conformance claim. Every successful request returned provider `together`; router 413s had no provider header.

The review gaps have been addressed with the shared helpers and required evidence shapes. Prior experiments remain under `prior_evidence` in each contract file; their limited original scope is not retroactively upgraded. New top-level fields supersede old scope notes. Product recommendations remain **provisional**, not decisions made by this worker.

Known-usage worker spend is **$0.2959222** across **238** calls (112 original + 126 rework), against the authorized **$5** budget. Including an approximate $0.002 reviewer spend and a $0.033 reserve for the usage-less successful stream gives **$0.3309222** tracked/reserved. Actual account billing is uninspected. Evidence: `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/spend.json`.

#### Review closure

| Review problem | Correction / explicit limit |
|---|---|
| 1 — sampling | All nine parameters rerun individually; each has `documented_range` and `recommended`, including explicit unknowns. |
| 2 — ladder | Prescribed divisor prompt, streamed `include_usage`, max_tokens 32768, three repeats for valid settings; shared summarize output plus final-answer verification. |
| 3 — omitted outlier | Original seeded CRT prompt rerun twice at 32768, four times the earlier allowance: one `length`, one correct `stop`. |
| 4 — streaming | No-stream_options probe added; finish, usage, DONE and observed ordering fields populated. |
| 5 — tools | Complete streamed get_weather matrix, including named forcing with none and actual returned-call continuation. |
| 6 — context | Recognized filler-message probes replace ignored-field inference. Nominal 1.1×/4-char window probe explicitly skipped per >4 MiB rule; denser filler directly establishes input rejection. |
| 7 — cache | Approximately 5K prefix with genuinely different final lines; prewarming caveat disclosed below. |
| 8 — output | Prescribed output probes rerun; required nested shape populated; separate ceiling is null. |
| 9 — structured/media | Prescribed city schema and highest effort; prescribed image, PDF and MP4 fixtures, HTTPS image and tool-role image. |
| 10 — envelope/helpers | All eleven contract files have stage/check/date/route/result. Shared probe/replay/summarize used; instrumentation and exact rerun commands documented. |
| 11 — profile | Final table maps observation → evidence → actual field → value, including wire casing, polling and full ReasoningEffortSpec form. |
| 12 — budget | Corrected to $5; original spend preserved and every rework call appended. |

#### Method and reproducibility

`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/methodology.json` records each scratch body, SHA-256, parameters and repeats. Requests and complete visible answers remain only under `/tmp/onb-inkling-2-small/contract/`. The wrapper imports the shared `probe.py` and observes `_chunk` to retain complete answers, reconstruct tool calls, and record event order; it does not change HTTP transport, routing, bodies, or retry behavior. A supplemental single-tool run captures actual key presence/nulls and the usage chunk's empty choices. `replay.py` runs unchanged via `runpy`, with a wrapper around its shared `post` solely for provider/usage accounting. There were no automatic retries.

Core reproducible commands, from the checkout root:

```bash
D=/tmp/onb-inkling-2-small/contract
uv run python onboarding/stages/contract/probe.py "$D/body-ladder-high.json" 3
uv run python onboarding/stages/contract/probe.py "$D/body-tool-single.json" 1
uv run python onboarding/stages/contract/probe.py "$D/body-stream-no-options.json" 1
uv run python onboarding/stages/contract/summarize.py "$D" 64
ONB_MODEL=thinkingmachines/Inkling-2-Small:together uv run python onboarding/stages/contract/replay.py
```

The required `dossier.json`/`access.json` input filenames are absent in this checkout. Inputs were the dossier report (`fast-agent-model-lab/onboarding/models/inkling-2-small/reports/01-dossier.md`), sources manifest (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/sources.json`), and `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/dossier-access.json`, not invented replacements. Prior sources were fetched 2026-11-14.

#### reasoning-ladder

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

#### reasoning-transport

**Evidence:** `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-reasoning-transport.json`, streaming evidence (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-streaming.json`).

Both non-streamed messages and streamed deltas use **`reasoning_content`**. In observed reasoning-bearing text streams it preceded content; there was no observed interleaving, not a guarantee of future ordering.

The shared `replay.py` obtained a real `get_code` call and sent its result back. Replaying the returned reasoning as **`reasoning`**, **`reasoning_content`**, and **omitted** each returned HTTP 200 with the correct code. All four calls returned `together`, with usage accounted. Preserving raw reasoning is supported and follows template/provider guidance; **mandatory replay is not established**. D7 registers preservation as a product choice rather than inventing a server requirement. Historical lookup replay results are retained.

#### output-limit

**Evidence:** `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-output-limit.json`.

| Prescribed probe | Result |
|---|---|
| Omit max_tokens; integers 1–20000 | 60014 completion tokens, including 1006 reasoning; `stop` |
| max_tokens=64; divisor prompt, high | Exactly 64 completion tokens, `length`, no visible content |
| max_completion_tokens=64; same prompt | Exactly 64 completion tokens, `length`, no visible content |
| max_tokens=1000000; “Reply with OK.” | HTTP 200, `stop`; accepted reservation, not a million-token generation |

The omitted-limit run stopped naturally, so **provider default output cap remains null**, with observed generation now at least 60014 tokens. The prior 40011-token constrained generation and accepted 524288 reservation remain valid supplemental evidence. The prior 1048576 reservation error concerns combined context, not a separate output ceiling. **Separate ceiling is null.**

Omit automatic max_tokens provisionally. D1's **128000** recommendation is only a nominal product budget required by `ModelParameters`, not vendor capacity or a value to send automatically. Provider default context reservation remains unknown; do not invent a `prompt_context_window` subtraction.

#### context-limit

**Evidence:** `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-context-limit.json`, [config.json](https://huggingface.co/thinkingmachines/Inkling-2-Small/raw/main/config.json).

Documented and error-reported window: **1048576**. The nominal playbook 1.1× window at ~4 chars/token would be ~4613734 bytes, above 4 MiB; that exact case is skipped under the playbook's “window not separable from body cap” rule. An additional denser recognized-message probe **does independently establish input rejection**:

| Serialized request bytes | HTTP / provider | Observation |
|---|---|---|
| 4194494 | 400 / together | Error: maximum context 1048576; request has **2097167 input tokens** |
| 4823638 | 400 / together | Error: maximum context 1048576; request has **2411739 input tokens** |
| 5243044 | 413 / absent | `request entity too large` |
| 6291620 | 413 / absent | `request entity too large` |

Payload is `messages[0].content`, repeating ` a`, not an ignored top-level field. Exact provider errors are retained in the evidence. The router accepts a recognized body above the configured **4718592**-byte budget, but rejects 5/6 MiB. The exact cutoff remains unknown, hence **warn** and provisional D8 recommendation to retain the conservative existing budget.

`full_window_tested: false`: the largest earlier successful input remains **300033 tokens**. No additional ~$0.30 full-window success call was made because rejection/cap characterization is already established; this consciously leaves usable full-window success unverified, rather than treating the configured/error limit as a successful 1M-input test.

#### streaming

**Evidence:** `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-streaming.json`.

Observed finishes: **stop, length, tool_calls**. Every successful streamed run sent `[DONE]`. With `include_usage`, terminal usage arrived at top-level chunk; the supplementary raw-shape inspection confirms an empty `choices` list on that usage chunk. The required run **without stream_options returned no usage**, despite a standard finish and DONE.

Observed sequences include reasoning → content → finish → usage → DONE, content-only, reasoning → tools → finish → usage → DONE, tools-only, and truncated reasoning → length → usage → DONE. Empty/role-only deltas are omitted from these compact order summaries. These are observations, not a fixed ordering assumption.

#### tool-calls

**Evidence:** `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-tool-calls.json`.

All cases used streamed **get_weather(city: string)** with `additionalProperties: false`, default effort **high** except the explicit none case:

| Case | HTTP | Calls / finish |
|---|---|---|
| Single; Paris; tool_choice omitted | 200 | Paris; tool_calls |
| Parallel; Paris and Tokyo; flag true | 200 | both cities, indexes 0/1; tool_calls |
| Parallel; flag omitted | 200 | both cities, indexes 0/1; tool_calls |
| Required; “Say hello.” | 200 | one get_weather; tool_calls |
| Named; “Say hello.” | 200 | one get_weather; tool_calls |
| Named with none | 200 | one get_weather; tool_calls |
| Actual single-call result replay | 200 | correct Paris/17°C final answer; stop |

There were **16 tool fragments across the seven-case matrix**: each call had one metadata fragment, then one argument fragment. The helper records continuations as `[index, function.arguments]`; supplemental raw-key inspection confirms **id/type/function.name omitted, not null** for the single-call continuation. IDs and indexes reconstruct correctly. No malformed-fragment evidence motivates manual stream mode.

Arguments arrived all at once per call: this is **not evidence of incremental argument token streaming**. The captured Together best-practices/single-call guides describe `stream=true` and `delta.tool_calls`; no separate incremental-argument switch was found. Forced cases intentionally provide no city; generated argument choices are not factual weather claims. None is a soft effort control even though this forced case happened to report zero reasoning.

#### structured

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

#### media

**Evidence:** `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-media.json`.

| Fixture / transport | Result |
|---|---|
| tests/e2e/multimodal/image.png, inline image_url | 200; correctly reads **evalstate** |
| Same fixture via public HTTPS image_url | 200; correctly reads **evalstate** |
| Same fixture in tool-role image_url result | 200; correctly reads **evalstate** |
| tests/e2e/multimodal/sample.pdf, file/file_data | 400; unsupported chat content part type file for gRPC backends |
| tests/e2e/multimodal/tmp6vsgdcet.mp4, video_url data URI | 400; modality video not supported by inkling2 |
| input_audio WAV negative | 400; modality audio not supported by inkling2 |

The new audio negative is a small WAV-like payload; the prior **valid silent WAV** also received the model-level unsupported-modality error and is retained. Native PDF failure is not a claim about rendered pages or extracted text. Prior JPEG/WebP and 128px/4096px PNG semantic probes remain valid; 4096px success is not an upper bound. Dedicated image-token usage is **null**, not inferred from total prompt tokens. No native PDF/audio/video capability is proposed.

#### sampling

**Evidence:** `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-sampling.json`, [Together API schema](https://docs.together.ai/reference/chat-completions.md), [HF chat API](https://huggingface.co/docs/inference-providers/tasks/chat-completion.md), [author release](https://thinkingmachines.ai/news/inkling-2-small/).

Each parameter was sent **alone** on the minimal request (alongside fixed model/effort/stream/output controls):

| Parameter | Tested value | HTTP | Documented range / recommendation |
|---|---|---|---|
| temperature | 0.2 | 200 | Together 0–1 vs HF 0–2; author eval **1.0**, not a runtime default |
| top_p | 0.8 | 200 | Probability threshold; no numeric bound in captured Together schema; recommendation null |
| top_k | 40 | 200 | Integer candidate count; numeric bounds/recommendation unpublished |
| min_p | 0.1 | 200 | 0–1; recommendation null |
| presence_penalty | 0.3 | 200 | -2–2; recommendation null |
| frequency_penalty | 0.3 | 200 | -2–2; recommendation null |
| repetition_penalty | 1.1 | 200 | Numeric range and recommendation null |
| seed | 42 | 200 | Integer; numeric bounds/recommendation unpublished |
| stop | ["BLUE"] | 200 | Array of strings; count/length bounds and recommendation unpublished |

All nine required rows have `status`, `error`, `documented_range`, and `recommended`. This is **acceptance**, not proof every knob affects sampling or of range endpoints/reproducibility. No sampling defaults are proposed for the alias.

#### usage-cache

**Evidence:** `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-usage-cache.json`, `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/spend.json`.

Usage includes prompt/completion/total tokens; prompt details cached/audio tokens; completion details reasoning/audio/accepted-prediction/rejected-prediction tokens. Missing dedicated image-token and cache-write fields remain missing.

The corrected stable prefix is about **5000 tokens** (exact prefix-only tokenization unknown), with **different actual final lines**, sent consecutively:

| Repeat | Whole prompt tokens | Cached tokens | Completion / reasoning | Finish |
|---|---|---|---|---|
| 1 | 5016 | 4864 | 6 / 0 | stop |
| 2 | 5017 | 4864 | 352 / 334 | stop |

**Prewarming caveat:** the initial new pair accidentally used a literal backslash-n separator; both reported zero cached tokens. The corrected pair reused that stable prefix, so the first corrected request was already a hit. This establishes **provider-reported cache hits on a suffix-mutated prefix**, not cold-fill probability, TTL, or a latency-derived cache claim. The earlier identical ~3K pair's 3072-token cache hit is retained but is not substituted for the prescribed new probe.

No retention cohorts were run. Leave `cache_ttl` unset and retain the settings model's **process_poll_default_wait_seconds=0**; cache observations do not justify a polling-policy change.

#### Spend, validation and remaining scope

The rework made **126** calls with usage-bearing totals **24961 input + 111545 completion tokens**, estimated **$0.1190333** at $0.30/M input + $1.00/M completion. Cached input is conservatively priced uncached, and reasoning is not charged twice. Total worker usage-bearing estimate, including the original $0.1768889, is **$0.2959222**. **35** responses across both runs lack usage; these are missing, not zero-cost assertions. User-supplied earlier access/comparison calls have unknown costs and are not invented into the ledger. See `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/spend.json`.

No repository runtime/source code was changed. Required repo checks were nevertheless run:
- `uv run scripts/lint.py`: pass.
- `uv run scripts/typecheck.py`: pass.
- `uv run scripts/format.py --check`: fails on the unrelated, concurrently owned `onboarding/notes/ptc-function-naming.md`.
- `uv run scripts/format.py --path onboarding/models/inkling-2-small`: no changes; the unrelated file was not touched. Full recheck and artifact validation results are recorded in `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/report-validation.json`.

Schema validation, envelopes, check IDs, budgets, complete-answer correctness, provider pinning, source hashes and evidence preservation are independently checked offline. No campaign, integration edits, harness inference, TUI, release work or full-window-success test occurred.

#### Product decisions

Existing D1–D4 remain open; this report updates the evidence rather than resolving them. D5/D6 belong to selector/class policy and are unchanged.

| Decision | Provisional recommendation / caveat |
|---|---|
| D1 — nominal output budget | 128000 as product budget only; omit default max_tokens; ceiling unknown |
| D2 — schema/tool policy | Together legacy schema + defer; later harness gate required |
| D3 — default effort | high per template; not proven to prevent intermittent looping |
| D4 — lowest effort | Expose none only as a soft minimum; short prompts can also emit reasoning |
| D7 — reasoning replay (newly registered) | Preserve raw reasoning per guidance; all three replay variants work here |
| D8 — request budget (newly registered) | Retain existing 4718592-byte default; no exact/media-specific cutoff claim |

#### proposed-profile

**Proposal only; not applied.** Evidence: `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-proposed-profile.json`. Targets are read from `src/fast_agent/llm/model_database.py`, `src/fast_agent/llm/reasoning_effort.py`, `src/fast_agent/llm/provider/openai/huggingface_router_profiles.py`, and `src/fast_agent/config.py` at c55165833. Unknown limits stay unset; explicit product recommendations are marked provisional.

| Observation | Evidence file | Target field | Value |
|---|---|---|---|
| Documented/error-enforced combined window | context-limit (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-context-limit.json`) | ModelParameters.context_window; HuggingFaceRouteProfile.context_window | 1048576; full-window success unverified |
| Default context reservation unknown | output-limit (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-output-limit.json`) | HuggingFaceRouteProfile.prompt_context_window | None; no invented subtraction |
| Separate output ceiling unknown | output-limit (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-output-limit.json`) | ModelParameters.max_output_tokens | 128000 **provisional D1 product budget**, not vendor maximum |
| Omitted output reaches 60014 tokens naturally | output-limit (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-output-limit.json`) | HuggingFaceRouteProfile.omit_default_max_tokens; default_max_tokens; max_output_tokens | True; None; None |
| Text and tested image formats work | media (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-media.json`) | ModelParameters.tokenizes | text/plain, image/png, image/jpeg, image/webp |
| Legacy schema validates; active schema skips auto tools | structured (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-structured.json`) | ModelParameters.json_mode; structured_tool_policy | schema; defer — **provisional D2** |
| Standard schema envelope rejected | structured (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-structured.json`) | HuggingFaceRouteProfile.structured_json_mode; use_legacy_json_schema | schema; True, Together-only |
| Seven efforts answer; none is soft; template default high | reasoning-ladder (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-reasoning-ladder.json`) | ModelParameters.reasoning_effort_spec | ReasoningEffortSpec(kind="effort", allowed_efforts=["none","minimal","low","medium","high","xhigh","max"], allow_toggle_disable=False, allow_auto=False, default=ReasoningEffortSetting(kind="effort", value="high")) — **provisional D3/D4** |
| Top-level effort accepted; toggle does not disable | reasoning-ladder (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-reasoning-ladder.json`) | HuggingFaceRouteProfile.reasoning | TopLevelReasoningEffort(default_effort="high", allowed_efforts=frozenset({"none","minimal","low","medium","high","xhigh","max"}), toggle_disable=False) |
| Reasoning field and all replay variants work | reasoning-transport (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-reasoning-transport.json`) | ModelParameters.reasoning; HuggingFaceRouteProfile.preserve_raw_reasoning | reasoning_content; True — **provisional D7**, not proven mandatory |
| Indexed tool fragments, no malformed-fragment evidence | tool-calls (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-tool-calls.json`), streaming (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-streaming.json`) | ModelParameters.stream_mode | openai; harness accumulation remains conformance work |
| Sampling acceptance, no general-use recommendation | sampling (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-sampling.json`) | ModelParameters.default_temperature; alias query defaults | None; {} |
| Cache hits observed, no retention/polling experiment | usage-cache (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-usage-cache.json`) | ModelParameters.cache_ttl; process_poll_default_wait_seconds | None; retain model default 0 (no override) |
| Exact wire ID is mixed case | methodology (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/methodology.json`) | ModelDatabase._PROVIDER_WIRE_MODEL_NAMES | (Provider.HUGGINGFACE, "thinkingmachines/inkling-2-small") → "thinkingmachines/Inkling-2-Small"; preserve :together suffix in routing |
| Configured byte budget is conservative, exact cap unknown | context-limit (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-context-limit.json`) | HuggingFaceSettings.max_request_bytes | Retain 4718592 — **provisional D8**, no global change |

## Catalog and provider integration

Report date: 2026-11-14T00:00:00; author: copilot.claude-opus-5.5; fast-agent: c55165833 (0.10.54, uncommitted working-tree changes)

### Outcome

Inkling 2 Small is in the catalog on the Hugging Face router, pinned to `:together`.
All results here are **offline**: the request fast-agent builds was checked with
`_prepare_api_request` and the CLI resolvers, with no provider calls. Whether the model
works end to end through the harness is still open; the conformance stage covers that.
Values that depend on open decisions D1, D2, D3, D4, D5, D7 and D8 are **provisional**.
Each one follows the decision's recommendation. Evidence:
`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/integration.json`.

#### model-database

`src/fast_agent/llm/model_database.py`:

- New `INKLING_2_REASONING_EFFORT_SPEC` with the ladder `none, minimal, low, medium, high,
  xhigh, max`. The default is `high` (D3), and `none` is exposed as the lowest effort (D4).
  `allow_toggle_disable` and `allow_auto` are false, as
  `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-proposed-profile.json` proposed.
- New `INKLING_2_SMALL_HF`:
  - `context_window=1_048_576`
  - `max_output_tokens=128_000`. This is the D1 nominal budget, not a vendor ceiling. No
    vendor ceiling is published, and the route leaves `max_tokens` out on the wire.
  - `tokenizes=OPENAI_VISION` (text, jpeg, png, webp)
  - `json_mode="schema"`, `structured_tool_policy="defer"` (D2)
  - `reasoning="reasoning_content"`
  - `stream_mode` left at its default, `openai`
  - `ptc_enabled` left at `False` (the PTC stage decides it)
  - `default_temperature`, `cache_ttl` and `model_specific` left unset. The contract had no
    evidence for them.
- `MODELS["thinkingmachines/inkling-2-small"]` (lowercase key).
- `_PROVIDER_WIRE_MODEL_NAMES[(HUGGINGFACE, "thinkingmachines/inkling-2-small")] =
  "thinkingmachines/Inkling-2-Small"`.

Resolution: `ModelFactory.resolve_model_spec('inkling')` returns
`Provider.HUGGINGFACE 128000 1048576 False`. `fast-agent check models --for-model inkling
--json` resolves to `thinkingmachines/Inkling-2-Small:together` on provider `hf`.

#### route-profile

`src/fast_agent/llm/provider/openai/huggingface_router_profiles.py` gets one rule:
`_INKLING_2_SMALL` with backend `together`. This model has no wildcard rule, so there is no
ordering hazard. The profile sets:

- `reasoning=_INKLING_2_REASONING`: `TopLevelReasoningEffort(default_effort="high",
  allowed_efforts=<7-step ladder>)`
- `preserve_raw_reasoning=True` (D7)
- `use_legacy_json_schema=True` (D2)
- `omit_default_max_tokens=True` (D1)

Two proposed fields were left out because they would be redundant:
- `structured_json_mode="schema"`: `json_mode` is already `"schema"` in ModelParameters.
- `context_window=1048576`: it already comes from ModelParameters.

`prompt_context_window`, `max_output_tokens` and `default_max_tokens` stay `None`, as
proposed.

**Deviation from the proposal: `toggle_disable` is left at its default (`True`), not
`False`.**
- Shared validation accepts `?reasoning=off` for any effort spec that includes `none`.
- With `toggle_disable=False`, an offline build of `inkling?reasoning=off` sent
  `reasoning_effort="high"`. The user asked for reasoning off and got the default. This is
  the silent-mis-send pitfall the playbook warns about.
- With the default, `off` sends `reasoning_effort="none"`, the soft minimum from D4.
- This sends no `reasoning.enabled` field. The contract showed that field is ignored, so it
  is not used.
- Docs must still not promise that reasoning is fully off (D4).

**Untested scope:**
- Only `:together` has a profile.
- Other HF backends (for example `:novita`) get no reasoning field and send
  `max_tokens=128000`.
- The contract covered `:together` only.

#### aliases

`model_aliases.py`: `"inkling": "hf.thinkingmachines/Inkling-2-Small:together"`. The alias
has no query defaults, because the contract proposed no sampling defaults
(`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-sampling.json`). `fast-agent model presets
--provider hf --json` lists `inkling` with model `thinkingmachines/Inkling-2-Small:together`.

This is a new family, so no alias was repointed or pinned. Like Muse Glimmer, the picker
uses the short alias with a display label, so no separate display-label alias was added
(D5, provisional).

#### selector

`model_selection.py`: `_builtin_entry("inkling", display_label="Inkling 2 Small (together)")`
in `Provider.HUGGINGFACE`, after Muse Glimmer, with `current=True` by default.
`fast-agent check models hf` shows the row. No existing entry is superseded.
`HUGGINGFACE_DEFAULT_MODEL` is unchanged.

#### hardcoded-sets

Searched the nearest siblings (Muse Glimmer, Kimi K3.1 Pro HF) and every known site. The
full list is in `integration.json` under `hardcoded_sets`. Nothing needed updating:

- Glimmer has no hits outside the catalog files.
- The Kimi hits are Moonshot-native (`llm_moonshot.py`) or the provider default
  (`config.py`).
- `FORCED_REASONING_MODELS`, `ASTRA_MODEL` and the Claude sets belong to other providers.
- The `mcp_agent.py` regexes (`codex`, `^gpt-5`, `^gpt-5.6`, `claude-`) do not match
  `thinkingmachines/inkling-2-small`, and its `default_provider` is HUGGINGFACE.

#### wire-tests

Added to `tests/unit/fast_agent/llm/providers/test_llm_huggingface_router_profiles.py`:

| Test | What it pins | Fails without the change because |
|---|---|---|
| `test_inkling_together_request_contract` (6 cases) | wire model `thinkingmachines/Inkling-2-Small:together`; `max_tokens` omitted; `reasoning_effort` high (default), low, none, off→none, max; lowercase input → mixed-case wire id; no `extra_body` | no alias/catalog (parse fails); no `omit_default_max_tokens` (128000 sent); `toggle_disable=False` (off→high); no wire map (lowercase id) |
| `test_inkling_together_preserves_explicit_max_tokens` | `?max_tokens=4096` reaches the request | omit rule would drop it if wrongly applied |
| `test_inkling_together_uses_legacy_schema_envelope` | `response_format == {"type": "json_object", "schema": …}` | without `use_legacy_json_schema` the json_schema envelope is sent |
| `test_inkling_together_replays_raw_reasoning_chunks` | `reasoning_content` is joined verbatim (`"first second"`) | without `preserve_raw_reasoning` the parts are joined with `"\n\n"` |

Also added to `tests/unit/fast_agent/llm/test_model_factory.py`:
`test_inkling_alias_uses_together_without_sampling_defaults`.

No new stream-replay test. The contract found standard indexed tool fragments with no
malformed continuation fragments
(`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-tool-calls.json`,
`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/contract-streaming.json`), so `stream_mode` stays
`openai`. `ptc_enabled` is unchanged, so the PTC test was not extended.

Focused runs:
- `uv run pytest tests/unit/fast_agent/llm -q -k inkling`: 10 passed.
- `uv run pytest tests/unit/fast_agent/llm tests/unit/fast_agent/ui/test_model_picker_common.py tests/unit/fast_agent/ptc -q`:
  3179 passed, 4 skipped.

#### quality-gate

| Command | Result |
|---|---|
| `uv run scripts/format.py --check` | fails on `onboarding/notes/ptc-function-naming.md` only. That file is untouched by this stage and owned by other work, so `format.py` (rewrite) was not run. `ruff format --check` on the 6 touched files: already formatted |
| `uv run scripts/lint.py` | pass |
| `uv run scripts/typecheck.py` | pass |
| `uv run scripts/cpd.py --check` | pass (7 approved findings, no new duplication) |
| `uv run pytest tests/unit -n auto -q` | 10673 passed, 9 skipped |
| `git diff --check` | clean |

#### Files changed

`git diff --stat` (this stage's files only):

```
 src/fast_agent/llm/model_aliases.py                |  1 +
 src/fast_agent/llm/model_database.py               | 23 ++++++++
 src/fast_agent/llm/model_selection.py              |  4 ++
 .../provider/openai/huggingface_router_profiles.py | 16 +++++
 .../test_llm_huggingface_router_profiles.py        | 69 ++++++++++++++++++++++
 tests/unit/fast_agent/llm/test_model_factory.py    |  8 +++
 6 files changed, 121 insertions(+)
```

Evidence added: `evidence/integration.json`. The offline request probe script is kept in
`/tmp/onb-inkling-2-small/probe_req.py`. It makes no network calls.

#### Carried forward

- **Conformance:**
  - Show that tool-then-schema works with `structured_tool_policy="defer"` and the legacy
    envelope (D2).
  - Show that streamed `reasoning_content` and tool fragments accumulate through the harness.
- **Surfaces:**
  - Document `inkling`.
  - Document that `reasoning=off`/`none` is a soft minimum, not a guaranteed disable.
  - Document that 128K is a nominal budget.
- **Decisions:** D1–D5, D7 and D8 remain open; the implementation follows each
  recommendation. The `toggle_disable` deviation above is reported for the reviewer. It
  needs no new decision, since it implements D4's "none as lowest effort" for the `off`
  spelling.

## Harness conformance

Report date: 2026-11-14T00:00:00; author: codexresponses.gpt-6-astra?reasoning=high; fast-agent: b812ee298 (0.10.54, working-tree suite changes)

### Outcome

**Conformance complete: 8 pass, 1 warn.** Live probes used the public `inkling`
alias through fast-agent on `hf.thinkingmachines/Inkling-2-Small:together`,
not a substitute endpoint. Streamed text, every effort, shell/function tool
loops, history/reasoning replay, structured output, supported images, unsupported
media handling, usage and interactive controls were verified.

**D4 caveat:** `none` is a soft minimum, not reasoning-off. `off` maps to
`reasoning_effort=none` and can also produce reasoning. The generic interactive
command calls `off` “disabled”; that label must not be read as a hard-off
guarantee. Results remain provisional on **D1, D2, D3, D4, D5, D7 and D8**.
No product choice was resolved by this worker; no new decision was needed.

The suite's separate **PTC uptake check is not a conformance check**. It passed
on the initial run and failed the final integer assertion on the second run,
despite two successful nested function calls. Both results are retained for
the PTC worker; this report does not claim reliable PTC uptake.

#### Evidence and check matrix

- Final suite evidence (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/conformance-suite.json`): all automated
  conformance results above are copied without softening; includes the PTC failure.
- Initial suite evidence (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/conformance-suite-initial.json`):
  preserves the original hard-off assertion failure and PTC pass.
- Manual evidence (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/conformance.json`): enveloped, sanitized
  observations, trace review, spend, suite defects and interactive evidence.
- Scratch scripts, histories and logs: `/tmp/onb-inkling-2-small/conformance/`.
  Raw request/chunk traces were deleted after review.

| Check | Command/probe | Result | Manual evidence key |
|---|---|---|---|
| text-stream | Suite + traced `go --model 'inkling?max_tokens=8192' --message 'Reply with exactly: ONB-TEXT-OK'` | pass: marker, `endTurn`, streaming request, 16 chunks | `text_stream` |
| reasoning-efforts | Suite + divisor prompt at all seven efforts and `off`; invalid `bogus` locally | warn: all answers 64, intended wire values; none/off still reason (D4) | `reasoning_efforts` |
| tool-loop | Suite deterministic functions, independent calls and real shell; manual `printf ONB-TOOL-OK` at default/high and max | pass: execution, continuation, final marker and exact reasoning replay | `tool_loop` |
| multi-turn | Suite three-turn history; CLI `--home` then `--resume` codeword recall | pass: recall and `reasoning_content` replay | `multi_turn` |
| structured | Suite + direct/schema-with-shell CLI + `check structured --mode all`; current-model matrix | pass: schema-valid direct, Pydantic and tool-first output | `structured` |
| media | Suite + CLI username question for PNG/JPEG/WebP; visible PDF removal; existing media e2e tests | pass: all image encodings understood; unsupported PDF omitted with clear warning | `media` |
| usage | Suite context accounting; raw usage vs history for text and both shell efforts | pass: five attempts match input/output/reasoning/cache; missing values remain null | `usage` |
| trace-review | Inspect normalized request/chunk files, compare replay/IDs, delete traces | pass | `trace_review` |
| interactive | `check models hf`, 106 unit tests, live TUI slash commands via individual keys, F6, picker | pass | `interactive` |

#### Reasoning and wire contract

Each manual effort used an explicit 8,192-token ceiling to bound probe cost.
All completed normally; no truncation retry was necessary. Suite calls kept
their 2,048-token ceiling.

| Requested | Effective top-level `reasoning_effort` | Divisor answer | Reported reasoning tokens |
|---|---|---|---|
| none | none | 64 | 137 |
| minimal | minimal | 64 | 122 |
| low | low | 64 | 147 |
| medium | medium | 64 | 137 |
| high | high | 64 | 134 |
| xhigh | xhigh | 64 | 151 |
| max | max | 64 | 169 |
| off | none | 64 | 197 |

All seven named values are directly supported, so no cross-ladder remap was
needed. Default requests transmitted `high`. `bogus` exited 1 without a request;
the factory raised `ModelConfigError`. The TUI rejected it with the allowed list.
These observations establish transport, not a monotonic reasoning-budget claim.

Requests transmitted `thinkingmachines/Inkling-2-Small:together`; usage attempts
identified upstream `together`. The only provider extension observed was
`reasoning_effort`. Standard switches were explained: streaming plus
`include_usage`, explicit probe `max_tokens`, and the ordinary parallel-tool
flag when tools were registered. No sampling override was introduced.

Schema output used the legacy `json_object` plus `schema` envelope. In the
shell/schema case, the first request had tools and no response format; the
follow-up had the schema envelope and no tools. This verifies D2's defer
recommendation through the harness rather than direct provider calls.

#### Trace and usage review

Reasoning appeared in `reasoning_content` before content or tool calls. Tool
calls used function type, index 0, a stable ID and one nonempty argument
fragment in the inspected manual loops. IDs matched the follow-up assistant
call and tool result; no raw IDs are stored in evidence. Finish reasons were
`tool_calls` and `stop`; final usage arrived separately. There were no error
chunks or empty non-usage JSON rows. Normalized capture does not retain raw
SSE keep-alives, so their absence on the HTTP stream is not claimed.

Follow-up tool and resumed-history requests preserved the exact accumulated
reasoning in `reasoning_content`. Visible content was also compared to joined
raw chunks, including the overlong broad-question WebP response: no harness
stream duplication was found. No new stream shape required a new replay test.
The existing Inkling raw-reasoning replay test and route tests passed:
**9 passed, 95 deselected**.

For the text turn and both two-request shell loops, all five attempts matched
provider input, output, reasoning and cached tokens. Reported cached zeros
were retained; unreported uncached/cache-write/tool-use/cost/context-token
fields remained null. Effective context was **1,048,576**, confirmed by suite
accounting and `/model status`.

After review, all request/chunk traces under the scratch directory, including
the relocated structured-matrix traces described below, were deleted.
`find . -name stream-debug -not -path './.venv/*'` returned nothing.

#### Media observations

The suite and existing e2e tests use the precise question “What is the user
name in this image?” The CLI returned the expected username for **PNG, JPEG
and WebP** with that question; all transmitted `image_url` parts.

The playbook's broader “What word is written in this image?” prompt did not
return the expected username on the three encodings; the fixture contains
several words. One WebP answer was markedly overlong, but completed with
provider `stop` and matched raw streamed content. These observations are
retained rather than converted into positive results. The precise username
checks, not the ambiguous prompt, support the capability finding.

PDF is not advertised. Its bytes were absent from outgoing requests. A normal
display run visibly reported removal of an unsupported application/pdf
document. The earlier `logger.type: none` scratch config hid that display
message by design. The suite also exercised an opaque unsupported MIME.
Supplemental existing e2e media tests: **2 passed, 2 skipped** (the two
unadvertised PDF cases), 8 deselected.

#### Interactive controls

The live picker showed **Inkling 2 Small (together)** with the exact route.
`/model status` showed schema output, two-phase structured tools, 1,048,576
context and 128,000 nominal output. `/model reasoning` accepted every named
effort and `off`, and rejected `bogus` with the allowed list.

Starting from `off`, F6 recovered to `none`, then visited
`none → minimal → low → medium → high → xhigh → max → none`.
This is the seven-value ladder, not a separate hard-disabled state.
Live glyphs and command-reported states were checked. Because `hc read`
strips colour, colour distinctness was additionally verified with the actual
renderer: inactive full block; green low gauges; green/yellow medium/high;
yellow/red xhigh/max. All seven rendered styles were distinct.
The 106 prescribed interactive unit tests passed. The probe terminal was
killed and removed.

**Playbook correction:** `hc send` pastes before Enter. Two attempted slash
commands through it unexpectedly invoked inference in this TUI; root cause
was not established. Only slash commands were sent, never an arbitrary
prompt. Sending the same commands as individual characters with `hc keys`
worked, and all remaining commands used that method. Those two incidental
calls are included in spend. Do not treat a successful `hc send` as evidence
that a slash command was executed.

#### Suite defects, edits and reruns

All edits were disclosed to and approved by the supervisor:

1. **`tests/e2e/onboarding/conftest.py`:** the original workspace fixture
   hardcoded a repository `.evidence/run-*` directory for raw traces. Changed
   it to system `TemporaryDirectory`; `TMPDIR` selects the permitted scratch
   parent. No model behavior was patched.
2. **`tests/e2e/onboarding/test_conformance.py`:** replaced the incorrect
   universal hard-off assertion with the recorded
   `reasoning_observed_when_disabled` observation and a warning. Correct
   answer, normal completion and wire capture checks remain. Hard/soft
   disable semantics belong to the provider contract. The initial
   `none` failure remains in initial evidence; final reasoning is **warn**,
   explicitly citing D4.
3. **`tests/e2e/onboarding/README.md`:** documented both changes.
4. **`tests/e2e/structured/test_current_model_structured_outputs.py`:**
   added `inkling` to the live structured matrix as required by the playbook.

The initial full suite was **18 passed, 1 failed** (the hard-off assertion).
The corrected full suite was **18 passed, 1 failed**, with all automated
conformance groups pass/warn and the failure solely in **PTC uptake**:
two nested `lookup_value` calls succeeded, tool errors were zero and the
final completion was `endTurn`, but its final integer was not the expected
sum. The final suite exit code is therefore **1**, not a claimed green
whole-suite run. See `ptc_handoff` and both suite evidence files.

Two further existing suite/CLI defects were reported, not silently patched:

- The shared existing structured fixture parses process CLI arguments.
  `pytest -m e2e` became fast-agent's message mode, causing `SystemExit(0)`
  in both selected tests. Running `pytest.main([...])` with
  `sys.argv = ['pytest']` avoids that collision; **2 passed, 32 deselected**.
- `check structured --json` emitted chat display before its final JSON
  array. The final array was parsed; direct, Pydantic and tools all passed.

**Trace-handling incident:** the initial structured-matrix invocation set
`FAST_AGENT_LLM_TRACE=0`, but any nonempty value enables capture. Because its
existing fixture changes cwd, six raw trace files briefly landed in
`tests/e2e/structured/stream-debug`. They were immediately moved into the
permitted scratch tree, never committed, and subsequently deleted after
review. The rerun **unset** the variable. No raw traces remain in the checkout.

#### Reproduction

Use the worktree under test and a credential configured through the normal
provider mechanism; do not print credentials.

```bash
W=/home/ssmith/source/fast-agent-onboarding
C=/tmp/onb-inkling-2-small/conformance
mkdir -p "$C"
cd "$W"
TMPDIR="$C" FAST_AGENT_RETRIES=0 ONBOARD_MODEL=inkling ONBOARD_TRACK=full \
  ONBOARD_EVIDENCE=onboarding/models/inkling-2-small/evidence/conformance-suite.json \
  uv run pytest tests/e2e/onboarding -m e2e -q -rs

# Manual CLI probes run from scratch, with --results r.json and tracing enabled.
cd "$C"
FAST_AGENT_LLM_TRACE=1 uv run --project "$W" fast-agent go --no-home \
  --model 'inkling?max_tokens=8192' --message 'Reply with exactly: ONB-TEXT-OK' --results text.json
for e in none minimal low medium high xhigh max off; do
  mkdir -p "eff-$e"
  (cd "eff-$e"; FAST_AGENT_LLM_TRACE=1 uv run --project "$W" fast-agent go --no-home \
    --model "inkling?reasoning=$e&max_tokens=8192" \
    --message 'How many positive divisors does 7560 have? Reply with only the number.' --results r.json)
done
uv run --project "$W" fast-agent check structured --models 'inkling?max_tokens=8192' --mode all --json
TEST_MODEL=inkling uv run --project "$W" pytest "$W/tests/e2e/llm/test_llm_e2e.py" \
  -m e2e -k 'vision or pdf or mcp_tool_result' -q -rs

# Existing matrix workaround: do not let FastAgent parse pytest's -m.
unset FAST_AGENT_LLM_TRACE
uv run --project "$W" python -c 'import sys, pytest; sys.argv = ["pytest"]; raise SystemExit(pytest.main(["/home/ssmith/source/fast-agent-onboarding/tests/e2e/structured/test_current_model_structured_outputs.py", "-m", "e2e", "-k", "inkling", "-q", "-rs"]))'
```

The full manual command sequences are retained in scratch `manual.sh`,
`manual-more.sh`, `media-formats.sh`, `media-recheck.sh`,
`media-visible.sh` and `structured-matrix-rerun.sh`. They include the shell,
schema, history and attachment arguments summarized in the matrix.

#### Spend and quality gate

Known usage-bearing manual calls: **35**, **17,073 input + 11,452 output**
tokens, estimated **$0.0165739**. Both suite runs total **55 attempts**,
estimated **$0.0097882**. The two incidental TUI calls cost an estimated
**$0.0008781**. A **$0.02 reserve** covers the untraced successful structured
matrix rerun. Total conformance estimated/reserved: **$0.0472402**.

Adding the contract's **$0.3309222** tracked/reserved amount gives
**$0.3781624**, below the authorized **$5** budget. This is token-rate
accounting ($0.30/M input, $1/M output), not verified account billing; prior
unpriced calls remain unknown rather than zero. See `spend` in manual evidence.

After code changes:

- `uv run scripts/format.py --check`: pass.
- `uv run scripts/lint.py`: pass.
- `uv run scripts/typecheck.py`: pass.
- `git diff --check`: pass.
- Interactive unit checks: 106 passed.
- Inkling route/replay unit checks: 9 passed.

No production integration code was changed. No campaigns, release actions,
commits or new agent workers were started. All probe terminals were removed.

## Programmatic tool calling

Report date: 2026-11-14T00:00:00; author: codexresponses.gpt-6-astra?reasoning=high; fast-agent: 380e9d75387dbe0bc81bcf78476b9189a2ee9559 (local wheel 0.10.54)

### Outcome

**Keep PTC off pending the matched campaign probes.** Checks: **4 pass, 1 warn,
0 fail, 2 skip**. The model is now **flagship**, so both benchmark panels are required.
The report remains **blocked awaiting supervisor launch**. Preparation is complete;
benchmarking has not started.

PTC works on all seven standard CLI evals, including images and concurrent subagents.
The corrected structured-result fixture passes 5/5 with actual nested function calls.
The original scalar-string fixture remains 1/5: that product boundary was not fixed.
The shell panel still misses the strict every-run nested-ledger criterion; its warning
disposition is an explicit supervisor-approved exception, not a claim that criterion passed.

The human authorized both matched panels and resolved D1–D10. D9 retains the off
default pending results; D10 authorizes the pinned campaign. The human supplied the
pricing catalog and token file, and the supervisor authorized four prepare/checks and
one wheel preflight per mode. Those are complete. Both benchmark checks are now
**skip — prepared, awaiting supervisor launch**. No worker campaign launch is permitted.

The corrected four-cohort plan (`fast-agent-model-lab/onboarding/models/inkling-2-small/plans/2026-11-14-ptc-bench-probe.md`) pins the real
bench-run environment, retries, deadlines, pricing and artifact; it records executed
preparation/preflights and exact unexecuted launch/dashboard commands. No production
runtime, pricing catalog, shared resources or user docs were manually changed.
The earlier supervisor-approved suite correction and its evidence remain below.

#### Preparation handoff — 2026-11-14

- Built from clean committed HEAD `380e9d75387dbe0bc81bcf78476b9189a2ee9559`:
  `uv build --wheel --out-dir /tmp/onb-inkling-2-small/ptc/wheel-380e9d753`.
  The checkout remained clean immediately after build.
- Original wheel:
  `/tmp/onb-inkling-2-small/ptc/wheel-380e9d753/fast_agent_mcp-0.10.54-py3-none-any.whl`.
- Copied with explicit supervisor permission to:
  `/home/ssmith/source/bench-run/state/wheels/fast-agent-380e9d753/fast_agent_mcp-0.10.54-py3-none-any.whl`.
- SHA-256, verified on both copies:
  `3b83acc61a099eee5e5581e2cea00b4aa38e72a6f3b840da1c4158a65a8a9cd8`.
  This unpublished local wheel is not equivalent to the same-version PyPI release.
- Prepared and checked IDs (each prepare and per-cohort check exited 0):
  - `tb21pr3-inkling2small-high-hftogether-fa380e9d753-ptcoff-001`
  - `tb21pr3-inkling2small-high-hftogether-fa380e9d753-ptcon-001`
  - `tb21sb8-inkling2small-high-hftogether-fa380e9d753-ptcoff-001`
  - `tb21sb8-inkling2small-high-hftogether-fa380e9d753-ptcon-001`
- Actual definitions: probe3-r1, 15 trials/arm, 5 slots; sandbox8-r1, 80 trials/arm,
  16 slots. Total 190 trials, sequential arms, HF `cpu-basic`, 6h agent / 8h sandbox,
  provider retries 3, whole-trial retries 0, no publication.
- Pricing supplied by the human:
  `state/pricing/card-packs-3debd26-pricing_catalog.json`,
  SHA-256 `cecca747e74274f4b41dcb54cf0785d912d1c49baf52c73b1f8e6d12841e6e67`,
  version `2026-11-14.1`, rule `inkling-2-small-together`.
  Rates per million tokens: input **$0.30**, cache reads **$0.03**, output **$1.00**.
  Each receipt pins these rates, rule and catalog hash; no pricing edit was made.
- `bin/bench prepare` registered only the four new cohorts as active; all four
  `bin/bench check runs/<id>` commands passed. Receipts match within each panel
  except `run_id`, `route` and `ptc`; configs are equal after normalizing those
  mode/job differences. Evidence records receipt, config and source-lock SHA-256.
- Human-provided auth file reference: `/home/ssmith/source/fast-agent/inf.token`.
  The token was read only into child-command `HF_TERMINAL_BENCH_INFERENCE_TOKEN`
  and mapped to `HF_TOKEN` for the preflights; never printed, copied, logged or persisted.
- **Exactly one wheel preflight per mode completed**, with provider retries 0:
  - off: CLI exit 0, `endTurn`, **authenticated-ok with marker mismatch**.
    The 147-character reply is a textual tool-call-like payload, not the requested
    marker. No tool executed. The supervisor accepted this result and instructed
    no further diagnostics; the non-sensitive reply is retained in benchmark evidence
    at the supervisor's request.
  - on: CLI exit 0, `endTurn`, **authenticated-ok**, exact `PREFLIGHT-OK` marker.
  Histories/logs remain under `/tmp/onb-inkling-2-small/ptc/campaign-preflight/`.
  These are the same two preflights for both panels, not separate per-cohort pairs.
  Numeric usage/cost is absent and remains null.
- Offline inspection confirmed `fast-agent-mcp` 0.10.54 installed from the staged wheel,
  not the editable checkout. The uv preflight environment used Python 3.15; this is
  not a validation of the cloud runtime, tool uptake or HF capacity.
- **No Harbor invocation, cloud probe, launch, resume, replacement or publication ran.**
  No other agent's runs, jobs, dashboards or terminals were modified. The supervisor
  still owns same-launch-shell readiness/capacity and spend gates.
- Provisional historical planning baseline: **$3.4–5.3 inference across four cohorts**,
  excluding preflights and unknown HF compute; **108–188 minutes sequential** plus
  setup/gates. These are cross-model analogies, not a measured Inkling forecast or a
  cap. The pinned rule is now confirmed; the campaign spend limit remains a supervisor/human gate;
  the $5 onboarding probe allowance is not silently extended.
- Preparation-handoff quality gates rerun: `uv run scripts/format.py --check`,
  `uv run scripts/lint.py`, and `uv run scripts/typecheck.py` all passed. No code changed.
- No new product decision was made; D9/D10 remain resolved. The supervisor accepted
  the off-mode preflight marker mismatch and instructed registration and completion
  without further diagnostics or worker launches.

Prior live eval/uptake evidence below was collected at the recorded earlier commits;
building and preflighting the new wheel does **not** constitute rerunning those evals.
The two text preflights establish authenticated inference, with the off-mode answer
mismatch explicitly retained; they do not establish benchmark performance.

#### Evidence

All live work selected `inkling` on
`hf.thinkingmachines/Inkling-2-Small:together`. CLI comparisons explicitly selected
`reasoning=high` and `ptc=on|off` plus the corresponding CLI flag. Exact conformance repeats
use the suite's explicit PTC override and catalog high; captured requests confirm high.

| Check | Evidence | Finding |
|---|---|---|
| evals | `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/ptc-evals.json` | PTC 7/7, direct 7/7 |
| uptake | `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/ptc-uptake.json` | Structured fixture 5/5; original 1/5, shell-ledger caveat and product-boundary reproduction retained |
| naming | `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/ptc-naming.json` | Same common names in live requests and offline runtime |
| bench-probe | `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/ptc-bench-probe.json` | Prepared/checked, awaiting supervisor launch; benchmark costs/results null |
| bench-sandbox | `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/ptc-bench-sandbox.json` | Flagship panel required; prepared/checked, awaiting supervisor launch; benchmark costs/results null |
| recommendation | `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/ptc-recommendation.json` | False per D9, pending matched results |
| applied | `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/ptc-applied.json` | Existing False; direct function/shell conformance and policy tests pass |

#### Evals

The unmodified `examples/ptc/evals.py` passed every PTC answer **and path** check:

- `orders`: Stark/Aperture; gathered multiple order calls in one sandbox invocation.
- `tool-image`: green; image carried in sandbox result.
- `tool-image-select`: yellow; only the selected image returned.
- `resource-image`: orange via `get_resource`.
- `attach-media`: blue via staged attachment.
- `subagents`: Paris/Tokyo via concurrent subagents in one script.
- `slash-direct`: media/orders; slash tool called directly as intended.

The same commands with `--no-ptc` passed all expected-answer checks. Direct results are
answer checks only; PTC-specific path checks were not applied to direct histories.
CLI histories/logs remain in `/tmp/onb-inkling-2-small/ptc/`; the stock PTC runner removes
its temporary histories on exit. Its seven pass records are captured in `evals-ptc.txt`.
This one panel is evidence of working paths, not reliability or equivalence.

#### Uptake: explain the fluctuation, do not erase it

##### Exact conformance repeats

The prior conformance runs were one pass and one final-sum assertion failure
(initial (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/conformance-suite-initial.json`),
rerun (`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/conformance-suite.json`)).

Five invocations of the **original scalar-result fixture** produced:

| Sample | Correct sum assertion | Sandbox calls | Nested lookup calls | Tool errors |
|---|---:|---:|---:|---:|
| 1 | fail | 1 | 2 | 0 |
| 2 | fail | 1 | 2 | 0 |
| 3 | pass | 2 | 4 | 0 |
| 4 | fail | 1 | 2 | 0 |
| 5 | fail | 1 | 2 | 0 |

All ended normally. There was no missing code-tool uptake: each called the code tool
and executed both lookups. Failed answers contained concatenated values rather than 423.
All five observations are retained, not just the passing repeat.

##### Offline cause and paired diagnostic

The original `lookup_value(key: str) -> int` function is exposed as an `Any`-returning
PTC function. Its plain scalar text is **not** converted to a JSON scalar:
`src/fast_agent/ptc/values.py::_maybe_json` parses only text beginning with `{` or `[`.
The actual harness, called locally **without any provider request**, yielded:

- alpha's converted value: string `"137"`; no structured content;
- beta's converted value: string `"286"`;
- sandbox `a + b`: `"137286"`;
- sandbox `int(a) + int(b)`: `423`;
- two successful nested calls, no sandbox error.

This is a generic function-result boundary, not an Inkling-specific naming or transport
defect. No global conversion change was made: changing scalar interpretation is outside
this worker's model-onboarding scope.

A separate three-pair diagnostic kept the same harness/tool values, using direct tools
versus the original PTC prompt. Direct passed **3/3**. PTC also passed the suite-equivalent
423-presence assertion **3/3**, but **every** PTC sample initially added uncast values and
then corrected with `int(...)` in a later script (2, 2 and 3 sandbox calls respectively).
This explains fluctuation through model recovery; it does not supersede the exact 1/5
repeat result or establish a stable success rate. The direct wording naturally omits
the Python-code instruction, so this is a diagnostic, not a controlled benchmark.

##### Supervisor-approved structured-fixture correction

The prior worker then corrected only the PTC test fixture: FastMCP exposes the integer
as structured content `{"result": int}`, and its generated schema is supplied through
supported PTC `returns` metadata. The model sees a `dict[str, Any]` return and
`Returns: {result: int}`. The prompt and final sum assertion are unchanged; ordinary
direct-function tests still use the text-only decorator. The test now records exposed
stubs for inspection.

All **five corrected repeats passed**, each with one `exec_sandbox`, two nested
`lookup_value` calls and zero tool errors. These are separate observations, not replacements
for the original five samples. See `corrected_fixture_repeats`, `suite_change` and
`product_finding` in `fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/ptc-uptake.json`.
The supervisor clarified that model uptake should be judged by real calls from code and
CLI eval paths, with scalar-text coercion reported as a separate product-boundary finding.
No production scalar conversion was changed, and these probes do not establish equivalence.

##### Three shell tasks

| Run | Sandbox calls | Nested ledger | Sandbox errors / caveat |
|---|---:|---|---|
| 1 | 3 | bash ×3 | none |
| 2 | 5 | bash ×1 | unsupported `subprocess` import, recovered |
| 3 | 4 | none | unsupported `subprocess` import; used filesystem pass-through |

All three reported the correct three Python files and top line counts (308, 114, 53).
No run attempted to call nested-only tools directly. Run 3 nevertheless fails the
playbook's requirement for at least one nested call in **every** run.

Important qualification: run 3 used `os.walk` and `open`, which are real host-filesystem
pass-through operations, not entries in `fast-agent-ptc-calls`. As documented in
`examples/ptc/README.md`, that ledger is not a complete filesystem audit. Its empty
ledger is **not** proof the model did no work. Under the supervisor's clarified
interpretation, this is a **warning**, alongside recovered unsupported imports, rather
than a model uptake failure. The strict playbook ledger condition remains unmet and is
not silently reclassified as a pass. PTC stays off pending the matched campaign.

#### Naming

Two traced CLI requests (PTC on/off) matched the offline appendix render from
`onboarding/notes/ptc-function-naming.md`:

- PTC outer tool: `exec_sandbox`.
- PTC nested functions and direct-mode tools:
  `bash`, `process`, `read_text_file`, `write_text_file`, `edit_file`, `attach_media`.
- PTC-only state helpers: `store`, `load`, `stored` (helpers, not nested tool dispatches).
- No `_tool` mangling or model-specific shell name.

`bash` is the common current fallback, not the proposal's future canonical `shell`
vocabulary. No rename was introduced. Inspected raw naming traces were deleted;
only whitelisted names/model/effort observations were retained.

#### Default and checks

The catalog still inherits `ptc_enabled=False`. No additional catalog mapping or new test
literal was needed for an unchanged default. After the PTC investigation:

- Direct function-tool checks (single and parallel) plus shell-tool check: **3 passed**.
- Corrected structured PTC fixture: **5/5 passed**, with nested calls in every sample.
- Initial PTC policy selection: **75 passed**, 8 dependency deprecation warnings.
- Final related unit selection (`ptc/test_ptc_policy.py`, `ptc/test_ptc_runtime.py`,
  `tools/test_function_tool_loader.py`): **221 passed**, 8 warnings.
- `uv run scripts/format.py --check`: pass.
- `uv run scripts/lint.py`: pass.
- `uv run scripts/typecheck.py`: pass.

The original PTC assertion failures remain recorded separately; these targeted reruns
are not presented as a green full-conformance rerun. The completion audit independently
reran format, lint and typecheck on 2026-11-14; all passed.

#### Reproduction and retention

```bash
W=/home/ssmith/source/fast-agent-onboarding
O=/tmp/onb-inkling-2-small/ptc
cd "$W/examples/ptc"
TMPDIR="$O" FAST_AGENT_RETRIES=0 uv run evals.py \
  --model 'inkling?reasoning=high&ptc=on' --jobs 3 > "$O/evals-ptc.txt" 2>&1

# Scratch runner uses evals.command/CASES, replacing --ptc with --no-ptc;
# then runs the three prescribed shell tasks. Histories stay under O.
cd "$W"
uv run python "$O/run-cli.py"

# Current corrected fixture; original scalar repeats remain in repeat-N.json/log.
# Do not overwrite those historical observations when reproducing.
for n in 1 2 3 4 5; do
  TMPDIR="$O" FAST_AGENT_RETRIES=0 ONBOARD_MODEL=inkling ONBOARD_TRACK=full \
    ONBOARD_EVIDENCE="$O/corrected-repeat-$n.json" \
    uv run pytest tests/e2e/onboarding/test_conformance.py::test_ptc -m e2e -q \
    > "$O/corrected-repeat-$n.log" 2>&1
done
env -u FAST_AGENT_LLM_TRACE uv run python "$O/offline-scalar.py"
env -u FAST_AGENT_LLM_TRACE FAST_AGENT_RETRIES=0 uv run python "$O/diagnose.py"

TMPDIR="$O" FAST_AGENT_RETRIES=0 ONBOARD_MODEL=inkling ONBOARD_TRACK=full \
  ONBOARD_EVIDENCE="$O/applied-conformance-final.json" \
  uv run pytest tests/e2e/onboarding/test_conformance.py::test_function_tool \
    tests/e2e/onboarding/test_conformance.py::test_shell_tool -m e2e -q
uv run pytest tests/unit/fast_agent/ptc/test_ptc_policy.py \
  tests/unit/fast_agent/ptc/test_ptc_runtime.py \
  tests/unit/fast_agent/tools/test_function_tool_loader.py -q
```

Raw responses, histories and logs stayed in the permitted scratch tree; no credentials
were printed, copied or stored. Suite-generated traces were temporary and removed with
their workspaces; naming traces were explicitly deleted after extraction.

Known recorded usage for repeats, paired diagnostics and the applied rerun totals
**51,408 prompt / 5,671 completion tokens**, estimated **$0.0210934** using uncached
vendor input pricing. CLI eval/shell/naming histories lack usage totals, so their
cost remains **null**, not zero; this is not a complete billed-spend claim.
Earlier-stage spend is in `evidence/spend.json`. The two new wheel preflights lack
numeric usage and their cost remains null; this is not zero spend. No benchmark
campaign was launched.

#### Follow-up

A reviewer should rerun one CLI eval per mode and summarize a retained shell history,
including the pass-through-versus-nested distinction. Preserve both original and corrected
fixture observations and the explicit supervisor-approved warning disposition.
Revisit the scalar-result boundary
in a separately scoped runtime task, and hand the prepared matched probes to the supervisor, who owns the remaining
launch-shell readiness/capacity and budget gates in the corrected plan. Small probes cannot justify enabling PTC by default.

## Documentation and product surfaces

Report date: 2026-11-14T00:00:00; author: copilot.claude-opus-5.5; fast-agent: 084f004a8 (0.10.54, uncommitted surfaces edits)

### Outcome

User-facing surfaces now describe Inkling 2 Small as integrated on
`hf.thinkingmachines/Inkling-2-Small:together` through the `inkling` alias.
This stage made no live calls. It documents what the contract, integration and
conformance stages established. Where wording depends on open decisions, it
follows the recommendations and is **provisional**: D1 (128,000 nominal output),
D2 (legacy schema + defer), D3 (default `high`), D4 (`none` exposed as a soft
minimum), D5 (alias + picker entry), D7 (preserve raw reasoning). D8 is not
mentioned on the page because the request-size budget is unchanged. PTC is
documented as **off by default**. The PTC stage is still running, so
`ptc-lists` is left for a follow-up run. Evidence:
`fast-agent-model-lab/onboarding/models/inkling-2-small/evidence/surfaces.json`.

Files changed by this stage:

- `docs/docs/models/providers/huggingface.md` (new section, after Muse Glimmer)
- `docs/docs/_generated/model_aliases_hf.md`, `docs/docs/_generated/models_reference.md` (regenerated)
- `onboarding/models/inkling-2-small/evidence/surfaces.json`, this report

#### provider-docs

The section is placed after "Muse Glimmer via Together" in
`docs/docs/models/providers/huggingface.md`. Each claim and its source:

| Claim | Source |
|---|---|
| `inkling` → `hf.thinkingmachines/Inkling-2-Small:together`, no sampling defaults, picker label `Inkling 2 Small (together)` | 03-integration aliases/selector; `check models hf` |
| Text + JPEG/PNG/WebP image input, text output | 03 model-database (`OPENAI_VISION`); 04 media (all three encodings pass) |
| PDF/audio/video unsupported; removed with a warning | 02 media (PDF, MP4, audio rejected); 04 media (visible PDF removal, opaque MIME) |
| 1,048,576 context | 02 context-limit; 04 usage (`/model status`, suite accounting) |
| No published output maximum; `max_tokens` omitted by default; nominal 128,000 budget; explicit `?max_tokens` sent | 01 limits; 02 output-limit; 03 route-profile; D1 |
| Top-level `reasoning_effort`, seven-value ladder, default `high`, sent unchanged | 03 route-profile; 04 reasoning-efforts table |
| `none` is a soft minimum; `off` sends `none` | 02 reasoning-ladder (warn); 03 deviation note; 04 D4 caveat; D4 |
| Reasoning content replayed unchanged | 03 route-profile (`preserve_raw_reasoning`); 04 trace review; D7 |
| Standard `json_schema` rejected; legacy `json_object` + `schema`; tool phase then schema-only final | 02 structured (warn); 04 structured (shell/schema request sequence); D2 |
| PTC off by default | `resolve_ptc_enabled()` → False; PTC stage pending |
| Only `:together` has a profile | 03 untested scope |

Caveats covered: soft `none`/`off` (reasoning-ladder warn, conformance warn),
no published output ceiling (output-limit warn), structured envelope that can
suppress tool calls (structured warn), and untested non-Together backends. The
context-limit warn concerns the router request-size budget (D8, unchanged) and
is already covered by the page's existing "Inline images and the request size
limit" section.

The dated note lists only what conformance verified: streaming, every
reasoning effort including `off`, shell and function tool loops, multi-turn
reasoning replay, JPEG/PNG/WebP image input, and schema output both direct and
after a tool call. PTC uptake is not claimed.

The docs do not mention the model on `docs/docs/models/index.md`. The
playbook default is to do that only for flagship models or a new provider, and
the class is mid (D6, provisional). No new provider page was needed.

#### generated-refs

```bash
uv run scripts/docs.py generate
git status --short docs/docs/_generated resources/shared
#  M docs/docs/_generated/model_aliases_hf.md
#  M docs/docs/_generated/models_reference.md
test ! -e docs/docs/_generated/_generation_warnings.md && echo no-warnings
# no-warnings
```

- `model_aliases_hf.md` adds `inkling` and the bare catalog key
  `thinkingmachines/inkling-2-small`. Other bare HF keys such as
  `zai-org/glm-5.3` appear the same way.
- `models_reference.md` adds the `inkling` row: Text, Vision; `json` (schema);
  effort ladder plus `off`.
- No `current_models_*` file changed. Those cover Responses, Codex, OpenAI and
  Meta routes, and this is an HF route.
- Nothing changed under `resources/shared`, and there was no unrelated drift.

#### defaults

- Provider default: `HUGGINGFACE_DEFAULT_MODEL = "moonshotai/Kimi-K3.1-Pro:together"`
  (`src/fast_agent/config.py:69`), unchanged. No decision moves it, and the
  playbook default is "No".
- Setup template: `examples/setup/fast-agent.yaml` names no HF default or
  current models, so it needed no change.
- Selector: `uv run fast-agent check models hf` lists `inkling` →
  `hf.thinkingmachines/Inkling-2-Small:together`. No entry was superseded, so
  the `--all` behaviour does not change.
- Examples: this is a new family with no predecessor alias, so there is nothing
  to repoint.

#### ptc-lists

Skipped: the PTC stage has not finished. Current resolution:

```text
inkling False | kimi True | glm53flash True | astra True | glimmer False
```

`inkling` appears in none of the copies (`docs/docs/models/index.md`,
`docs/docs/ref/config_file.md`, `docs/docs/ref/go_command.md`,
`examples/ptc/README.md`, `huggingface.md`, `moonshot.md`), so they are
consistent with it being off. Pre-existing drift, not changed here: the PTC
comment in `examples/setup/fast-agent.yaml` (line 122) still says only
GLM-5.3-Flash defaults on. The playbook pitfalls record this, and the parent
asked me not to touch PTC lists beyond what PTC-off requires. If the PTC stage
turns PTC on for Inkling, rerun this check: add Inkling to every copy listed
above, fix the setup-template comment, and change the PTC sentence in the new
huggingface.md section.

#### docs-build

`uv run scripts/docs.py build` exited 0. It checks social cards, regenerates the
references (no further drift) and runs `zensical build --strict`. The log is in
`/tmp/onb-inkling-2-small/surfaces-docs-build.log`.

`uv run scripts/format.py --check` also passes (2021 files already formatted).
This stage changed no code, so lint and typecheck were not rerun.

#### Not done

- No live calls, campaigns, commits or release actions.
- No decisions were raised. The playbook defaults applied: a section on the
  existing page, no overview mention for a mid-class model, and no change to the
  provider default.

#### Addendum 2026-11-14: default backend changed to DeepInfra

The maintainer chose DeepInfra as the default Inkling 2 Small backend
(PTC stays off). The evidence above is historical and unchanged; the surfaces
were updated as follows:

- `inkling` now resolves to `hf.thinkingmachines/Inkling-2-Small:deepinfra`.
  Together is pinned by the new `Inkling 2 Small (together)` alias, following
  the existing `<Model> (<backend>)` naming for HF backend variants (for
  example `GLM 5.2 (deepinfra)`); no short `-<backend>` alias exists.
- The HF picker entry (still the `inkling` alias) is labelled
  `Inkling 2 Small (deepinfra)`; there is no separate Together entry.
- Unit tests pin `inkling` to DeepInfra with the manual stream path and
  `Inkling 2 Small (together)` to Together with the SDK stream path; the
  Together request-contract tests use the Together alias, while the
  schema-envelope test constructs the provider with the full
  `thinkingmachines/Inkling-2-Small:together` model ID.
- `tests/e2e/structured` runs both `inkling` and `Inkling 2 Small (together)`.
- `huggingface.md` leads with DeepInfra as the maintainer-selected default,
  keeps Together as the pinned alternative and the un-pinned route note; the
  generated HF alias table was regenerated. The tb21-probe3 counts (DeepInfra
  30/30, Together 28/30) are stated only as observations from small, separate,
  non-matched probes (different wheels, scheduling and times; 15 trials per
  mode per backend), with no reliability or speed claim. PDF is described as
  unsupported by fast-agent for this model, not by the backend. Examples and the setup template do not name the alias.
- Live alias smoke (4 provider requests): `inkling` text and `--shell` tool
  turns were served by `deepinfra`; `Inkling 2 Small (together)` text was
  served by `together` (from `upstream_provider` in the usage channel).
- Strict docs build (re-verified after review of `24fa5ecd9`): the earlier
  `docs-build` pass above is historical. The integration pages then lacked
  social cards and had four broken relative links. `compile` now rewrites
  report links for publication, both cards are committed, the integration
  index is in the Models nav, and `uv run scripts/docs.py build --strict`
  exits 0.
