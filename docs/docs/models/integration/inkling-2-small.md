# Inkling 2 Small — integration report

What fast-agent verified when adding Inkling 2 Small. The full onboarding record and its evidence are in the fast-agent-model-lab repository.

## Model card

```yaml
name: Inkling 2 Small
vendor: Thinking Machines
id: thinkingmachines/Inkling-2-Small
route: hf.thinkingmachines/Inkling-2-Small:deepinfra
class: flagship
source: https://huggingface.co/thinkingmachines/Inkling-2-Small
```

## Wire contract

| Check | Result | Finding |
| --- | --- | --- |
| Each effort accepted, with observable effect; default and off behaviour | ⚠️ caveat | 66 valid-setting runs correct; adjacent efforts overlap, none remains soft, historical default outlier reproduces at 4x allowance |
| Reasoning field, streaming shape and replay requirement | ✅ pass | reasoning_content identified; shared replay succeeds with reasoning, reasoning_content and omission |
| Default and maximum output tokens; truncation finish reason | ⚠️ caveat | Omission generated 60014 tokens; both limit fields truncate at 64; separate output ceiling unknown |
| Enforced context window and request size limits | ⚠️ caveat | Input rejection confirms 1048576 window; recognized 4.6 MiB reaches Together, 5/6 MiB get router 413 |
| Chunk shape, finish reasons, terminal usage | ✅ pass | Standard finishes and DONE; terminal usage with include_usage, absent without it |
| Single, parallel and forced tool calls; streamed arguments | ✅ pass | Complete streamed get_weather matrix including forced none and real-call continuation; fragment shapes recorded |
| JSON schema, JSON object, and schema with tools | ⚠️ caveat | City schema validates via legacy envelope; standard schema rejected and legacy auto-tool cases skip the tool |
| Each documented modality; unsupported types rejected cleanly | ✅ pass | Required image fixture passes inline, HTTPS and tool role; sample PDF, MP4 and audio cleanly rejected |
| Accepted and rejected sampling parameters | ✅ pass | All nine parameters sent individually; ranges and recommendations recorded including null unknowns |
| Usage fields and cached-token reporting | ✅ pass | Usage fields recorded; corrected roughly 5K prefix with different final lines reports 4864 cached tokens on repeat |
| Proposed model parameters and route profile | ✅ pass | Evidence-to-field table complete; product recommendations provisional under D1-D4/D7-D8 |

## Harness conformance

| Check | Result | Finding |
| --- | --- | --- |
| Streamed text turn | ✅ pass | 1 pass |
| Every ladder value through the alias; effective wire value recorded | ⚠️ caveat | 7 pass, 1 warn |
| Tool call, execution, continuation, final answer | ✅ pass | 3 pass |
| History and reasoning replay across turns | ✅ pass | 1 pass |
| Structured output validated against the schema | ✅ pass | 2 pass |
| Each supported modality understood; unsupported types handled | ✅ pass | 3 pass |
| Usage and context accounting | ✅ pass | 1 pass |
| Raw request and stream traces inspected | ✅ pass | Requests/chunks inspected; route, reasoning order/replay, tool IDs/fragments, finishes and usage verified; raw traces deleted |
| Picker, effort cycling and check commands | ✅ pass | Picker, all seven efforts, off, invalid-value rejection and full F6 wrap verified; seven distinct renderer styles; 106 unit tests pass |

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
