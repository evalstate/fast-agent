# Reflection Beam 1.0 — integration report

What fast-agent verified when adding Reflection Beam 1.0. The full onboarding record and its evidence are in the fast-agent-model-lab repository.

## Model card

```yaml
name: Reflection Beam 1.0
vendor: Reflection
id: reflection/Beam-1.0-501B-A23B
route: hf.reflection/Beam-1.0-501B-A23B:together
class: mid
source: https://huggingface.co/reflection/Beam-1.0-501B-A23B
```

## Wire contract — `hf.reflection/Beam-1.0-501B-A23B:nebius`

Tested 2026-11-15.

| Check | Result | Finding |
| --- | --- | --- |
| Each effort accepted, with observable effect; default and off behaviour | ⚠️ caveat | minimal..max and omission accepted, 2/2 correct on the divisor prompt; harder prompt 3/3 except medium 2/3; effort reaches the template (prompt-token grouping) but adjacent levels overlap (minimal+low median 2297 vs xhigh+max+omitted 3155 reasoning tokens); none and every OpenRouter disable form 400 mandatory; bogus 400; chat_template_kwargs ignored |
| Reasoning field, streaming shape and replay requirement | ⚠️ caveat | delta/message reasoning + reasoning_details, reasoning first; reasoning_content, reasoning, reasoning+details and details-only replay all accepted and recalled 2/2; omitted or empty reasoning on ANY assistant turn (tool or plain) is 400 (allow_missing_thinking=false); fast-agent sends reasoning_content only for cataloged models |
| Default and maximum output tokens; truncation finish reason | ✅ pass | Hard ceiling 131072: omitted and explicit 200000 both stop at exactly 131072 (length); 131073..1000000 accepted silently; max_tokens and max_completion_tokens both truncate at 64 with no visible text; no 65,536 cap |
| Enforced context window and request size limits | ⚠️ caveat | 1048576 enforced by Nebius (400 states 1153450 tokens); router body cap exactly 5,242,880 bytes (5,242,881 -> 413), not 4,718,592; OpenRouter pre-check reserves no output when max_tokens is omitted; full window not tested |
| Chunk shape, finish reasons, terminal usage | ✅ pass | stop/length/tool_calls, [DONE], usage in its own top-level chunk with or without include_usage; reasoning then content/tool_calls |
| Single, parallel and forced tool calls; streamed arguments | ✅ pass | Single, parallel (2 calls even with parallel_tool_calls false), required, named, named at minimal, tool_choice none, continuation with replayed reasoning all correct; standard continuation fragments, no nulls, arguments whole; plain bash(command) is replaced by the trained definition, fast-agent bash is not |
| JSON schema, JSON object, and schema with tools | ✅ pass | json_schema strict/non-strict/max/minimal valid 8/8; prompted json_object 2/2; legacy envelope not enforced (1/2); schema with tools calls the tool 4/4 and the final answer validates 2/2 |
| Each documented modality; unsupported types rejected cleanly | ✅ pass | Text-only: image (data, https, in tool message), video and audio get clean OpenRouter 404 "No endpoints found that support ..."; PDF answered correctly via OpenRouter file parser (+USD 0.002), not advertised (D9) |
| Accepted and rejected sampling parameters | ✅ pass | temperature, top_p, top_k, min_p, presence/frequency/repetition penalty, seed, stop, logprobs accepted; temperature 2.5 rejected by OpenRouter; vendor values 0.7/0.99 are generation defaults and eval settings |
| Usage fields and cached-token reporting | ✅ pass | OpenRouter usage with reasoning_tokens, cached_tokens, cache_write_tokens, cost and cost_details; 6749-token prefix cached 0, 0, then 6720 |
| Proposed model parameters and route profile | ✅ pass | reasoning_content catalog mode; TopLevelReasoningEffort minimal..max no disable; json_mode schema; omit max_tokens, max_output_tokens 131072; text only; case-sensitive wire id; identical for both upstreams |

## Wire contract — `hf.reflection/Beam-1.0-501B-A23B:together`

Tested 2026-11-15.

| Check | Result | Finding |
| --- | --- | --- |
| Each effort accepted, with observable effect; default and off behaviour | ⚠️ caveat | minimal..max and omission accepted, divisor 2/2 and digit-sum 3/3 for every effort; effort reaches the template (prompt-token grouping identical to Nebius); adjacent levels overlap (minimal+low median 2427 vs xhigh+max+omitted 3463); none and OpenRouter disable forms 400 mandatory; bogus 400; chat_template_kwargs ignored |
| Reasoning field, streaming shape and replay requirement | ⚠️ caveat | reasoning + reasoning_details, reasoning first; four replay forms accepted and recalled 2/2; omitted/empty reasoning on any assistant turn 400 (template render error process_messages_failed); fast-agent replays only for cataloged models |
| Default and maximum output tokens; truncation finish reason | ✅ pass | Hard ceiling 131072: omitted and explicit 200000 stop at exactly 131072 (length); integers 1-20000 omitted ran to 108967 tokens (no 65,536 cap); both limit fields truncate at 64 |
| Enforced context window and request size limits | ⚠️ caveat | 1048576 enforced by Together (400 context_length_exceeded, 1153450 tokens); 6 MiB 413 and 5 MiB accepted as on Nebius (router cap 5,242,880 B, bracketed on Nebius); full window not tested |
| Chunk shape, finish reasons, terminal usage | ✅ pass | stop/length/tool_calls, [DONE], top-level usage with or without include_usage |
| Single, parallel and forced tool calls; streamed arguments | ✅ pass | Single, parallel (flag ignored), required, named, named at minimal, none, replayed continuation correct; no explicit nulls; arguments in 1-2 fragments; plain bash(command) replaced by the trained definition (131 tokens, rule 0/5), fast-agent bash and run_shell rendered as sent |
| JSON schema, JSON object, and schema with tools | ✅ pass | json_schema strict/non-strict/max/minimal 8/8 valid; json_object 2/2; legacy envelope 0/2 (extra note key); schema with tools calls the tool 4/4, final valid 2/2 |
| Each documented modality; unsupported types rejected cleanly | ✅ pass | Image, tool-image, video and audio: clean OpenRouter 404; PDF answered correctly via OpenRouter file parser (+USD 0.002), not advertised |
| Accepted and rejected sampling parameters | ✅ pass | All nine parameters plus logprobs accepted; temperature 2.5 rejected by OpenRouter |
| Usage fields and cached-token reporting | ✅ pass | OpenRouter usage fields; 6749-token prefix cached 16, 6720, 6720 |
| Proposed model parameters and route profile | ✅ pass | Same profile as Nebius: one RouterProfileRule for backends nebius and together |

## Harness conformance — `hf.reflection/Beam-1.0-501B-A23B:together`

Tested 2026-11-16.

| Check | Result | Finding |
| --- | --- | --- |
| Streamed text turn | ✅ pass | Full suite 18/18 on beam (:together) at a430ee6d3 (code = 08e30a723); text-stream pass, reasoning_effort max on the wire (conformance-suite-together-release.json) |
| Every ladder value through the alias; effective wire value recorded | ⚠️ caveat | Suite 7/7 on :together: minimal..max sent unchanged, default max (conformance-suite-together-release.json); none/off -> minimal, warning log-only (D13): Nebius, 16 Nov; not rerun on Together |
| Tool call, execution, continuation, final answer | ✅ pass | Suite 3/3 on :together (single, two independent, shell); sentinel bash + process wait/list/stop 3/3 (conformance-suite-together-release.json, preference-lifecycle-together.json) |
| History and reasoning replay across turns | ⚠️ caveat | Suite history pass on :together; reasoning-less history ends in 400 "Provider returned error — Together: ... missing its reasoning", no account id; traceback remains (repoint-together.json) |
| Structured output validated against the schema | ✅ pass | Suite 2/2 and test_current_model_structured_outputs -k beam 2/2 on :together at a430ee6d3 (repoint-together.json) |
| Each supported modality understood; unsupported types handled | ✅ pass | Suite 3/3 on :together: image and PDF removed with a notice (text-only), unsupported MIME handled (conformance-suite-together-release.json) |
| Usage and context accounting | ⚠️ caveat | Suite usage pass on :together, context 1048576, cached tokens reported; provider usage.cost arrives but cost_usd stays null (conformance-suite-together-release.json) |
| Raw request and stream traces inspected | ⚠️ caveat | test_stop_sequence fails 5/5 on :together (D12 caveat); Together chunk shape traced at 2cb73fce9 (17/17), not re-traced since (repoint-together.json) |
| Picker, effort cycling and check commands | ✅ pass | Offline at 08e30a723: picker "Beam 1.0 (together)", beam -> :together, PTC off (release-head-offline.json); TUI walk-through Nebius, 16 Nov; not rerun on Together |

## Harness conformance — `hf.reflection/Beam-1.0-501B-A23B:together`

Tested 2026-11-16.

| Check | Result | Finding |
| --- | --- | --- |
| Streamed text turn | ✅ pass | beam -> :together at release head a430ee6d3: suite text-stream pass, reasoning_effort max on the wire; earlier repoint text turn ONB-TEXT-OK |
| Every ladder value through the alias; effective wire value recorded | ⚠️ caveat | Suite 7/7 (minimal..max and default max, sent unchanged); none/off map to minimal with a log-only warning (D13: kept log-only in 0.10.55) |
| Tool call, execution, continuation, final answer | ✅ pass | Suite 3/3 (single, two independent, shell); repoint --shell smoke ONB-SMOKE-OK; lifecycle sentinel bash run_in_background + process wait/list/stop 3/3 |
| History and reasoning replay across turns | ⚠️ caveat | Suite history pass; reasoning-less history now live on Together: 400 "Provider returned error — Together: ... missing its reasoning ...", no account id in output or log; shared fatal path still prints a traceback |
| Structured output validated against the schema | ✅ pass | Suite 2/2; test_current_model_structured_outputs -k beam 2/2 (schema-only, with-tools) |
| Each supported modality understood; unsupported types handled | ✅ pass | Suite 3/3: image and PDF removed with a notice (text-only), opaque MIME handled; e2e vision/PDF tests skip by supports_mime |
| Usage and context accounting | ⚠️ caveat | Suite usage pass, context 1048576; OpenRouter raw usage.cost present (sentinel 14 attempts) but cost_usd stays null |
| Raw request and stream traces inspected | ⚠️ caveat | Together chunk shape reviewed earlier at 2cb73fce9 (body provider Together 17/17), no new traces; stop sequences fail 5/5 on Together at a430ee6d3 (D12 caveat applies) |
| Picker, effort cycling and check commands | ✅ pass | Offline: check models huggingface lists beam -> :together; picker row "Beam 1.0 (together)"; PTC off; TUI walk-through not rerun (route-independent UI code) |

## Structured output — Wire contract

Evidence: `fast-agent-model-lab/onboarding/models/beam-1/evidence/contract-structured.json`.
Validated with `jsonschema` against the playbook schema.

| Mode | Valid |
|---|---|
| `json_schema` strict | 2/2 |
| `json_schema` non-strict | 2/2 |
| `json_schema` at `max` | 2/2 |
| `json_schema` at `minimal` | 2/2 |
| `json_object` with the schema in the prompt | 2/2 |
| legacy `{"type":"json_object","schema":…}` | **1/2**: one `{"answer": …}`; the schema is not enforced |
| `json_schema` + `get_weather` | tool called 2/2 (streamed) and 2/2 (two-step); final answer valid 2/2 |

`json_schema` adds no prompt tokens (58 with or without), so it is enforced
by constrained decoding, not rendered into the prompt. Schema with tools does
not fabricate: it calls the tool first, then returns valid JSON. Use
`json_mode="schema"` with no tool restriction, and not the legacy envelope.

## Structured output — Wire contract: together

`fast-agent-model-lab/onboarding/models/beam-1/evidence/contract-together-structured.json`

- `json_schema` strict, non-strict, at `max` and at `minimal`: 2/2 valid
  each.
- `json_object` with the schema in the prompt: 2/2.
- Legacy envelope: **0/2**. Both answers added a `note` key, so the schema is
  not enforced.
- Schema with tools: tool called 2/2 streamed and 2/2 in the two-step test,
  and the final answer validates 2/2.
