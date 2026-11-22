---
title: Tencent Eval
social:
  title: Tencent Eval
  tagline: Configure Tencent Eval Hy4-flash models with reasoning streams, tools, structured output, and vision.
  description: Configure Tencent Eval (hy4-flash-dev) through the native OpenAI-compatible Chat Completions provider in fast-agent.
  alt: fast-agent social card — Tencent Eval provider
---

# Tencent Eval

Use the `tencent-eval` provider for Tencent's Hunyuan Hy4-flash evaluation Chat Completions API (`hy4-flash-dev`).

!!! note "Direct model selection only"
    `tencent-eval` is intentionally omitted from the interactive model picker. Select it directly with `--model tencent-eval.hy4-flash-dev`, `--model hy4-flash-dev`, or `--model tencent-eval`.

## Configure

Set either `TENCENT_EVAL_API_KEY` or `HY4_API_KEY`:

```bash
export TENCENT_EVAL_API_KEY="your-api-key"
# or Tencent's guide variable:
export HY4_API_KEY="your-api-key"
```

Or configure it in `fast-agent.secrets.yaml` / `fast-agent.yaml` (config file credentials take precedence over `TENCENT_EVAL_API_KEY`, which takes precedence over `HY4_API_KEY`):

```yaml
tencent-eval:
  api_key: "${TENCENT_EVAL_API_KEY}"
```

Optional provider settings:

```yaml
tencent-eval:
  api_key: "${TENCENT_EVAL_API_KEY}"
  base_url: "https://eval.hunyuan.tencent.com/openapi/v2/data"
  default_model: "hy4-flash-dev"
  # default_headers:
  #   X-Custom-Header: value
```

You can also override the endpoint via `TENCENT_EVAL_BASE_URL` when `tencent-eval.base_url` is not set in config.

## Models

| Model | Context | Maximum output | Input | Structured output |
|---|---:|---:|---|---|
| `hy4-flash-dev` | 960,000 | 65,536 | Text, images (PNG verified) | JSON Schema (`strict: true`) & JSON Object |

The `960,000`-token context limit is server-verified from endpoint input validation (`Input tokens exceed the configured limit of 960000 tokens`). `fast-agent` omits `max_tokens` on the wire by default unless explicitly set (verified up to `35,315` completion tokens on a single turn without `max_tokens`), and `GET /models` returns `404`.

Select the model with a provider-qualified ID, bare canonical ID, or provider default:

```bash
fast-agent go --model tencent-eval.hy4-flash-dev --message "Review this code."
fast-agent go --model hy4-flash-dev --message "Review this code."
fast-agent go --model tencent-eval --message "Review this code."
```

## Reasoning

`hy4-flash-dev` uses binary reasoning (`high` vs `no_think`), streams reasoning separately in `reasoning_content`, and defaults to `high`. Live sweeps confirmed that `no_think` and `none` disable reasoning while `low`, `medium`, `high`, `xhigh`, and `max` all enable reasoning without demonstrated ordered budgets, so `fast-agent` exposes binary `high` vs `none`/`off` (sent on the wire as `reasoning_effort: "no_think"`), with any other effort value falling back to `high`:

```bash
fast-agent go --model "tencent-eval.hy4-flash-dev?reasoning=high" --message "Solve this step by step."
fast-agent go --model "tencent-eval.hy4-flash-dev?reasoning=none" --message "Reply with only the final answer."
fast-agent go --model "tencent-eval.hy4-flash-dev?reasoning=off" --message "Reply with only the final answer."
```

`fast-agent` preserves raw `reasoning_content` across multi-turn tool-call replay for `hy4-flash-dev`.

## Capabilities and behavior

- **Streaming & usage**: Supports SSE streaming (`stream: true`) with `stream_options: {"include_usage": true}` and reports nested `completion_tokens_details.reasoning_tokens` (already included in `completion_tokens`).
- **Tool calling**: Supports single and parallel function calls in both streaming and non-streaming modes. Note that the evaluation endpoint may still return multiple tool calls even when `parallel_tool_calls: false` is sent.
- **Structured output**: Supports both `response_format: {"type": "json_schema", "json_schema": {..., "strict": true}}` and `response_format: {"type": "json_object"}`.
- **Vision**: Verified with inline base64 PNG data URLs (`image/png`).
- **Prompt caching**: Automatic server-side prefix caching (`usage.prompt_tokens_details.cached_tokens`) is verified in both non-streaming and multi-turn streaming sessions (62.8%–94.5% cache hits observed in TB2.1 runs) and is normalized by `fast-agent` when reported.
- **Model discovery**: The evaluation endpoint does not expose `GET /models`; use the static model ID `hy4-flash-dev`.
