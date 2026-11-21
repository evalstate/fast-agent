---
title: llama.cpp
social:
  title: llama.cpp
  tagline: Connect fast-agent directly to a local llama.cpp server.
  description: Connect fast-agent directly to a local llama.cpp server with fast-agent local or saved model overlays.
  alt: fast-agent social card — llama.cpp
---

# llama.cpp

Start `llama-server` (or `llama server`) and run `fast-agent local`:

```bash
llama server -m ./model.gguf --mmproj ./mmproj.gguf
fast-agent local
```

`fast-agent local` (or `fast-agent --local`) probes `LLAMA_ARG_HOST`/`LLAMA_ARG_PORT`,
`http://localhost:9931/v1`, and `http://localhost:8080/v1` in order, selects the active
model, and connects over the OpenResponses `/v1/responses` API with no config file needed.

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../../assets/models/llamacpp-local.cast"
  data-fa-asciinema-cols="96"
  data-fa-asciinema-rows="24"
  data-fa-asciinema-poster="npt:0:04"
  data-fa-asciinema-speed="1"
  data-fa-asciinema-idle-time-limit="1.3"
  data-fa-asciinema-fit="width"
  data-fa-asciinema-autoplay="true"
>
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-terminal-theme="light">Light</button>
    <button type="button" data-fa-terminal-theme="dark">Dark</button>
  </div>
  <div data-fa-asciinema-target></div>
</div>

<!--
Cast asset:
- Source: docs/docs/assets/models/llamacpp-local.cast
- Replay locally: asciinema play docs/docs/assets/models/llamacpp-local.cast
-->

## Everyday usage

`fast-agent local` accepts the same flags as `fast-agent go`:

```bash
# One-shot prompt
fast-agent local -m "Summarize this repo"

# Multimodal input (works automatically when llama-server has --mmproj loaded)
fast-agent local --attach screenshot.png -m "What error is shown?"

# Target a specific server URL or router-mode model
fast-agent local --base-url http://localhost:8080 --model Qwen3.6-35B-A3B
```

On connect, fast-agent reads `/props` for live context length (`n_ctx`), loaded
modalities (`vision`, `audio`, `video`), and supported `reasoning_effort` levels. If the
model matches an entry in fast-agent's model database (for example `Inkling-2-Mini`,
`Qwen3.6-35B-A3B`, or `Gemma-4`), fast-agent applies its harness preferences while
keeping the live server's limits and capabilities authoritative.

In router mode with multiple models, fast-agent picks the single loaded model
automatically or opens the interactive picker without autoloading unloaded models while
browsing.

## Persisted and remote overlays

When you want a saved model name (for `--model <name>`, config files, or authenticated
remote endpoints), import an overlay:

```bash
# Interactive picker (choose "Start now" or write an overlay)
fast-agent model llamacpp

# Non-interactive discovery and import
fast-agent model llamacpp list --json
fast-agent model llamacpp import unsloth/Qwen3.6-35B-A3B-GGUF --name qwen-local
fast-agent go --model qwen-local
```

For remote or authenticated servers, pass `--url` with `--auth env` (`--api-key-env`) or
`--auth secret_ref` (`--secret-ref`):

```bash
fast-agent model llamacpp import \
  --url https://lab.example \
  unsloth/Qwen3.6-35B-A3B-GGUF \
  --name qwen-lab \
  --auth env \
  --api-key-env LLAMA_CPP_TOKEN
```

Overlays store discovered capability metadata and output limits without freezing the
server's sampling policy. Pass `--include-sampling-defaults` to persist `temperature`,
`top_k`, `top_p`, and `min_p`.

## See also

- [Model Overlays](../model_overlays.md) — overlay manifest format and precedence rules.
- [Command switches](../../ref/cmd_switches.md) — CLI flags and examples.
