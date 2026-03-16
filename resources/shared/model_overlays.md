<ModelOverlays>
---

# Model Overlays

## Purpose

Model overlays are environment-local named model entries. They let you bind a short token such as
`qwen-local` to:

- a provider
- a wire model name
- a custom `base_url`
- auth settings
- request defaults
- local model metadata

Use overlays when you need stable local model names for self-hosted endpoints, alternate gateways,
or llama.cpp imports.

---

## Location

Overlays are loaded from the active environment directory:

- `model-overlays/*.yaml`
- `model-overlays.secrets.yaml`

With the default environment directory, that is usually:

- `.fast-agent/model-overlays/`
- `.fast-agent/model-overlays.secrets.yaml`

---

## Minimal manifest

```yaml
name: qwen-local
provider: openresponses
model: unsloth/Qwen3.5-9B-GGUF
connection:
  base_url: http://localhost:8080/v1
  auth: none
defaults:
  temperature: 0.8
  top_p: 0.95
  max_tokens: 2048
metadata:
  context_window: 75264
  max_output_tokens: 2048
picker:
  label: Qwen local
  description: Imported from llama.cpp
  current: true
```

---

## Top-level fields

- `name` — overlay token used in `--model`, `default_model`, agent cards, or model references.
- `provider` — fast-agent provider used to dispatch requests.
- `model` — wire model name sent to the backend.
- `connection` — endpoint and auth settings.
- `defaults` — request defaults applied unless explicitly overridden.
- `metadata` — local model metadata for picker/runtime display.
- `picker` — label/description/current/featured flags for selection UIs.

---

## Connection

`connection` supports:

- `base_url`
- `auth`: `none` | `env` | `secret_ref`
- `api_key_env`
- `secret_ref`
- `default_headers`

### Auth modes

No auth:

```yaml
connection:
  base_url: http://localhost:8080/v1
  auth: none
```

Environment variable auth:

```yaml
connection:
  base_url: https://gateway.example/v1
  auth: env
  api_key_env: LAB_MODEL_TOKEN
```

Secret reference auth:

```yaml
connection:
  base_url: https://gateway.example/v1
  auth: secret_ref
  secret_ref: lab-qwen
```

Companion secrets file entry:

```yaml
lab-qwen:
  api_key: your-secret-token
```

---

## Defaults

`defaults` maps to model-string-style runtime settings.

Common fields:

- `reasoning`
- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `max_tokens`
- `transport`
- `service_tier`
- `web_search`
- `web_fetch`

These are compiled into the resolved model selection and applied unless a run supplies an explicit override.

---

## Metadata

`metadata` is used for local model understanding and display.

Common fields:

- `context_window`
- `max_output_tokens`
- `tokenizes`
- `fast`

Use this for models that are not part of the built-in catalog or when local runtime limits differ from known defaults.

---

## Picker

`picker` controls model-picker presentation.

Common fields:

- `label`
- `description`
- `current`
- `featured`

---

## Usage

Once present, an overlay name can be used anywhere a model string is accepted:

- `fast-agent go --model qwen-local`
- `default_model: "qwen-local"`
- agent card `model: qwen-local`
- model reference target such as `$system.fast`

Example:

```yaml
model_references:
  system:
    fast: "qwen-local"
```

---

## llama.cpp import

`fast-agent model llamacpp` provides an interactive import flow, plus `list`, `preview`,
and `import` subcommands for llama.cpp-compatible servers.

Default runtime base URL:

- `http://localhost:8080/v1`

Shared options:

- `--env` to target a specific fast-agent environment directory
- `--url` or `--base-url` to point at the llama.cpp server
- `--auth`, `--api-key-env`, and `--secret-ref` to control persisted overlay auth and discovery auth
- `--name` to choose the overlay token explicitly

These shared options can be used on the interactive command or placed before a subcommand.

Endpoint behavior:

- root URLs are normalized to `/v1`
- model discovery uses `/v1/models`
- runtime metadata is read from `/props`

Subcommands:

- `fast-agent model llamacpp` opens the interactive picker and writes the selected overlay
- `fast-agent model llamacpp list` prints discovered models; add `--json` for machine-readable output
- `fast-agent model llamacpp preview <model-id>` prints the generated overlay YAML without writing files
- `fast-agent model llamacpp import <model-id>` writes the overlay; add `--json` for machine-readable output
- `fast-agent model llamacpp import <model-id> --start-now` writes the overlay and immediately launches `fast-agent go --model <overlay>`
- `fast-agent model llamacpp import <model-id> --start-now --with-shell` launches `fast-agent go -x --model <overlay>`

The generated overlay:

- uses `openresponses` as the provider
- stores the normalized `/v1` `base_url`
- records the selected auth mode
- copies discovered defaults such as `temperature`, `top_k`, `top_p`, `min_p`, and `max_tokens`
- records discovered metadata such as `context_window`, `max_output_tokens`, and `tokenizes`

---

</ModelOverlays>
