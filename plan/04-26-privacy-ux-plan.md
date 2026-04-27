# Privacy filter model UX plan

## Summary

Make privacy-filter model selection and caching easy for normal users while
keeping advanced users able to point at custom or fine-tuned OpenAI Privacy
Filter-compatible checkpoints.

The desired default command should be simple:

```bash
fast-agent export latest --privacy-filter
```

For first use, if the model is not cached, the user should get a clear message
with the exact command to download/cache it:

```bash
fast-agent export latest --privacy-filter --download-privacy-filter
```

For model choice, users should not need symlinks or knowledge of ONNX filenames.
They should be able to choose a variant or repo explicitly:

```bash
fast-agent export latest --privacy-filter --privacy-filter-variant q4f16
fast-agent export latest --privacy-filter --privacy-filter-variant q4
fast-agent export latest --privacy-filter --privacy-filter-model owner/custom-privacy-filter
```

The Hugging Face Hub cache should remain the cache of record. Do not introduce a
second fast-agent model cache for normal hub-based use.

## Goals

- Keep the common path one flag: `--privacy-filter`.
- Avoid surprising network downloads.
- Prefer the conservative validated `q4` model variant by default.
- Let users select model variants without filesystem symlink workarounds.
- Let users use custom or fine-tuned OpenAI Privacy Filter-compatible repos.
- Reuse Hugging Face Hub caching and revision pinning semantics.
- Keep error messages actionable and copy/pasteable.
- Preserve a clear security boundary: only validated OpenAI Privacy
  Filter-compatible model directories/repos are accepted.

## Non-goals

- Do not auto-download large model files without an explicit download/cache flag.
- Do not support arbitrary ONNX token-classification models as a generic backend.
- Do not add Torch or Transformers to the privacy export path.
- Do not require users to understand ONNX external data filenames.
- Do not invent a fast-agent-specific cache layout for Hugging Face models.

## Recommended defaults

Default backend:

```text
backend: onnxruntime
model repo: openai/privacy-filter
revision: 7ffa9a043d54d1be65afb281eddf0ffbe629385b
variant: q4
```

Rationale for `q4` as default:

- It is the quantized variant already validated as the conservative default in
  the original implementation plan.
- It is still laptop-friendly compared with `quantized`, `fp16`, and full ONNX
  variants.
- It avoids changing the product default based on limited local smoke testing.
- Users can still choose `q4f16` explicitly when they want the smallest observed
  artifact.

Supported v1 variants:

```python
VARIANT_FILES = {
    \"q4f16\": [
        \"onnx/model_q4f16.onnx\",
        \"onnx/model_q4f16.onnx_data\",
    ],
    \"q4\": [
        \"onnx/model_q4.onnx\",
        \"onnx/model_q4.onnx_data\",
    ],
    \"quantized\": [
        \"onnx/model_quantized.onnx\",
        \"onnx/model_quantized.onnx_data\",
    ],
}
```

Defer `fp16` and `full` until validated because they use larger/multiple external
data files and are less laptop-friendly.

## User-facing CLI

Recommended options:

```bash
--privacy-filter
--download-privacy-filter
--privacy-filter-model <repo-or-path>
--privacy-filter-revision <revision>
--privacy-filter-variant <q4f16|q4|quantized|auto>
--privacy-filter-path <path>              # alias/compatibility escape hatch
--show-redactions
```

### Option meanings

#### `--privacy-filter`

Enable privacy-filtered export.

If model files are cached, use them. If not cached, fail with a clear message
unless `--download-privacy-filter` is present.

#### `--download-privacy-filter`

Permit Hugging Face Hub download for the selected model/revision/variant.

This should be explicit because downloads can be close to 1 GB.

#### `--privacy-filter-model`

Accept either:

- a Hugging Face repo id, e.g. `openai/privacy-filter`
- a local directory path

Heuristic:

- If the value is an existing local path, treat it as local model directory.
- Otherwise treat it as a Hugging Face repo id.

This is more user-friendly than forcing users to remember separate `repo` and
`path` flags.

Examples:

```bash
fast-agent export latest --privacy-filter \\
  --privacy-filter-model openai/privacy-filter

fast-agent export latest --privacy-filter \\
  --privacy-filter-model my-org/privacy-filter-finetune \\
  --privacy-filter-revision main

fast-agent export latest --privacy-filter \\
  --privacy-filter-model ./models/privacy-filter
```

#### `--privacy-filter-path`

Keep as a backward-compatible explicit local-path alias.

If both `--privacy-filter-model` and `--privacy-filter-path` are set, reject the
combination and ask the user to choose one.

#### `--privacy-filter-revision`

Hub revision to resolve. Default remains the pinned OpenAI revision for
`openai/privacy-filter`. For custom repos, default can be `main` unless the user
passes an explicit revision.

Recommended behavior:

- If model is the default `openai/privacy-filter` and no revision is provided,
  use the pinned revision.
- If model is any other repo and no revision is provided, use `main`.

#### `--privacy-filter-variant`

Select a known ONNX artifact set.

Default:

```text
q4
```

Accepted:

```text
q4f16
q4
quantized
auto
```

`auto` behavior:

- Local path:
  - If exactly one supported variant exists, select it.
  - If multiple supported variants exist, fail and list variants with a suggested
    explicit `--privacy-filter-variant`.
  - If none exists, fail and list required filenames.
- Hub repo:
  - Use the default variant unless a variant is explicitly provided. Avoid doing
    extra broad repo inspection in the common path.

### How users discover available variants

Fast-agent has two separate concepts:

```text
model:   which OpenAI Privacy Filter-compatible checkpoint/repo/path
variant: which supported ONNX artifact set inside that model
```

For v1, fast-agent should advertise the variants it knows how to load:

```text
q4f16
q4
quantized
```

These names are not arbitrary quant names discovered from model metadata. They
are product-supported artifact layouts mapped to exact files:

```text
q4f16    -> onnx/model_q4f16.onnx + onnx/model_q4f16.onnx_data
q4       -> onnx/model_q4.onnx + onnx/model_q4.onnx_data
quantized -> onnx/model_quantized.onnx + onnx/model_quantized.onnx_data
```

For the default `openai/privacy-filter` repo, docs can list the variants that
have been validated. For custom Hub repos, fast-agent cannot promise a variant
exists until it checks the selected repo/revision. The check is cheap for a
selected variant because `snapshot_download(..., allow_patterns=...)` either
finds/downloads the exact files or fails.

Recommended UX:

- `--privacy-filter-variant q4` means “try to load the known q4 artifact files
  from this model.”
- If those files are not present, fail with an error that says the selected
  model/revision does not contain files for variant `q4`.
- For local paths, also scan for other known variants and suggest the ones found.
- For Hub repos, do not do broad repo inspection during the normal export path.
  If variant discovery is needed later, add an explicit inspection command or
  flag that may perform a metadata lookup, e.g.
  `fast-agent privacy-filter variants my-org/privacy-filter-finetune`.

Local directory variant discovery is deterministic:

1. For each supported variant, check whether all required files exist.
2. If the user selected a variant, load that variant's ONNX model path.
3. If the user selected `auto`:
   - exactly one complete supported variant found => use it
   - multiple complete supported variants found => fail and ask the user to pick
   - none found => fail and list required filenames

Example local directory:

```text
models/privacy-filter/
  config.json
  tokenizer.json
  tokenizer_config.json
  viterbi_calibration.json
  onnx/
    model_q4f16.onnx
    model_q4f16.onnx_data
```

With:

```bash
fast-agent export latest \
  --privacy-filter \
  --privacy-filter-model ./models/privacy-filter \
  --privacy-filter-variant auto
```

fast-agent selects `q4f16` and loads:

```text
./models/privacy-filter/onnx/model_q4f16.onnx
```

#### `--show-redactions`

Print detected original text spans to stderr for local debugging.

This must keep a strong warning because it can re-leak sensitive data to the
terminal/logs.

## Slash command parity

Support the same options for:

```text
/session export latest --privacy-filter
/session export latest --privacy-filter --privacy-filter-variant q4
/session export latest --privacy-filter --privacy-filter-model my-org/filter
```

If progress is emitted in CLI via stderr, slash commands should include progress
only when the command channel can surface it usefully. At minimum the final
outcome should include:

- model repo/path
- revision when known
- variant
- redaction summary

## Config-file support

Add a config section for defaults:

```yaml
privacy_filter:
  model: openai/privacy-filter
  revision: 7ffa9a043d54d1be65afb281eddf0ffbe629385b
  variant: q4
  max_window_tokens: 1024
  window_overlap_tokens: 32
  intra_op_threads: 1
  inter_op_threads: 1
```

CLI flags override config.

Environment variables remain supported as low-level overrides for benchmarking
and operational tuning:

```bash
FAST_AGENT_PRIVACY_FILTER_MAX_WINDOW_TOKENS
FAST_AGENT_PRIVACY_FILTER_WINDOW_OVERLAP_TOKENS
FAST_AGENT_PRIVACY_FILTER_INTRA_OP_THREADS
FAST_AGENT_PRIVACY_FILTER_INTER_OP_THREADS
FAST_AGENT_PRIVACY_FILTER_DEVICE=auto|cpu|cuda
FAST_AGENT_PRIVACY_FILTER_CUDA_DEVICE_ID=0
```

For v1.1, config support can be limited to model/revision/variant and window
settings. Advanced ONNX Runtime knobs can remain env-only until needed.

Device selection should also be exposed as a normal user-facing option once the
ONNX CUDA path is validated:

```bash
--privacy-filter-device <auto|cpu|cuda>
```

Recommended default:

```text
auto
```

`auto` means use CUDA when the installed ONNX Runtime package exposes
`CUDAExecutionProvider`; otherwise use CPU. `cpu` forces
`CPUExecutionProvider`. `cuda` requires `CUDAExecutionProvider` and fails with an
actionable install/runtime error if it is unavailable.

## Model resolver behavior

Represent the selected model as a typed request:

```python
@dataclass(frozen=True, slots=True)
class PrivacyFilterModelRequest:
    model: str | None = None          # repo id or local path
    path: Path | None = None          # explicit local path alias
    revision: str | None = None
    variant: str = "q4"
    allow_download: bool = False
```

Resolve to:

```python
@dataclass(frozen=True, slots=True)
class PrivacyFilterModelResolution:
    model_dir: Path
    source: Literal[\"hub\", \"local\"]
    repo_id: str | None
    revision: str | None
    variant: str
    onnx_model_path: Path
```

Resolution order:

1. Validate optional dependencies before model resolution.
2. Parse `--privacy-filter-model`:
   - existing path => local
   - otherwise repo id
3. If `--privacy-filter-path` is set, use local path.
4. Determine revision:
   - default repo => pinned revision
   - custom repo => user revision or `main`
5. Determine variant:
   - explicit variant => use it
   - `auto` with local path => inspect local files
   - omitted => default `q4`
6. For local path:
   - validate `config.json`
   - validate tokenizer/calibration files
   - validate selected variant files
   - return exact ONNX model path
7. For Hub repo:
   - call `snapshot_download(..., local_files_only=True, allow_patterns=...)`
   - if cached, validate and return
   - if uncached and `allow_download=False`, fail with exact download command
   - if `allow_download=True`, download selected allow-list only

## Helpful errors

### Variant mismatch in local path

If a user points at a q4f16 directory but variant defaults to q4 or another
variant, do not say only “missing model_q4.onnx”. Say:

```text
Privacy filter model directory does not contain files for variant 'q4'.

Found files for variant: q4f16

Run with:
  --privacy-filter-variant q4f16
```

If multiple variants are present:

```text
Privacy filter model directory contains multiple variants: q4, q4f16.

Choose one:
  --privacy-filter-variant q4f16
```

### Uncached Hub model

Include selected model details:

```text
Privacy filter model is not cached.

Selected model:
  repo: openai/privacy-filter
  revision: 7ffa9a043d54d1be65afb281eddf0ffbe629385b
  variant: q4

Required download is approximately 900 MB.

Run again with:
  fast-agent export latest --privacy-filter --download-privacy-filter
```

For custom repo:

```text
Run again with:
  fast-agent export latest --privacy-filter \\
    --privacy-filter-model my-org/filter \\
    --privacy-filter-revision main \\
    --download-privacy-filter
```

## Download size hints

Keep approximate variant sizes in code for user messages:

```python
VARIANT_DOWNLOAD_SIZE_HINTS = {
    \"q4f16\": \"approximately 800 MB\",
    \"q4\": \"approximately 900 MB\",
    \"quantized\": \"approximately 1.6 GB\",
}
```

Do not attempt to calculate exact size before dependency checks. For Hub repos,
exact metadata lookup is optional; static hints are good enough and avoid extra
network behavior.

## Backend loading

The ONNX backend should not hard-code `model_q4.onnx`.

Pass the resolved ONNX path into the sanitizer:

```python
OpenAIPrivacyFilterOnnxSanitizer(
    model_dir=resolution.model_dir,
    onnx_model_path=resolution.onnx_model_path,
    model_info=PrivacyFilterModelInfo(...),
)
```

This avoids filename symlinks and makes variant support explicit.

### ONNX Runtime execution providers

Do not hard-code `providers=["CPUExecutionProvider"]`.

Select providers explicitly:

```python
def resolve_onnx_execution_providers(
    ort: Any,
    device: Literal["auto", "cpu", "cuda"],
    *,
    cuda_device_id: int = 0,
) -> list[str | tuple[str, dict[str, str]]]:
    available = set(ort.get_available_providers())
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return [
            ("CUDAExecutionProvider", {"device_id": str(cuda_device_id)}),
            "CPUExecutionProvider",
        ]
    if device == "cuda":
        raise SessionExportPrivacyFilterError(
            "CUDA was requested for the privacy filter, but ONNX Runtime does not "
            "expose CUDAExecutionProvider. Install a CUDA-enabled ONNX Runtime "
            "package such as `onnxruntime-gpu` with compatible CUDA/cuDNN libraries, "
            "or use `--privacy-filter-device cpu`."
        )
    return ["CPUExecutionProvider"]
```

Notes:

- Keep the default safe and portable: `auto` silently falls back to CPU when the
  installed ONNX Runtime package is CPU-only.
- `cuda` should be strict so benchmarking/CI does not accidentally run on CPU.
- Add an optional extra such as `privacy-cuda` only if packaging can avoid
  conflicting `onnxruntime` and `onnxruntime-gpu` installs. Otherwise document
  the manual install path for CUDA users.
- Include the selected provider in progress/metadata, e.g.
  `Using privacy filter: openai/privacy-filter @ ... (q4, cuda)`.

## Future backend: MLX

MLX should be a separate backend, not another `variant` in the ONNX resolver.
The model/quant vocabulary should stay:

```text
backend: onnxruntime | mlx
variant: backend-specific artifact set
```

Likely CLI shape:

```bash
--privacy-filter-backend onnxruntime|mlx|auto
--privacy-filter-variant q4        # ONNX example
--privacy-filter-variant mlx-4bit  # MLX example, exact names TBD
```

What MLX support would need:

1. A real OpenAI Privacy Filter-compatible MLX checkpoint, with the same
   tokenizer/config labels and calibrated Viterbi behavior as the ONNX path.
2. A backend-specific variant map for MLX artifact files, e.g. safetensors
   shards plus any MLX quantization metadata. Do not reuse ONNX filenames.
3. A `TraceSanitizer` implementation that returns the same logits contract used
   by the current Viterbi/redaction code:
   `[batch, sequence, labels]` aligned with `tokenizer.json` offsets.
4. Platform/dependency guards. MLX is primarily interesting on Apple Silicon;
   tests on non-Apple hosts should cover resolver/errors/contracts without
   importing or executing MLX.
5. Golden parity tests against the ONNX backend on a small text fixture before
   exposing MLX as more than experimental.

Recommended rollout:

- Land ONNX variant/device resolution first.
- Add backend metadata to `PrivacyFilterModelResolution` and trace export
  metadata.
- Add an experimental `mlx` resolver path behind explicit
  `--privacy-filter-backend mlx`.
- Do not make `auto` pick MLX until parity and installation behavior are proven.

## Progress output

During CLI export, emit:

```text
Checking privacy filter dependencies...
Resolving privacy filter model...
Using privacy filter: openai/privacy-filter @ ... (q4)
Loading privacy filter model...
Privacy filter model loaded.
Exporting session trace...
Privacy filter scanning 8,929 characters in 3 windows...
Privacy filter window 1/3 (33%)...
Privacy filter window 2/3 (67%)...
Privacy filter window 3/3 (100%)...
```

The model line should use local path wording for local models:

```text
Using privacy filter: /path/to/model (q4f16)
```

## Documentation updates

Update:

```text
docs/guides/privacy_filter.md
docs/ref/export_command.md
```

Key docs messages:

- `--privacy-filter` is the common path.
- Default model is `openai/privacy-filter`, variant `q4`.
- The Hub cache is used automatically.
- Downloads require `--download-privacy-filter`.
- Use `--privacy-filter-model` for custom/fine-tuned repos or local dirs.
- Use `--privacy-filter-variant` to switch variants.
- `--show-redactions` prints sensitive text to stderr and is local-debug only.

## Tests

### Model resolver

- default request selects `openai/privacy-filter`, pinned revision, `q4`
- q4f16 local directory validates without symlinks
- q4 local directory validates without symlinks
- variant mismatch reports found variants and suggested flag
- multiple local variants with `auto` reports ambiguity
- one local variant with `auto` selects that variant
- hub cache lookup uses selected variant allow-list
- uncached model error includes selected repo/revision/variant and download flag
- custom repo defaults revision to `main`
- default repo defaults revision to pinned revision

### CLI/slash parsing

- parse `--privacy-filter-model`
- parse `--privacy-filter-revision`
- parse `--privacy-filter-variant q4f16`
- reject `--privacy-filter-model` with `--privacy-filter-path`
- reject variant without `--privacy-filter`
- slash command passes the same request fields as CLI

### Backend

- sanitizer loads selected ONNX path rather than hard-coded q4 path
- model info metadata records repo/revision/variant/path
- trace metadata includes selected variant

### Integration/smoke

Default unit tests should not download real models.

Optional manual smoke:

```bash
uv run --extra privacy fast-agent export latest \\
  --privacy-filter \\
  --download-privacy-filter \\
  --output /tmp/privacy.jsonl
```

Variant smoke:

```bash
uv run --extra privacy fast-agent export latest \\
  --privacy-filter \\
  --privacy-filter-variant q4 \\
  --download-privacy-filter \\
  --output /tmp/privacy-q4.jsonl
```

Custom repo smoke should be manual unless a small fixture repo exists.

## Implementation readiness

This plan is ready to implement in stages for ONNX model/variant resolution and
CUDA provider selection.

### Ready for implementation

- Typed model request/resolution structs.
- ONNX variant map for `q4f16`, `q4`, and `quantized`.
- Local and Hub model resolution using Hugging Face Hub cache semantics.
- Passing the resolved ONNX model path into the sanitizer.
- CLI and slash-command parsing for:
  - `--privacy-filter-model`
  - `--privacy-filter-revision`
  - `--privacy-filter-variant`
  - `--privacy-filter-device`
- ONNX Runtime provider selection for `auto`, `cpu`, and `cuda`.
- Better progress and error messages.
- Unit tests with local fixture directories and a small resolver download seam or
  Hugging Face cache simulator. No real model download in default tests.

### Defer / explicit non-blockers

- Config-file defaults can follow the CLI/slash implementation unless needed for
  the first merge.
- Exact Hugging Face metadata size lookups are not needed; static size hints are
  sufficient.
- A `privacy-cuda` package extra should be deferred until dependency-resolution
  behavior is checked. The runtime code can support CUDA without the extra.
- MLX runtime support is not ready for production implementation until an actual
  OpenAI Privacy Filter-compatible MLX checkpoint and Apple Silicon smoke test
  path are available. Backend abstractions can be shaped now so MLX can be added
  later without reworking ONNX.

### Suggested implementation slices

1. Resolver/data-model slice:
   - add `PrivacyFilterModelRequest`
   - add `PrivacyFilterModelResolution`
   - return `onnx_model_path`
   - support local `auto` variant detection
   - improve local mismatch errors
2. Sanitizer slice:
   - accept `onnx_model_path`
   - accept resolved `PrivacyFilterModelInfo`
   - remove hard-coded `model_q4.onnx`
3. CLI/slash slice:
   - parse model/revision/variant/device flags
   - reject incompatible flag combinations
   - propagate selected fields to the handler
4. CUDA slice:
   - add provider resolver
   - add `auto|cpu|cuda`
   - include selected provider/device in progress/metadata
5. Docs/tests slice:
   - add resolver fixture tests
   - add provider-selection tests using a fake ONNX Runtime module object
   - update privacy/export docs
   - run manual smoke only when optional dependencies/models are available

### Merge gate

Before merging code changes:

```bash
uv run scripts/lint.py
uv run scripts/typecheck.py
uv run pytest tests/unit/fast_agent/privacy tests/unit/fast_agent/commands/test_session_export_handler.py
```

## Rollout

1. Add typed model request/resolution structs.
2. Add variant map and default `q4`.
3. Make resolver return selected ONNX path.
4. Update ONNX backend to load selected path.
5. Add CLI/slash flags for model/revision/variant.
6. Improve local-path mismatch errors.
7. Update progress line to show selected model.
8. Update docs.
9. Run manual q4f16/q4 smoke benchmarks.
