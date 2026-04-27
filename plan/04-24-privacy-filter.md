# Privacy-filtered session export

## Summary

Add an optional privacy sanitization path to `fast-agent export` so exported
session traces can be redacted before they are written locally or uploaded to a
remote dataset.

Recommended backend:

- OpenAI Privacy Filter ONNX `q4` model
- ONNX Runtime for inference
- `tokenizers` for `tokenizer.json`
- a small NumPy port of OpenAI's official constrained BIOES Viterbi decoder

The initial scope should sanitize **content as it is exported**. In practical
terms, redact the text-bearing parts of the Codex export records generated from:

- session metadata base instructions / resolved prompt text
- user messages
- assistant messages
- assistant reasoning / thinking summaries
- tool calls
- tool results
- developer/system prompt content included in the export

The export should remain structurally valid after redaction and should include
trace metadata recording that privacy filtering was applied, which model was
used, the redaction mode, limitations, and span counts.

Implementation should be split into two phases:

1. **Export plumbing:** add the sanitizer interface, redaction summary metadata,
   optional Codex writer/exporter plumbing, and fake-sanitizer tests. Do not
   expose a user-facing privacy-filter flag until a real backend is wired.
2. **Backend and UI:** add the ONNX Runtime OpenAI Privacy Filter backend,
   model resolution/download behavior, CLI/slash flags, dependency checks, and
   user-facing warnings.

## Goals

- Prevent accidental PII/secrets leakage when users run `fast-agent export`,
  especially with `--hf-dataset`.
- Keep redaction local/offline by default.
- Avoid adding PyTorch or Transformers to the default fast-agent dependency set.
- Make the sanitizer an explicit opt-in feature.
- Redact exported content before any local JSONL write or remote upload.
- Include privacy-filter metadata and redaction counts in the exported trace.
- Preserve enough trace structure for downstream debugging/evaluation.

## Non-goals for v1

- Do not claim anonymization, compliance, or safety guarantees.
- Do not redact binary payloads or embedded base64 image/audio/file data in v1.
- Do not redact file paths, directory names, filenames, resource URLs, image
  URLs, MIME types, tool names, IDs, or other structural metadata in v1.
- Do not fine-tune the privacy model.
- Do not add full `transformers`/`torch` as base dependencies.
- Do not make every export sanitized by default until user experience and false
  positive behavior are better understood.

## Backend choice

Use ONNX Runtime rather than Transformers.

Rationale:

- `fast-agent` does not currently depend on `torch` or `transformers`.
- Export sanitization is an inference-only CLI operation.
- The OpenAI repo already ships quantized ONNX artifacts.
- The q4 ONNX model has been smoke-tested locally with ONNX Runtime on CPU.
- ONNX Runtime keeps the optional dependency footprint smaller and avoids a
  heavy PyTorch install.
- The official decoder logic can be ported independently of the PyTorch model
  runtime.

Suggested optional extra:

```toml
[project.optional-dependencies]
privacy = [
    "onnxruntime>=1.25",
    "tokenizers>=0.22",
    "numpy>=2",
]
```

`huggingface_hub` is already a dependency and can be reused to fetch the model
artifacts when needed.

## References and relevant code

External references:

- Model repo: <https://huggingface.co/openai/privacy-filter>
- Source repo: <https://github.com/openai/privacy-filter>
- Model card PDF:
  <https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf>
- Official decoder implementation:
  - `opf/_core/decoding.py`
  - `opf/_core/sequence_labeling.py`
  - `opf/_core/spans.py`

Local cloned reference used during planning:

```text
external/privacy-filter/opf/_core/decoding.py
external/privacy-filter/opf/_core/sequence_labeling.py
external/privacy-filter/opf/_core/spans.py
```

The local clone may not be present in all checkouts. When porting the decoder,
use the upstream source as the reference and update third-party attribution as
needed. This repository already has a `NOTICE` file for OpenAI-derived code
such as `apply_patch`.

fast-agent files likely involved:

```text
src/fast_agent/cli/commands/export.py
src/fast_agent/acp/slash/handlers/session.py
src/fast_agent/commands/handlers/session_export.py
src/fast_agent/commands/session_export_help.py
src/fast_agent/commands/shared_command_intents.py
src/fast_agent/ui/interactive/command_dispatch.py
src/fast_agent/session/trace_exporter.py
src/fast_agent/session/trace_export_models.py
src/fast_agent/session/trace_export_codex.py
src/fast_agent/session/trace_export_hf.py
```

Potential new files:

```text
src/fast_agent/privacy/__init__.py
src/fast_agent/privacy/dependencies.py
src/fast_agent/privacy/model_resolver.py
src/fast_agent/privacy/privacy_filter_onnx.py
src/fast_agent/privacy/sanitizer.py
src/fast_agent/privacy/viterbi.py
```

## Optional dependency management

Keep privacy filtering out of the default dependency set. The base
installation should not pull ONNX Runtime unless the user opts into privacy
filtering.

When `--privacy-filter` is used, check optional dependencies before resolving
or downloading model files. Report exactly what is missing and show the install
command.

Suggested helper:

```python
from importlib.util import find_spec

PRIVACY_EXTRA_REQUIREMENTS = {
    "onnxruntime": "onnxruntime",
    "tokenizers": "tokenizers",
    "numpy": "numpy",
}

def missing_privacy_dependencies() -> list[str]:
    return [
        package
        for module, package in PRIVACY_EXTRA_REQUIREMENTS.items()
        if find_spec(module) is None
    ]
```

Suggested error text:

```text
Privacy filtering requires optional dependencies that are not installed:
  - onnxruntime
  - tokenizers
  - numpy

Install them with:
  uv pip install "fast-agent-mcp[privacy]"

or:
  pip install "fast-agent-mcp[privacy]"
```

If the package name differs in the final distribution, use the actual project
install name in this message.

Dependency checks should happen before any network access. This avoids a bad
experience where the user downloads an 875 MB model and then learns ONNX
Runtime is not installed.

## Model artifacts

Default model:

```text
repo_id: openai/privacy-filter
revision: 7ffa9a043d54d1be65afb281eddf0ffbe629385b
variant: q4
```

This should be the default because it is the exact repo revision and ONNX
variant already smoke-tested locally with ONNX Runtime CPU.

Use only the files needed for ONNX q4 inference:

```text
config.json
tokenizer.json
tokenizer_config.json
viterbi_calibration.json
onnx/model_q4.onnx
onnx/model_q4.onnx_data
```

Use `huggingface_hub.snapshot_download(...)` from Python rather than shelling
out to `hf download`.

Suggested implementation:

```python
from huggingface_hub import snapshot_download

DEFAULT_PRIVACY_FILTER_REPO = "openai/privacy-filter"
DEFAULT_PRIVACY_FILTER_REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"
DEFAULT_PRIVACY_FILTER_VARIANT = "q4"

COMMON_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "viterbi_calibration.json",
]

VARIANT_FILES = {
    "q4": [
        "onnx/model_q4.onnx",
        "onnx/model_q4.onnx_data",
    ],
}

model_dir = snapshot_download(
    repo_id=DEFAULT_PRIVACY_FILTER_REPO,
    revision=DEFAULT_PRIVACY_FILTER_REVISION,
    allow_patterns=COMMON_FILES + VARIANT_FILES["q4"],
)
```

`snapshot_download(...)` automatically uses the Hugging Face Hub cache by
default. On Linux this is typically:

```text
~/.cache/huggingface/hub
```

On Windows/macOS it follows the Hugging Face cache conventions. It also respects
environment variables such as `HF_HOME` and `HF_HUB_CACHE`.

This means fast-agent does not need its own model cache for the default path.
It can let Hugging Face manage cached snapshots and simply keep an explicit
`--privacy-filter-path` escape hatch for users who want to manage files
themselves.

Do not download:

```text
model.safetensors
original/model.safetensors
onnx/model.onnx*
onnx/model_fp16.onnx*
onnx/model_q4f16.onnx*
onnx/model_quantized.onnx*
```

The q4 ONNX data file is large, roughly 875 MB, and the complete allowed
artifact set is close to 1 GB. V1 should avoid surprising implicit downloads in
non-interactive contexts.

Recommended behavior:

- If `--privacy-filter-path` is provided, load from that directory.
- If no model path is provided, resolve the pinned default from the Hugging Face
  cache with `snapshot_download`.
- Prefer trying `local_files_only=True` first in non-interactive contexts so an
  export does not unexpectedly start an 875 MB download.
- If files are not cached, show a clear message explaining the download size and
  how to fetch the default model.
- In non-interactive export, fail with an actionable error unless an explicit
  download/cache option is provided.
- If explicit download is allowed, call `snapshot_download` with the pinned repo,
  revision, and allow-list patterns above.

Potential internal helper:

```python
def resolve_privacy_filter_model_dir(
    *,
    model_path: Path | None,
    repo_id: str = DEFAULT_PRIVACY_FILTER_REPO,
    revision: str = DEFAULT_PRIVACY_FILTER_REVISION,
    variant: str = DEFAULT_PRIVACY_FILTER_VARIANT,
    allow_download: bool = False,
) -> Path:
    ...
```

Resolution order:

1. If `model_path` is set, validate and use it.
2. Try Hugging Face cache with `local_files_only=True`.
3. If missing and `allow_download=True`, download the pinned default files.
4. Otherwise fail with a clear command/help message.

Suggested uncached-model message:

```text
Privacy filter model is not cached.

The default model is:
  openai/privacy-filter @ 7ffa9a043d54d1be65afb281eddf0ffbe629385b, variant q4

Required download is approximately 1 GB.

Run again with:
  fast-agent export latest --privacy-filter --download-privacy-filter

or provide a local model directory:
  fast-agent export latest --privacy-filter --privacy-filter-path /path/to/model
```

For `/session export`, render the equivalent command using slash-command syntax.

## Overrides and variant support

The default should stay narrow and safe:

```text
openai/privacy-filter @ 7ffa9a043d54d1be65afb281eddf0ffbe629385b, q4
```

Advanced users may still need overrides for testing or future model updates.
Design the resolver so these are possible without making the CLI noisy:

```text
privacy_filter.repo_id
privacy_filter.revision
privacy_filter.variant
privacy_filter.path
privacy_filter.cache_dir
```

For v1, only `q4` needs to be officially supported. The code can still use an
explicit variant map so later variants are easy to add:

```python
VARIANT_FILES = {
    "q4": [
        "onnx/model_q4.onnx",
        "onnx/model_q4.onnx_data",
    ],
    # Future options, after validation:
    # "quantized": ["onnx/model_quantized.onnx", "onnx/model_quantized.onnx_data"],
    # "q4f16": ["onnx/model_q4f16.onnx", "onnx/model_q4f16.onnx_data"],
    # "fp16": ["onnx/model_fp16.onnx", "onnx/model_fp16.onnx_data", ...],
    # "full": ["onnx/model.onnx", "onnx/model.onnx_data", ...],
}
```

Do not accept arbitrary ONNX files as a generic backend. Accept a validated
OpenAI Privacy Filter model directory or a known repo/revision/variant
combination. The loader should validate:

- `config.json` exists and has `model_type == "openai_privacy_filter"`
- `tokenizer.json` exists
- `viterbi_calibration.json` exists, or zero biases are used
- the selected ONNX file exists
- required ONNX external data files exist
- ONNX Runtime inputs include `input_ids` and `attention_mask`
- ONNX Runtime output includes logits shaped like `[batch, sequence, labels]`
- `len(id2label)` matches the logits label dimension
- labels are the expected BIOES-style privacy labels

## User interface sketch (phase 2)

Possible CLI and slash-command flags:

```bash
fast-agent export latest --privacy-filter
fast-agent export latest --privacy-filter --output sanitized.jsonl
fast-agent export latest --privacy-filter --hf-dataset owner/name
fast-agent export latest --privacy-filter --privacy-filter-path ~/.cache/fast-agent/privacy-filter
fast-agent export latest --privacy-filter --download-privacy-filter
/session export latest --privacy-filter
/session export latest --privacy-filter --output sanitized.jsonl
/session export latest --privacy-filter --hf-dataset owner/name
```

Potential later options:

```bash
--privacy-filter-repo openai/privacy-filter
--privacy-filter-revision 7ffa9a043d54d1be65afb281eddf0ffbe629385b
--privacy-filter-variant q4
--privacy-filter-mode content-only|all-strings
--privacy-filter-placeholder typed|generic
--privacy-filter-dry-run
--privacy-filter-report path.json
```

Recommended v1 defaults:

- mode: `content-only`
- placeholders: typed labels such as `<PRIVATE_EMAIL>` and `<SECRET>`
- trim leading/trailing whitespace from detected spans
- sanitize before write/upload
- keep structural metadata unchanged, including IDs, tool names,
  model/provider fields, timestamps, cwd, and session metadata file
  paths/directories. Text blocks generated for messages/tool payloads may
  contain paths, filenames, resource URLs, or MIME types; v1 should still run
  the privacy filter over those text blocks as a whole rather than
  special-casing subfields.
- write privacy-filter metadata into the trace
- print a compact redaction summary after export

When `--privacy-filter` is enabled, emit a concise warning at the point where
the command surface can correctly do so. For CLI, stderr is preferred when the
message is emitted directly; for `/session export`, use the normal command
output channel.

```text
Warning: privacy filtering is best-effort and applies to exported text content
only. It can miss private data and can redact benign text. File paths, directory
names, resource URLs, filenames, binary payloads, images, audio, and base64/file
data are not redacted in this version. Review sanitized exports before sharing
or uploading them.
```

If `--hf-dataset` is also set, make the warning stronger because the export is
leaving the local machine:

```text
Warning: privacy filtering is best-effort and applies to exported text content
only. File paths, directory names, resource URLs, filenames, binary payloads,
images, audio, and base64/file data are not redacted in this version. Review
sanitized exports before uploading. Upload will use the sanitized JSONL file
only.
```

This mirrors the OpenAI model-card risk framing: the model is a redaction/data
minimization aid, not an anonymization, compliance, or safety guarantee.

After a successful privacy-filtered export, print a summary like:

```text
Privacy filter redacted 12 text span(s):
  private_person: 4
  private_email: 3
  private_phone: 1
  secret: 4
```

If no spans were redacted:

```text
Privacy filter redacted 0 text span(s).
```

For `--hf-dataset`, the export and upload may happen in a single command, but
the privacy summary is still an export concern. Print the summary as part of the
export result, then print the upload success message. The uploaded file must be
the same sanitized JSONL file and should contain the privacy-filter metadata.

```text
Privacy filter redacted 12 text span(s):
  private_person: 4
  private_email: 3
  secret: 5
```

Do not print the redacted source text or detected original span text in the
summary. Counts by label are enough and avoid re-leaking sensitive data.

For upload messaging, prefer inspecting the exported trace metadata or the
`ExportResult` redaction metadata:

- If privacy metadata is present, print a short best-effort reminder:

  ```text
  Uploaded privacy-filtered trace. Privacy filtering is best-effort; review
  shared traces for remaining sensitive data.
  ```

- If privacy metadata is absent, print:

  ```text
  No privacy filter applied. Visit https://fast-agent.ai/privacy-filter for more
  information.
  ```

A generic warning is acceptable if metadata inspection is not available on a
particular upload path, but the combined `export --hf-dataset` path should know
whether the just-written export contains privacy metadata.

## Export pipeline placement

Current high-level path:

```text
SessionTraceExporter.export(...)
  -> resolve session/history
  -> writer = CodexTraceWriter()
  -> writer.write(resolved, output_path)
  -> optionally upload output_path to HF
```

Add sanitization before writing and therefore before upload:

```text
SessionTraceExporter.export(...)
  -> resolve session/history
  -> build writer with optional sanitizer
  -> writer produces records
  -> sanitize record content
  -> write sanitized JSONL
  -> optionally upload sanitized JSONL to HF
```

Important: the uploaded file must be the sanitized file.

## Default output filename

When `--privacy-filter` is enabled and the user did **not** pass `--output`,
include a privacy suffix in the generated filename so sanitized exports are easy
to distinguish from normal exports.

Current default shape:

```text
<session_id>__<agent_name>__codex.jsonl
```

Privacy-filtered default shape:

```text
<session_id>__<agent_name>__codex-privacy.jsonl
```

If the user provides `--output`, respect it exactly. Do not rewrite explicit
paths.

This should be implemented in `SessionTraceExporter._resolve_output_path(...)`
using the privacy-filter request state. It also helps prevent a user from
mistaking an unsanitized and sanitized export when both are generated from the
same session.

## Sanitization strategy

The user-facing principle is: sanitize content as it gets exported.

In the Codex writer, that means applying the sanitizer to text-bearing payloads
created for:

1. Session metadata `base_instructions.text`
2. Developer/system prompt content
3. User messages
4. Assistant messages
5. Assistant reasoning/thinking summaries
6. Tool call arguments
7. Tool result output
8. Turn event summaries such as `message` and `last_agent_message`

Do **not** blindly redact all strings in v1. Preserve structural fields such as:

```text
type
role
id
turn_id
session_id
agent_name
model
provider
timestamp
created_at
call_id
tool/function name
status
phase
cwd
file paths
directory names
filenames
resource URLs
image URLs
MIME types
```

Blind all-string redaction can break trace consumers and can damage IDs, tool
names, model/provider fields, timestamps, cwd/session metadata paths, and other
schema-like metadata. A later `all-strings` mode can be added if needed. Text
content generated for message/tool payloads may still include filenames, URLs,
or wrappers; keep v1 simple and run the privacy filter over those exported text
blocks instead of trying to split every generated string into subfields.

## Content surfaces in `trace_export_codex.py`

Primary file:

```text
src/fast_agent/session/trace_export_codex.py
```

Known text-bearing construction points:

- `_session_meta_payload(resolved, meta)`
  - `base_instructions.text`
  - should also receive privacy-filter metadata after records are generated
- `_developer_message_item(system_prompt)`
  - `content[].text`
- `_text_item(text, output_text=...)`
  - `text`
- `_embedded_text_item(...)`
  - wraps embedded text resource contents inside a `<fastagent:file ...>` block
- `_reasoning_item(message)`
  - `summary[].text`
- `_function_call_item(call_id, call)`
  - `arguments`
- `_tool_result_output(result)`
  - plain string result, or item list with `input_text.text`
- `_user_event_payload(message)`
  - `message`
- `_turn_complete_payload(turn_id, last_agent_message)`
  - `last_agent_message`

The cleanest implementation is not a generic recursive sanitizer over all record
fields. Instead, thread a sanitizer into the writer and sanitize at these
text-bearing construction points where semantic intent is known. Keep this
deliberately simple: if a helper constructs a text content block for export, run
the privacy filter over that text block as a whole. Do not over-special-case
embedded file wrappers, attachment summaries, or argument strings in v1.

Example direction:

```python
class CodexTraceWriter:
    def __init__(self, sanitizer: TraceSanitizer | None = None) -> None:
        self._sanitizer = sanitizer
```

Helper:

```python
def _sanitize_text(sanitizer: TraceSanitizer | None, text: str) -> str:
    if sanitizer is None:
        return text
    return sanitizer.sanitize_text(text)
```

Then apply it before placing content into export records.

## Tool calls and tool results

Yes: include tool calls and tool results in the sanitization surface.

Reasoning:

- Tool call arguments may contain user-provided text, file paths, URLs, API
  payloads, email addresses, account IDs, or secrets.
- Tool results often contain the highest-risk data because they may include raw
  file contents, API responses, logs, search results, or command output.
- If exported traces are used for training/evaluation, unsanitized tool I/O can
  leak sensitive information even when user/assistant messages are redacted.

Recommended v1 handling:

- For function call `arguments`, sanitize the exported JSON argument string
  directly in v1. This keeps the implementation simple and matches the fact that
  the Codex trace field is itself a string.
- For tool result strings, redact directly.
- For tool result item lists, redact `input_text.text` fields.
- Preserve `call_id`, function/tool names, status, and item type fields.

Potential future improvement: parse JSON arguments, recursively sanitize only
string values, and re-serialize. That would better preserve structural keys if a
model ever over-redacts field names, but it also changes formatting/order and is
not required for v1. The v1 raw-string approach should still avoid printing or
uploading the unsanitized argument string.

## Reasoning / thinking

Include reasoning/thinking summaries in v1.

Reasoning text can quote user content, tool outputs, secrets, or private data.
The current exporter creates reasoning records via:

```text
_reasoning_item(message)
```

Sanitize:

```text
summary[].text
```

Do not remove the reasoning item entirely unless a future policy option asks for
that. Redacted reasoning is more useful than missing reasoning for trace review.

## User and assistant messages

Sanitize all text content created from user and assistant messages.

For normal text blocks:

```text
content[].text
```

For embedded text resources, sanitize the generated `<fastagent:file ...>` text
block as a whole. This may redact filenames or MIME-like text inside the wrapper,
but it is simpler and reduces the risk of missing sensitive content hidden in
generated text.

Do not attempt OCR or binary redaction in v1.

## Developer/system prompt

The exporter can include the resolved prompt as both session metadata
(`base_instructions.text`) and a developer message. It may contain private
deployment details, examples, internal URLs, or secrets.

Sanitize both `session_meta.payload.base_instructions.text` and developer
message `content[].text`.

This may slightly reduce reproducibility of the exported trace, but it matches
the safety goal of a privacy-filtered export.

## Sanitizer API sketch

Add a small internal abstraction rather than coupling the writer directly to
ONNX Runtime:

```python
@dataclass(frozen=True, slots=True)
class RedactionSpan:
    label: str
    start: int
    end: int
    text: str

class TraceSanitizer(Protocol):
    def sanitize_text(self, text: str) -> str: ...
    def detect_spans(self, text: str) -> list[RedactionSpan]: ...
```

Return redaction counts as part of the sanitizer/export result, either by
letting the sanitizer maintain a small summary object for the export run or by
having `sanitize_text` return both text and spans:

```python
@dataclass(slots=True)
class SanitizedText:
    text: str
    spans: tuple[RedactionSpan, ...]
```

The writer/exporter can aggregate:

```text
total_redactions
redactions_by_label
```

and include those numbers in CLI output and in the exported trace metadata. This
also makes tests simple because a fake sanitizer can return deterministic spans
and counts.

Suggested result metadata:

```python
@dataclass(frozen=True, slots=True)
class PrivacyFilterModelInfo:
    repo_id: str | None
    revision: str | None
    variant: str | None
    backend: str

@dataclass(frozen=True, slots=True)
class RedactionSummary:
    total: int
    by_label: dict[str, int]
    model: PrivacyFilterModelInfo | None = None
```

Add `redaction: RedactionSummary | None = None` to `ExportResult`. Preserve it
when `SessionTraceExporter.export(...)` reconstructs `ExportResult` after a Hugging
Face upload.

Suggested trace metadata shape on the first `session_meta` record:

```json
{
  "privacy_filter": {
    "applied": true,
    "mode": "content-only",
    "backend": "onnxruntime",
    "model": {
      "repo_id": "openai/privacy-filter",
      "revision": "7ffa9a043d54d1be65afb281eddf0ffbe629385b",
      "variant": "q4"
    },
    "redactions": {
      "total": 12,
      "by_label": {
        "private_person": 4,
        "private_email": 3,
        "secret": 5
      }
    },
    "limitations": [
      "file_paths_not_redacted",
      "directory_names_not_redacted",
      "filenames_not_redacted",
      "resource_urls_not_redacted",
      "binary_payloads_not_redacted",
      "images_audio_not_redacted"
    ]
  }
}
```

Because `CodexTraceWriter.write(...)` already builds records before writing,
metadata can be added or updated on the initial `session_meta` record after
sanitization counts are known and before the JSONL file is opened.

ONNX implementation:

```text
src/fast_agent/privacy/privacy_filter_onnx.py
src/fast_agent/privacy/viterbi.py
src/fast_agent/privacy/sanitizer.py
```

Pipeline:

```text
text
  -> tokenizers.Tokenizer.encode(text)
  -> input_ids + attention_mask + offsets
  -> onnxruntime.InferenceSession.run(...)
  -> log_softmax
  -> constrained BIOES Viterbi decode
  -> token spans
  -> char spans
  -> trim whitespace
  -> replace spans with typed placeholders
```

## Long text and tool output chunking

Tool outputs can contain large logs, file contents, API responses, command
output, or pasted documents. Privacy filtering must not rely on tokenizer or
model truncation. If text exceeds the model context/window size, split it into
token windows and scan every window.

Recommended v1 behavior:

- tokenize with character offsets
- split tokenized text into windows below the model limit, accounting for
  special tokens
- use a small overlap, e.g. 32-64 tokens, to catch spans crossing boundaries
- run inference and Viterbi decode per window
- map token spans back to original text character offsets
- merge duplicate or overlapping spans from overlapping windows
- replace final spans right-to-left
- fail closed if a long text cannot be fully scanned

The sanitizer API should hide this. Callers still use:

```python
sanitizer.sanitize_text(text)
```

and the implementation handles chunking internally. Unit tests should include a
long text/tool-output fixture that requires more than one window and verifies no
tail text is silently skipped.

## Viterbi decoder

Use OpenAI's official decoder behavior as the reference.

Source inspected:

```text
external/privacy-filter/opf/_core/decoding.py
external/privacy-filter/opf/_core/sequence_labeling.py
external/privacy-filter/opf/_core/spans.py
```

Official default:

```text
decode_mode = "viterbi"
```

Argmax is supported by OpenAI, but should only be used for debugging/smoke tests.

The decoder should:

- consume log-probabilities, not raw logits
- use start, transition, and end scores
- set impossible BIOES transitions to `-1e9`
- load the six transition biases from `viterbi_calibration.json`
- default biases to zero if no calibration file exists

Valid start labels:

```text
O
B-*
S-*
```

Valid end labels:

```text
O
E-*
S-*
```

Valid transitions:

```text
O     -> O | B-* | S-*
B-X   -> I-X | E-X
I-X   -> I-X | E-X
E-X   -> O | B-* | S-*
S-X   -> O | B-* | S-*
```

## Placeholder format

Use typed placeholders by default:

```text
<ACCOUNT_NUMBER>
<PRIVATE_ADDRESS>
<PRIVATE_DATE>
<PRIVATE_EMAIL>
<PRIVATE_PERSON>
<PRIVATE_PHONE>
<PRIVATE_URL>
<SECRET>
```

When replacing multiple spans, apply replacements right-to-left so character
offsets remain valid.

Trim leading/trailing whitespace from spans before replacement. This avoids
output such as:

```text
My name is<PRIVATE_PERSON>
```

when the detected span includes a leading space. Prefer:

```text
My name is <PRIVATE_PERSON>
```

## Error handling

If `--privacy-filter` is not set, export behavior must remain unchanged.

If `--privacy-filter` is set but optional dependencies are missing, fail with an
actionable message that lists the missing packages, e.g.:

```text
Privacy filtering requires optional dependencies that are not installed:
  - onnxruntime
  - tokenizers

Install them with:
  uv pip install "fast-agent-mcp[privacy]"

or:
  pip install "fast-agent-mcp[privacy]"
```

Check dependencies before any Hub cache lookup or download.

If model files are missing:

- explain the required approximate 1 GB download and how to cache the model
- fail unless `--download-privacy-filter` is provided or
  `--privacy-filter-path` points to a valid local model directory
- do not start a network download from `--privacy-filter` alone, even in an
  interactive slash-command context

If model inference fails:

- fail closed by default for `--privacy-filter`
- do not silently write/upload unsanitized data
- include a clear error explaining that the export was not written/uploaded

Potential future option:

```bash
--privacy-filter-on-error fail|warn
```

Default should be `fail`.

## Testing plan

Unit tests should avoid loading the real 875 MB model.

Test layers:

1. Viterbi decoder
   - valid BIOES paths
   - invalid transitions are impossible
   - start/end constraints
   - bias loading/validation
2. Redaction replacement
   - right-to-left replacement
   - overlapping spans
   - whitespace trimming
   - typed placeholders
   - redaction counts by label
3. JSON argument sanitization
   - exported argument strings are redacted directly
   - JSON structure changes caused by redaction are acceptable in v1 because the
     Codex trace field is itself a string
4. Codex writer integration with a fake sanitizer
   - session metadata `base_instructions.text` is sanitized
   - trace privacy metadata is present when filtering is enabled
   - user message content is sanitized
   - assistant message content is sanitized
   - reasoning summary text is sanitized
   - function call arguments are sanitized
   - tool result output is sanitized
   - structural fields are preserved
   - redaction summary counts are aggregated
   - session metadata paths/cwd, binary payloads, IDs, and tool names are
     preserved in v1
5. Export service integration
   - sanitized local export is written
   - sanitized file is what gets uploaded when `--hf-dataset` is used
   - failure in sanitizer prevents write/upload
   - CLI/result output includes total redaction counts and counts by label
   - upload messaging distinguishes privacy-filtered vs unfiltered traces when
     metadata is available
   - uncached model fails with a clear ~1 GB download message unless
     `--download-privacy-filter` is set
   - `/session export ... --privacy-filter` parses and invokes the same export
     path as `fast-agent export`
6. Long text handling
   - text longer than one model window is chunked
   - overlapping window detections are merged
   - no unscanned tail text is silently written

A fake sanitizer can simply replace known substrings:

```python
class FakeSanitizer:
    def sanitize_text(self, text: str) -> str:
        return text.replace("Alice", "<PRIVATE_PERSON>")
```

This keeps tests fast and deterministic.

## Rollout plan

Phase 1: export plumbing, no user-facing flag yet.

1. Add sanitizer interfaces and a fake/no-op implementation for tests.
2. Add optional sanitizer plumbing to `CodexTraceWriter` and
   `SessionTraceExporter`.
3. Add redaction summaries to `ExportResult` and privacy metadata to
   `session_meta`.
4. Add tests proving sanitized local export, sanitized HF upload input,
   fail-closed behavior, default `codex-privacy.jsonl` naming, and structural
   field preservation.

Phase 2: backend and user interface.

1. Implement NumPy Viterbi decoder based on OpenAI's official code.
2. Implement ONNX Runtime privacy-filter backend.
3. Add model-file resolution/download helper.
4. Add long-text chunking for model-window-safe scanning.
5. Add CLI flags to `fast-agent export` and `/session export` with dependency
   checks, warnings, and actionable missing-model messages.
6. Add documentation and examples.

## Open questions

- Should there be a config-file section for the privacy-filter defaults in v1,
  or should v1 stay CLI/slash-option only?
- Should there be a mode to omit reasoning entirely instead of redacting it?
- Should a later mode omit or placeholder binary/base64 payloads, images, audio,
  file paths, filenames, and resource URLs?
