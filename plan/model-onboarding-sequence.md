# Sequence Plan: Model Onboarding + `/models` Command Family

Date: 22 February 2026  
Related: [plan/modelux.md](./modelux.md), [plan/card-pack.md](./card-pack.md)

## Status

- Current phase: **PR 3** (alias set/unset mutation)
- PR 1 status: **✅ Completed**
  - Deterministic model settings layering for project + env config
  - Secrets merge behavior preserved
  - `model_aliases` read/write helper with `env|project` target support
  - Dry-run previews + atomic writes + unit tests
- PR 2 status: **✅ Completed**
  - Added `/models` command family (read-only) for TUI and ACP slash handling
  - Implemented `/models`, `/models doctor`, `/models aliases`, `/models catalog <provider> [--all]`
  - Added parser + dispatch + completion + help wiring
  - Added shared `models_manager` handler + ACP markdown adapter + unit tests

Note: CLI command naming has shifted from `setup` to `scaffold` for file generation.
That rename is adjacent UX work and does not change this `/models` sequence.

## Decision summary

- Keep **`/model`** as runtime/session controls (existing behavior).
- Introduce **`/models`** for persistent model configuration and onboarding.
- Do **not** introduce slash `/config` in this slice (avoid broad config-editor scope creep).
- CLI parity target: `fast-agent models ...` (keep `fast-agent config model` as a separate form-based entrypoint).

---

## Why this split is important

Current `/model` is dynamic runtime tuning (`reasoning`, `verbosity`, `web_search`, `web_fetch`).
Using the same command for persistent config edits will confuse users and complicate ACP behavior.

`/models` provides a clear mental model:
- **`/model`** = “change this running session”
- **`/models`** = “configure model defaults/aliases and onboarding”

---

## Product goals

1. When model setup is missing or incomplete, onboarding is the easiest path forward.
2. Alias tokens (`$namespace.key`) are easy to inspect, set, and resolve.
3. Card packs surface unresolved aliases immediately after install/update.
4. ACP/non-interactive mode remains deterministic and script-friendly.
5. Writes are explicit, previewed, and target the correct config layer.

---

## Command grammar (target)

## Slash commands

### Read-only

- `/models`
- `/models doctor`
- `/models aliases`
- `/models catalog <provider> [--all]`

### Mutations

- `/models aliases set <token> <model-spec> [--target env|project] [--dry-run]`
- `/models aliases unset <token> [--target env|project] [--dry-run]`
- `/models aliases pick <token> [--provider <provider>] [--all] [--target env|project] [--dry-run]`
- `/models aliases resolve [--apply] [--target env|project] [--all]`

## CLI parity

- `fast-agent models`
- `fast-agent models doctor`
- `fast-agent models aliases`
- `fast-agent models aliases set ...`
- `fast-agent models aliases unset ...`
- `fast-agent models aliases pick ...`
- `fast-agent models aliases resolve [--apply]`
- `fast-agent models catalog <provider> [--all]`

---

## UX contract

1. **Warn first, do not hard-block** (except explicit command validation errors).
2. **Always print write target path** before apply.
3. **`--dry-run` prints exact key/value changes** and exits 0 without mutating files.
4. **ACP-safe output**: no interactive prompts unless explicit interactive mode permits them.
5. **Manual entry always available** even when curated catalog suggestions exist.

---

## Data/validation contract

- Alias token format remains exact: `^\$<namespace>\.<key>$` (reuse `core/model_resolution.py`).
- Namespace/key validation remains in `Settings.model_aliases` validator.
- Alias resolution checks should call existing `resolve_model_alias(...)` and map failures to actionable messages.
- Card-pack alias detection scans:
  - `card-pack.yaml` `model_aliases_required` (if present)
  - installed card `config.model` values where model is alias token

---

## Sequence plan (PR checklist)

## PR 1 — Config layering + model alias write service (foundation)

Status: **✅ Completed**

### Scope

- Add deterministic config layering for model settings:
  - project config
  - env config (`<env>/fastagent.config.yaml`)
  - secrets merge behavior preserved
- Add read/write helper focused on `model_aliases` subtree mutation.

### Likely files

- `src/fast_agent/config.py`
- new helper module (recommended): `src/fast_agent/llm/model_alias_config.py`

### Required behaviors

- write target enum: `env|project`
- dry-run change preview object
- atomic writes
- preserve existing YAML as much as feasible (document if formatting rewrite occurs)

### Tests

- precedence/order tests
- env-target write tests
- project-target write tests
- dry-run no-mutation tests

### Acceptance

- one helper call can: list aliases, set, unset, preview, write with target path.

---

## PR 2 — `/models` read-only surface

### Scope

- Add parser/dispatch/completion/help for `/models`.
- Implement read-only handlers:
  - `/models`
  - `/models doctor`
  - `/models aliases`
  - `/models catalog <provider> [--all]`

### Likely files

- `src/fast_agent/ui/command_payloads.py`
- `src/fast_agent/ui/prompt/parser.py`
- `src/fast_agent/ui/interactive/command_dispatch.py`
- `src/fast_agent/ui/prompt/completion_sources.py`
- `src/fast_agent/ui/prompt/command_help.py`
- `src/fast_agent/acp/slash_commands.py`
- `src/fast_agent/acp/slash/dispatch.py`
- new shared handler: `src/fast_agent/commands/handlers/models_manager.py`
- new ACP wrapper: `src/fast_agent/acp/slash/handlers/models_manager.py`

### Tests

- parser tests for `/models ...`
- completer tests
- ACP markdown rendering tests

### Acceptance

- `/models doctor` can report unresolved aliases and missing provider readiness without mutation.

---

## PR 3 — Alias set/unset mutation

### Scope

- Implement:
  - `/models aliases set ...`
  - `/models aliases unset ...`
- include `--target` and `--dry-run`.

### Behavior details

- reject invalid token format with clear usage help
- print target file path
- print before/after preview

### Tests

- set/unset success
- invalid token errors
- dry-run output deterministic

### Acceptance

- users can configure alias mappings without editing YAML manually.

---

## PR 4 — Card-pack unresolved alias warning integration

### Scope

- After `/cards add` and `/cards update`, run unresolved alias audit.
- If unresolved aliases exist, append warning block + next-step command.

### Likely files

- `src/fast_agent/commands/handlers/cards_manager.py`
- helper in `src/fast_agent/cards/manager.py` or `models_manager.py`

### Required warning content

- unresolved alias list
- source pack/card context (where known)
- suggested action: `/models aliases resolve`

### Tests

- unit tests for audit helper
- integration test for `/cards add` warning path

### Acceptance

- card pack install/update always exposes unresolved alias risk immediately.

---

## PR 5 — Guided picker (`pick`)

### Scope

- Implement curated picker:
  - provider shortlist from configured credentials + common fallbacks
  - model shortlist from `ModelSelectionCatalog.list_current_entries(...)`
  - optional `--all` expansion

### Behavior details

- always allow manual model-spec entry fallback
- preview + confirm + apply

### Tests

- provider shortlist behavior
- curated ordering behavior
- `--all` behavior

### Acceptance

- one command can map an alias with minimal friction.

---

## PR 6 — Batch resolver (`resolve`)

### Scope

- Implement unresolved-alias loop resolver.
- `--apply` required for mutation in ACP/non-interactive mode.

### Behavior details

- default mode: inspection + plan
- apply mode: per-alias resolution summary with skips/failures

### Tests

- multi-alias resolution flow
- partial skip flow
- ACP-safe non-interactive output

### Acceptance

- user can resolve all unresolved aliases in one command.

---

## PR 7 — Entry-point onboarding flow (first-run experience)

### Scope

- Hook onboarding check before interactive session starts.
- Trigger when model readiness fails (e.g., unresolved default alias, no usable provider key for selected model).

### Likely files

- `src/fast_agent/cli/runtime/agent_setup.py`
- optional startup notices via `queue_startup_notice(...)`

### Behavior details

- interactive mode: concise prompt to launch `/models` guidance
- non-interactive mode: deterministic error + exact next command text

### Tests

- interactive startup notice behavior
- non-interactive failure guidance behavior

### Acceptance

- missing model config no longer produces confusing provider errors as first user experience.

---

## PR 8 — `check` and docs polish

### Scope

- Extend `fast-agent check` output with:
  - unresolved alias table
  - source provenance (project/env) for each alias
  - card-pack alias requirement summary
- documentation updates for `/models` and migration guidance.

### Likely files

- `src/fast_agent/cli/commands/check_config.py`
- docs pages and command help text

### Acceptance

- operators can diagnose model onboarding state from one `check` run.

---

## Output examples (target)

## 1) `/models doctor`

```text
# models

Readiness: action required

Unresolved aliases:
- $system.fast (required by card: code_reviewer)
- $system.code (required by pack: mcp-working-conductor)

Provider readiness:
- OpenAI: not configured
- Anthropic: configured

Next step:
/models aliases resolve
```

## 2) `/models aliases set $system.fast claude-haiku-4-5 --dry-run --target env`

```text
# models

Dry run only (no files changed)
Target: /abs/path/.fast-agent/fastagent.config.yaml

Changes:
model_aliases.system.fast:
  old: <unset>
  new: claude-haiku-4-5
```

## 3) `/cards add alpha` with unresolved aliases

```text
Installed card pack: alpha
location: .fast-agent/card-packs/alpha
managed files: 3

Warning: unresolved model aliases detected:
- $system.fast (card: alpha)

Resolve now with:
/models aliases resolve
```

---

## Test matrix summary

- Unit:
  - token validation + resolver helpers
  - config target write helpers (`env|project`, dry-run)
  - picker ranking/selection
  - card-pack alias audit helper
- Integration:
  - ACP slash `/models` read + mutation paths
  - `/cards add/update` warning integration
  - startup onboarding checks in CLI flow

Rule: no mocks/monkeypatching; use temp dirs and real files.

---

## Open questions (resolve before PR 1)

1. ✅ If env target is requested but env config file does not exist, should we auto-create it? **Resolved: yes**.
2. Should `/models` default to `doctor` when called without args? (recommended: yes)
3. Do we want `/config models` alias later? (recommended: yes, post-slice convenience alias only)

---

## Final acceptance criteria

1. First-time users without model config get clear guided next steps.
2. Card pack installs/updates always reveal unresolved alias requirements.
3. Alias resolution is easy, previewable, and layer-aware.
4. `/model` runtime UX remains unchanged and stable.
5. `/models` becomes the primary onboarding entrypoint for model setup.
