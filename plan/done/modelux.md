# Plan: Model Alias UX (`$namespace.key`) for Cards and Packs

## Why this is a separate plan

`plan/card-pack.md` is the umbrella plan for card pack lifecycle (install/update/remove/registry/config layering).

This document isolates **model selection UX** so card-pack delivery is not blocked by model-picker complexity.

Boundary:

- Card-pack system should ship first with alias detection + warnings.
- Model UX then ships as an independent slice with focused commands and tests.

---

## Goals

1. Help users resolve unresolved alias tokens like `$system.fast` quickly and safely.
2. Keep UX consistent across interactive TUI and ACP slash command sessions.
3. Reuse existing model alias + catalog infrastructure rather than inventing a new model system.
4. Persist user choices in the right layer (prefer env-local overlay) without surprising project-level edits.

---

## Current state (relevant code)

- Alias format + resolution:
  - `src/fast_agent/core/model_resolution.py`
  - exact token format `^\$<namespace>.<key>$`
- Alias storage/validation:
  - `src/fast_agent/config.py` (`Settings.model_aliases`)
- Model catalog + curated suggestions:
  - `src/fast_agent/llm/model_selection.py`
  - `src/fast_agent/cli/commands/check_config.py` (catalog display)
- Existing runtime model slash command (reasoning/verbosity/web tools):
  - `src/fast_agent/acp/slash/handlers/model.py`
  - `src/fast_agent/commands/handlers/model.py`
- Planned card-pack alias detection hook points:
  - from `plan/card-pack.md` Slice B / D

---

## UX principles

1. **Warn first, don’t block** by default.
2. **Guided but skippable**: users can defer alias mapping.
3. **Local-first persistence**: prefer `<env>/fastagent.config.yaml` overlay for pack-specific model choices.
4. **Deterministic output** in non-interactive ACP mode.
5. **No hidden rewrites**: show exactly what keys/values will be written before apply.

---

## Proposed command surface

## Slash commands

### Read-only

- `/model aliases`
  - list configured aliases from effective config
  - highlight unresolved aliases currently required by installed/loaded cards

### Mutations

- `/model aliases set $system.fast <model-spec>`
- `/model aliases unset $system.fast`
- `/model aliases pick $system.fast`
  - guided provider/model picker using curated catalog + optional “all models” expansion

### Batch helper

- `/model aliases resolve`
  - walk unresolved aliases and prompt per alias
  - supports `--apply` / `--dry-run` behavior in ACP-safe form

## CLI parity

- `fast-agent model aliases`
- `fast-agent model aliases set ...`
- `fast-agent model aliases pick ...`
- `fast-agent model aliases resolve [--apply] [--env ...]`

(Exact command namespace can be adjusted during implementation; above is the target UX.)

---

## Resolution flow

Given an unresolved token (e.g. `$system.code`):

1. Validate token format.
2. Offer source options:
   - card-pack suggested default (if present)
   - curated model aliases for selected provider
   - direct model spec entry
3. Show preview:
   - resolved model string
   - provider and transport implications (if applicable)
4. Confirm write target (env overlay vs project config).
5. Persist and re-run alias validation.

If unresolved aliases remain, command ends with actionable summary (not a silent pass).

---

## Selection algorithm

Use existing catalog components for consistency:

1. Provider shortlist:
   - providers with configured keys / enabled local auth from config
   - plus common providers when no credentials found
2. Model shortlist:
   - curated/current catalog entries first (`ModelSelectionCatalog.list_current_entries`)
   - optional `--all` expansion for provider
3. Entry output value:
   - write canonical model string from catalog entry
   - do not write provider alias shortcuts unless explicit user input

No ranking/recommendation heuristics in v1 beyond curated ordering.

---

## Persistence and config layering

Default write target:

- `<env>/fastagent.config.yaml` (or `--env` equivalent)

Fallbacks:

- if no env path and explicit user request, allow project `fastagent.config.yaml`

Write semantics:

- update only `model_aliases` subtree
- preserve unrelated keys/comments where feasible (or document formatting rewrite behavior)
- include dry-run preview before write in ACP mode

This plan assumes config-layering support from `plan/card-pack.md` Slice C.

---

## Integration points with Card Packs

From pack install/update:

- detect required aliases from card models and/or pack manifest metadata
- if unresolved aliases exist:
  - emit concise warning summary
  - suggest `/model aliases resolve`

Optional follow-up prompt:

- interactive sessions may offer immediate “resolve now?” step
- ACP non-interactive sessions should only print command guidance

---

## Implementation slices

### Slice M1: Alias inspection + set/unset

- `/model aliases` read-only view
- `/model aliases set/unset`
- unresolved alias report integration with card-pack warnings

### Slice M2: Guided picker

- `/model aliases pick $token`
- provider/model selection from curated catalog
- preview + apply

### Slice M3: Batch resolver

- `/model aliases resolve`
- per-alias loop with skip/apply summary
- ACP-safe non-interactive mode support

### Slice M4: Polish

- docs, completer support, richer check output
- optional “show where alias came from” (config layer provenance)

---

## Testing strategy

Unit tests:

- token parsing/validation errors
- alias set/unset update logic
- picker output from curated/all model catalogs
- persistence writes to env config overlay

Integration tests:

- slash command flows (interactive + ACP-safe text outcomes)
- unresolved alias warning from card-pack install path
- end-to-end resolution then successful model alias expansion at runtime

Regression tests:

- ensure existing `/model reasoning|verbosity|web_search|web_fetch` behavior unchanged

No mocks/monkeypatching; use temp environments and real config files.

---

## Risks and mitigations

- **Risk:** command surface bloat under `/model`.
  - **Mitigation:** keep alias commands nested and discoverable via usage help.
- **Risk:** user confusion on where aliases are written.
  - **Mitigation:** always print target file path + dry-run preview.
- **Risk:** provider catalog mismatch vs configured credentials.
  - **Mitigation:** mark suggestions as curated defaults; allow manual model spec input always.

---

## Acceptance criteria

1. Users can list, set, unset, and resolve model alias tokens without editing YAML manually.
2. Unresolved aliases required by installed cards are clearly surfaced with a one-command recovery path.
3. Alias changes persist in env-local config by default and survive subsequent runs.
4. Existing model runtime commands remain backward-compatible.
5. Model UX can be released independently after core card-pack manager rollout.
