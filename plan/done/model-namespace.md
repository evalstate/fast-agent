# Model namespace plan (`$system.*`) — Phase 1 (no list support)

## Goal
Add configurable model namespaces so users can define reusable model references in `fastagent.config.yaml` and use them anywhere a model string is accepted.

Primary target:
- `system.default`
- `system.plan`
- `system.fast`

Usage example:
- `model: "$system.fast"`
- `default_model: "$system.default"`

## Scope for Phase 1

### In scope
- String-based namespace references only (no lists).
- Extensible namespace map (not hardcoded to only `system`).
- Resolution integrated into the existing model selection flow.
- Backward compatibility for all existing plain model strings.

### Out of scope (deferred)
- List-valued aliases (`$system.fast` -> multiple models).
- Automatic fallback across multiple configured models.
- Fan-out behavior driven by namespace values.

---

## Proposed config shape

```yaml
# Existing behavior still works
# default_model: "gpt-5-mini.low"

# New namespace map (extensible)
model_aliases:
  system:
    default: "responses.gpt-5-mini.low"
    plan: "codexplan"
    fast: "claude-haiku-4-5"

# Optional: use alias as default
# default_model: "$system.default"
```

Notes:
- `model_aliases` is a nested map: `namespace -> key -> model string`.
- `system` is conventional, not special-cased in schema.
- Values must be **strings** in Phase 1.

---

## Implementation plan

### 1) Config schema
**File:** `src/fast_agent/config.py`

- Add typed field on `Settings`:
  - `model_aliases: dict[str, dict[str, str]]`
- Default should be empty map.
- Keep `default_model: str | None` unchanged.

### 2) Namespace resolver utility
**File:** `src/fast_agent/core/model_resolution.py`

Add helper(s), e.g.:
- `resolve_model_alias(model: str, aliases: Mapping[str, Mapping[str, str]] | None) -> str`

Behavior:
- If string does **not** start with `$`, return as-is.
- If format is `$<namespace>.<key>`, resolve from `model_aliases`.
- If missing namespace/key, raise `ModelConfigError` with actionable message.
- Support recursive alias expansion with cycle protection (`$a.b -> $c.d -> ...`).

Phase-1 simplification:
- Require exact alias token format only (`$system.fast`).
- Do not support query/effort suffix on the alias token itself in this phase.

### 3) Apply resolver at runtime entry points
Update all model-string entry paths before `ModelFactory.parse_model_string/create_factory`:

- `src/fast_agent/core/model_resolution.py` (`resolve_model_spec` return path)
- `src/fast_agent/agents/llm_decorator.py` (`set_model`)
- `src/fast_agent/mcp/sampling.py` (when request provides explicit model)

This keeps behavior consistent across:
- config default model
- CLI override
- per-agent model
- runtime model changes
- MCP sampling calls

### 4) UX/docs/templates

- Update setup template:
  - `src/fast_agent/resources/setup/fastagent.config.yaml`
- Update config reference docs:
  - `docs/docs/ref/config_file.md`
- Optional: mention in model docs/examples that `$system.default|plan|fast` is a user-defined convention.

### 5) Tests

Add unit tests for:
- successful resolution of `$system.default`, `$system.plan`, `$system.fast`
- passthrough behavior for non-`$` model strings
- precedence still works (env/config/CLI/explicit) with aliases
- unknown alias/key errors
- recursive alias cycle detection

Suggested files:
- `tests/unit/fast_agent/core/test_model_selection.py` (extend)
- new `tests/unit/fast_agent/core/test_model_namespace_resolution.py` (if cleaner)

---

## Backward compatibility

- Existing `default_model` and agent `model` strings remain valid.
- Existing alias behavior in `ModelFactory.MODEL_ALIASES` is unchanged.
- New feature is opt-in via `model_aliases` + `$namespace.key` syntax.

---

## Risks / edge cases

- **Confusion with env-var syntax** (`${VAR}`): document that `$system.fast` is model-alias syntax, not env interpolation.
- **Partial adoption paths**: if some call paths bypass resolver, behavior diverges. Mitigate by covering all `ModelFactory.create_factory/parse_model_string` entry points.
- **Recursive loops**: must fail fast with explicit cycle errors.

---

## Phase 2 (later)
When ready, extend `model_aliases` values from `str` to `str | list[str]` and define explicit semantics (priority fallback vs fan-out). Not included in this phase.
