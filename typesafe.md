# Type Safety Plan (ty)

This document describes the rules and approach we will use to make the codebase type-safe with `ty`.
It uses a small set of examples from the `ty` docs to anchor our conventions, then lays out a
reproducible plan for getting to a clean `ty check`.

Sources (examples below are based on these pages):
- [ty rules](https://docs.astral.sh/ty/rules/)
- [ty suppression](https://docs.astral.sh/ty/suppression/)
- [ty configuration](https://docs.astral.sh/ty/configuration/)

## Examples From ty (reference patterns)

### Rule levels via CLI and config
Use rule-level settings to gradually tighten checks:

```shell
ty check \
  --warn unused-ignore-comment \
  --ignore redundant-cast \
  --error possibly-missing-attribute \
  --error possibly-missing-import
```

Equivalent `pyproject.toml`:

```toml
[tool.ty.rules]
unused-ignore-comment = "warn"
redundant-cast = "ignore"
possibly-missing-attribute = "error"
possibly-missing-import = "error"
```

### Targeted suppressions
Prefer narrow, rule-specific suppressions:

```py
sum_three_numbers("one", 5)  # ty: ignore[missing-argument, invalid-argument-type]
```

Multi-line suppression can go on the first or last line of the violation:

```py
sum_three_numbers(  # ty: ignore[missing-argument]
    3,
    2
)
```

### Whole-function suppression
Use `@no_type_check` only when a function is intentionally dynamic:

```py
from typing import no_type_check

@no_type_check
def main():
    sum_three_numbers(1, 2)  # no error for the missing argument
```

## Modern, Pythonic Typing Rules

These rules are what we will follow as we make the codebase type-safe. They are aligned with
Python 3.13+ and current typing guidance.

- Use builtin generics and PEP 604 unions: `list[str]`, `dict[str, int]`, `X | None`.
- For unused features or parameters, check call sites/usage before removal and confirm with the
  user before deleting behavior, even if it appears unused.
- For command parsing, prefer a discriminated `TypedDict` union (e.g., `kind` field) over ad-hoc
  nested dicts. If the command surface is shared across modules or grows over time, keep the
  payload types in a small dedicated module; otherwise colocate them with the parser.
- For dynamic/optional collection attributes, narrow to `collections.abc.Collection` or `Sized`
  when you only need `len()` or membership; avoid materializing unless multiple passes or sorting
  is required. Exclude `str`/`bytes` when treating a value as a general collection.
- Use `getattr` only for truly dynamic attributes (plugin/duck-typed objects); immediately narrow
  the result with `isinstance`/helper guards and avoid masking real missing attributes.
- For core agent types with concrete classes, prefer `isinstance` narrowing against those classes
  over `getattr`/duck-typing to keep behavior explicit and enforceable by the type checker.
- Avoid `hasattr` checks when accessing attributes that all implementations share. Instead, add
  the attribute to the Protocol so the type checker can verify access. This eliminates dynamic
  lookups and makes the interface explicit.
- When accepting an `Iterable` but needing multiple passes or sorting, materialize to `list`
  once at the boundary to avoid exhausting generators.
- Annotate public APIs and module boundaries first (CLI entry points, FastAPI routes, shared utils).
- Avoid `Any` unless crossing untyped boundaries; when unavoidable, localize it and add a comment.
- Prefer `TypedDict` or `Protocol` over loose `dict[str, object]` and `Any` for structured data.
- Use `Literal` or `Enum` for fixed choices; use `Final` for constants.
- Prefer `collections.abc` types for inputs (`Sequence`, `Mapping`, `Iterable`) and concrete types
  for outputs (e.g., `list`, `dict`) when callers rely on mutability.
- If a capability is optional, pick a single shape (property or method) and document it in the
  protocol; avoid supporting both unless required by existing implementations.
- For pydantic models, prefer explicit field types and `Annotated[...]` where validation metadata
  is needed.
- Use `Self` for fluent APIs and `TypeAlias` for complex, reused types.
- Use `type: ignore` only when interacting with third-party APIs that are untyped or known-broken;
  otherwise prefer `ty: ignore[rule]` with the specific rule.

## Reproducible Plan

1. **Baseline**: run `ty check` on `src/fast_agent` and capture the initial error set.
2. **Triage**: group issues by module and rule; fix the highest-signal errors first.
3. **Configure**: set rule levels in `pyproject.toml` so low-signal rules are `warn` while we
   converge; keep `possibly-missing-attribute` and `possibly-missing-import` at `error`.
4. **Annotate**: add types to public APIs, then internal helpers, then tests.
5. **Refine**: replace broad `Any` or `object` with `TypedDict`, `Protocol`, `Literal`, or
   `Enum` as appropriate.
6. **Suppress sparingly**: use `# ty: ignore[rule]` only when the type system cannot express a
   valid pattern; include a short reason.
7. **Enforce**: add `ty check` to CI once warnings are near-zero; tighten rules to `error` as
   we converge.

## **IMPORTANT: Refactoring `*args, **kwargs` Signatures**

When replacing `def func(*args, **kwargs)` with explicit parameters to satisfy the type checker,
**always verify all call sites first**. The `*args` pattern accepts any number of positional
arguments, and removing it can silently break callers that pass positional args you didn't capture.

**Before:**
```python
def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
```

**After (WRONG - missed `read_timeout`):**
```python
def __init__(self, read_stream, write_stream, **kwargs) -> None:
    super().__init__(read_stream, write_stream, **kwargs)
```

**After (CORRECT):**
```python
def __init__(self, read_stream, write_stream, read_timeout=None, **kwargs) -> None:
    super().__init__(read_stream, write_stream, read_timeout, **kwargs)
```

**Checklist before removing `*args`:**
1. Grep for all call sites of the function/class
2. Check if any caller passes positional arguments beyond what you're capturing
3. Check type hints or protocols that define expected signatures (e.g., `Callable[[A, B, C], R]`)
4. When in doubt, keep `*args` and pop known params from it, or add explicit params for all
   positional args callers use

This error is insidious because it causes runtime `TypeError` failures, not type-check failures.

## Decision Log (initial)

- We will use `ty: ignore[rule]` over bare `ty: ignore` and avoid `type: ignore` unless an external
  dependency forces it.
- We will prefer modern syntax (`X | Y`, builtin generics) given Python 3.13+.
- We will represent parsed UI commands as a lightweight discriminated `TypedDict` union (with a
  `kind` field) in a dedicated module, while leaving free-form user input as plain `str`.
