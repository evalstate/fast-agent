# Development checks

Install the locked development tools with `uv sync --locked --group dev`, then run:

```shell
uv run scripts/format.py --check
uv run scripts/lint.py
uv run scripts/typecheck.py
uv run scripts/deadcode.py
uv run scripts/cpd.py --check
```

The Ruff wrappers use the project's locked dependencies rather than separate
inline script pins. Keep the Ruff pre-commit revision aligned with the project
pin when upgrading it.

## Dead-code checks

Ruff's `F` rules already catch unused imports and local assignments. The
additional Vulture gate scans `src/fast_agent` and both published wrapper source
trees at **100% confidence**, covering unused arguments and unreachable code.
Tests, examples, generated benchmark code, and scripts are not part of that
production-source gate. The only shipped-source exclusion is
`acp/server/unsupported_methods.py`: this pure protocol shim deliberately rejects
unsupported requests or ignores notifications while retaining the SDK's keyword
signatures. Excluding that file avoids globally suppressing common parameter
names in real implementations.

For manual review, use:

```shell
uv run scripts/deadcode.py --min-confidence 60
uv run scripts/deadcode.py path/to/module.py
```

Lower-confidence findings include public APIs, schema fields, and callbacks
registered by frameworks; they are review leads, not permission to remove code.
Confirm call sites and obtain approval before deleting behavior. Review
intentional protocol/callback exceptions in `scripts/vulture_allowlist.py`.
Vulture matches names globally, so allowlist additions can hide unrelated uses
of the same name; do not bulk-add findings or treat a clean scan as proof that
all code is exercised.

## Duplication checks

CPD is a separate structural duplication gate, not a dead-code detector. It
scans `src`, with documented exclusions in `scripts/cpd.py`, and compares
findings against `scripts/cpd_baseline.json`. The baseline includes the PMD
version because tokenizer changes can affect findings. Review changed findings
before updating it; do not regenerate a large baseline just to make CI pass.

The runner uses a cached JRE, a compatible system Java, or downloads its pinned
Temurin fallback. Downloaded tar archives use Python's `data` extraction filter.
