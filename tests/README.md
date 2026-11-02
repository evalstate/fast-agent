# Running and profiling tests

Run the suites with the same worker strategy as CI:

```sh
uv run pytest tests/unit -n auto -q --durations=20
uv run pytest tests/integration -n 2 --dist loadfile -q --durations=20
```

Integration tests use real local servers. `loadfile` keeps each module on one
worker so module-scoped ACP process fixtures are reused. MCP test ports are
allocated dynamically, including when separate suite runs overlap. Docker
tests skip when Docker is unavailable. Select the integration directory without
`-m integration`: some legitimate harness and adapter tests have no marker.

To profile elapsed test work, omit `-n` and `--dist`. Compare `setup`, `call`,
and `teardown` durations as well as total wall time; worker startup and repeated
fixture setup can outweigh parallelism for small selections. Use the same
machine and worker count for before/after comparisons.

The main avoidable costs found in the October 2026 audit were repeated SDK
imports in child processes, production startup grace periods in unrelated
process tests, and oversized image fixtures. Keep fresh subprocesses for import
isolation contracts, but batch cases with identical ambient configuration. Tests
of process polling should synchronize on output; startup grace itself is covered
in `unit/fast_agent/tools/test_background_launch_settle.py`.

Prefer observable behavior and independent inputs over reconstructing expected
results from production tables. The catalog filtering test uses a small input
with duplicates, a retired model, and a second provider, so it can detect broken
filtering and ordering without copying the live catalog.
