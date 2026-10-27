---
title: Extension Reference
description: Reference documentation for optional fast-agent integration adapters and extension APIs.
social:
  title: Extension Reference
  tagline: Optional integration adapters for fast-agent.
  description: Reference documentation for optional fast-agent integration adapters and extension APIs.
  alt: fast-agent social card — Extension Reference
---

# Extension Reference

This reference covers optional integration adapters that sit around fast-agent
rather than the core `FastAgent` class. Install the dependency extra for the
integration you need.

## GEPA

GEPA support lives in `fast_agent.integrations.gepa`.

```bash
uv add "fast-agent-mcp[gepa]"
```

The published `gepa` extra installs PyPI GEPA:

```text
gepa>=0.1.1
```

The `trackio` extra is temporarily suspended, and `gepa` no longer installs
Trackio. Trackio requires `pillow<13`, which conflicts with fast-agent's
security requirement of `pillow>=13.0.0`. GEPA optimization remains available
without Trackio monitoring. Trackio integration helpers are retained for a
future compatible release.

Use the [GEPA Optimization guide](../guides/gepa/) for workflow guidance. In
short:

- use `FastAgentBatchEvaluator` with `optimize_anything()` for aggregate
  candidate scoring over a full JSONL batch;
- use `FastAgentSingleTaskAdapter` with `optimize_anything()` when the
  evaluator is naturally one fast-agent call at a time;
- use `FastAgentRowWiseBatchAdapter` with `gepa.api.optimize()` when GEPA should
  treat each input row as an optimization instance;
- use `make_gepa_trackio_dashboard()` to combine GEPA Trackio config with
  fast-agent evaluation/reflection callback metrics;
- use `FastAgentReflectionLM` when GEPA reflection calls should use fast-agent
  model aliases, configuration, and audit artifacts.

--8<-- "docs/docs/_generated/extension_reference.md"
