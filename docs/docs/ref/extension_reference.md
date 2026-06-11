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

The `gepa` extra currently installs GEPA from the Trackio integration branch and
adds Trackio:

```text
gepa @ git+https://github.com/evalstate/gepa.git@feat/trackio
trackio>=0.26.0
```

Use the [GEPA Optimization guide](../guides/gepa/) for workflow guidance. In
short:

- use `FastAgentBatchEvaluator` with `optimize_anything()` for aggregate
  candidate scoring over a full JSONL batch;
- use `FastAgentRowWiseBatchAdapter` with `gepa.api.optimize()` when GEPA should
  treat each input row as an optimization instance;
- use `FastAgentReflectionLM` when GEPA reflection calls should use fast-agent
  model aliases, configuration, and audit artifacts.

--8<-- "docs/docs/_generated/extension_reference.md"
