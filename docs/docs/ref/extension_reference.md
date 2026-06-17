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

The published `gepa` extra installs PyPI GEPA and Trackio:

```text
gepa>=0.1.1
trackio>=0.27.0
```

PyPI packages cannot declare direct Git dependencies in extras. Trackio-specific
GEPA helpers require a GEPA release with Trackio support; until that support is
available on PyPI, install the integration branch in your application
environment:

```bash
uv add "gepa @ git+https://github.com/evalstate/gepa.git@feat/trackio"
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
