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
uv pip install 'fast-agent-mcp[gepa]'
```

The published `gepa` extra installs PyPI GEPA and Trackio:

```text
gepa>=0.1.1
trackio>=0.44.0
```

Trackio 0.44.0 permits Pillow <14, compatible with fast-agent's security
requirement of `pillow>=13.0.0`. For batch monitoring without GEPA, install
only the Trackio extra with `uv pip install 'fast-agent-mcp[trackio]'`.

!!! warning "GEPA API compatibility"

    PyPI GEPA 0.1.4 lacks Trackio options in `optimize_anything.TrackingConfig`
    and `gepa.api.optimize()`. Installing the extra does not change that API.
    Before using `make_gepa_trackio_dashboard()`, `make_gepa_tracking_config()`,
    or `gepa_api_trackio_kwargs()`, check that your GEPA version supports the
    Trackio options. Otherwise omit those calls and their tracking arguments;
    GEPA optimization and batch Trackio monitoring remain available.

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
