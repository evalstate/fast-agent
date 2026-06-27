---
title: Welcome Guide
description: An introduction to using `fast-agent`
social:
  title: Install fast-agent
  tagline: Set up fast-agent, configure secrets, and verify your first run.
  description: Set up fast-agent, configure secrets, and verify your first run.
  alt: fast-agent social card — Install fast-agent
---

Getting started with **`fast-agent`** requires Python 3.12 or newer and
[`uv`](https://docs.astral.sh/uv/).

Install or upgrade the CLI:

```bash
uv tool install fast-agent-mcp -U
```

If you have multiple Python versions installed, pin the supported Python version:

```bash
uv tool install fast-agent-mcp -U --python 3.12
```

Verify the install:

```bash
fast-agent check
```

Then start the interactive prompt:

```bash
fast-agent go
```
