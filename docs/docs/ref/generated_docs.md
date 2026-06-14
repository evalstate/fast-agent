---
social:
  title: Generated Reference Docs
  tagline: Understand how generated API and model reference pages are produced.
  description: Understand how generated API and model reference pages are produced.
  alt: fast-agent social card — Generated Reference Docs
---

# Generated Docs

Some parts of the documentation are generated from the `fast-agent` Python package to prevent drift, including model alias tables, the models reference page, workflow/request/TUI references, configuration snippets, and the plugin API reference.

## Regenerate

From the `fast-agent` repo root:

```bash
uv run scripts/docs.py generate
```

The generator assumes the normal in-repo `docs/` layout. If you need to run it from an unusual
checkout, point generation at the local `fast-agent` source:

```bash
FAST_AGENT_REPO_PATH=../fast-agent uv run python docs/generate_reference_docs.py
```

Generated files are written to `docs/_generated/` and included in pages via `pymdownx.snippets`.

## Source-Backed Config Values

When documentation needs to show defaults from `fast-agent.yaml`, prefer generated snippets from
the Pydantic settings models over hand-written literals. For example, the compaction YAML snippet
and settings table are generated from `fast_agent.config.CompactionSettings`, then included in both
the configuration reference and the compaction guide. If a code default changes, run
`uv run scripts/docs.py generate` and review the generated diff rather than editing each page by
hand.

Keep the annotated setup template in `examples/setup/fast-agent.yaml`; packaged setup resources are
copied from there during build. When setup defaults mirror code-owned settings, add focused tests so
the sample config and generated docs fail visibly if the code default changes.
