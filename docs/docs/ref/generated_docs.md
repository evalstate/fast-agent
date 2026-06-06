---
social:
  title: Generated Reference Docs
  tagline: Understand how generated API and model reference pages are produced.
  description: Understand how generated API and model reference pages are produced.
  alt: fast-agent social card — Generated Reference Docs
---

# Generated Docs

Some parts of the documentation are generated from the `fast-agent` Python package to prevent drift, including model alias tables, the models reference page, workflow/request/TUI references, and the plugin API reference.

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
