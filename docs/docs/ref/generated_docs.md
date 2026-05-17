---
social:
  title: Generated Reference Docs
  tagline: Understand how generated API and model reference pages are produced.
  description: Understand how generated API and model reference pages are produced.
  alt: fast-agent social card — Generated Reference Docs
---

# Generated Docs

Some parts of the documentation are generated from the `fast-agent` Python package to prevent drift (e.g. model preset tables and the models reference page).

## Regenerate

From the `fast-agent` repo root:

```bash
uv run scripts/docs.py generate
```

If you are working from a separate docs checkout, point generation at the local `fast-agent` source:

```bash
FAST_AGENT_REPO_PATH=../fast-agent uv run python generate_reference_docs.py
```

Generated files are written to `docs/_generated/` and included in pages via MkDocs `pymdownx.snippets`.
