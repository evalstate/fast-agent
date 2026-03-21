# Generated Docs

Some parts of the documentation are generated from the `fast-agent` Python package to prevent drift (e.g. model preset tables and the models reference page).

## Regenerate

From the `fast-agent-docs` repo root:

```bash
uv run python generate_reference_docs.py
```

If you don't have `fast_agent` installed in the docs venv, run it against a local checkout:

```bash
FAST_AGENT_REPO_PATH=../fast-agent uv run python generate_reference_docs.py
```

Generated files are written to `docs/_generated/` and included in pages via MkDocs `pymdownx.snippets`.
