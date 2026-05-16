# Docs Automation

The docs live in this repository, so pages can include source examples directly and generated
content can be refreshed from the codebase.

## Current Flow

```bash
uv run scripts/docs.py generate
uv run scripts/docs.py build
uv run scripts/docs.py screenshot
uv run scripts/docs.py assess
```

- `generate` refreshes `_generated/` from the Python source tree.
- `build` runs a strict Zensical build.
- `screenshot` captures the built local site and the live site for visual comparison with
  `google-chrome`.
- `assess` runs deterministic screenshot checks for capture dimensions, blank or unstyled pages,
  the blue home-page header, and visible dark terminal areas.

## Terminal Captures

Use `scripts/docs_terminal_capture.py` to run a command and write a terminal-style SVG that can be
embedded in docs:

```bash
uv run scripts/docs_terminal_capture.py \
  --command "uvx fast-agent-mcp@latest --help" \
  --output docs/docs/ref/terminal-uvx.svg
```

The script uses a pseudo-terminal through `script(1)` when available, then renders the captured ANSI
output with Rich. This makes CLI examples reproducible while still looking close to what users see
in a terminal.

Example output:

![Terminal capture](terminal-fast-agent-go.svg)

## Visual Assessment

`scripts/docs_visual_assess.py` adapts the screenshot QA pattern used by the visual inspection tools
in `/home/ssmith/temp/html-agent-dev`: deterministic checks run first, and an optional vision judge
can inspect the same screenshots with a docs-specific rubric.

```bash
# Run deterministic checks only
uv run scripts/docs.py assess

# Write the vision prompt/card without calling a model
uv run scripts/docs_visual_assess.py --dry-run

# Run the vision judge when credentials are available
uv run scripts/docs_visual_assess.py --vision --model gpt-5.5
```

The deterministic path is intended for routine local and CI use. The vision path adds checks that
pixel metrics cannot reliably catch: literal Markdown artifacts, overlapping labels, awkward mobile
wrapping, weak feature copy, and whether CLI examples read like real terminal output.

## Source-Backed Includes

`pymdownx.snippets` is configured with both `docs/docs` and the repository root. Documentation
pages can include examples directly from `examples/`:

```markdown
--8<-- "examples/workflows/parallel.py"
```

Prefer direct includes for examples that are meant to stay runnable. This keeps docs and examples
on one source of truth and makes drift visible in ordinary code review.

## Provider Overviews

Provider prose can live next to provider implementation code:

- `src/fast_agent/llm/provider/anthropic/provider_docs.md`
- `src/fast_agent/llm/provider/openai/provider_docs.md`

`docs/generate_reference_docs.py` copies those files into `_generated/provider_overview_*.md`.
The public provider page includes the generated snippets, so feature prose can be reviewed beside
the implementation it describes.

## Proposed Next Automations

- Add a CI docs job that runs `uv run scripts/docs.py generate`, fails if generated files changed,
  then runs `uv run scripts/docs.py build` and `uv run scripts/docs.py assess`.
- Add a snippet verifier that scans docs for `--8<--` includes and confirms every referenced file
  exists under an allowed root.
- Add example smoke tests for docs-included examples so pages cannot point at broken sample code.
- Store baseline screenshots for the home page and provider docs page, then compare new Chrome
  screenshots in CI with a small pixel-difference threshold.
