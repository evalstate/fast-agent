---
social:
  title: Docs Automation
  tagline: Generate reference docs, social cards, screenshots, and site builds.
  description: Generate reference docs, social cards, screenshots, and site builds.
  alt: fast-agent social card — Docs Automation
---

# Docs Automation

The docs live in this repository, so pages can include source examples directly and generated
content can be refreshed from the codebase.

## Current Flow

```bash
uv run scripts/docs.py generate
uv run scripts/docs.py social
uv run scripts/docs.py build
uv run scripts/docs.py screenshot
uv run scripts/docs.py assess
```

- `generate` refreshes `_generated/` from the Python source tree.
- `social` renders per-page Open Graph PNGs from HTML using `google-chrome`.
- `build` verifies the committed social PNGs exist, then runs a strict Zensical build.
- `screenshot` captures the built local site and the live site for visual comparison with
  `google-chrome`.
- `assess` runs deterministic screenshot checks for capture dimensions, blank or unstyled pages,
  the designed home-page header, and visible terminal areas.

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

## Social Cards

Every Markdown page gets a committed 1200×630 PNG under `docs/assets/social/`. `overrides/main.html`
points Open Graph and Twitter metadata at those page-specific images.

Regenerate cards locally after adding or renaming docs pages:

```bash
uv run scripts/docs.py social
```

The script renders a designed HTML card with `google-chrome`, then optimizes the PNG. The normal
docs build does not require Chrome; it only fails if an expected image is missing, which catches
Cloudflare Pages builds where a new page was committed without its card.

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

## Plugin API Reference

`docs/generate_plugin_api_docs.py` reads the plugin command dataclasses and runtime protocol from
`src/fast_agent/command_actions/`, writes `_generated/plugin_api.md`, and the plugins guide includes
that snippet. This keeps handler signatures, context fields, result fields, and runtime methods
aligned with the implementation.

## Proposed Next Automations

- Add a CI docs job that runs `uv run scripts/docs.py generate`, fails if generated files changed,
  then runs `uv run scripts/docs.py build` and `uv run scripts/docs.py assess`.
- Add a snippet verifier that scans docs for `--8<--` includes and confirms every referenced file
  exists under an allowed root.
- Add example smoke tests for docs-included examples so pages cannot point at broken sample code.
- Store baseline screenshots for the home page and provider docs page, then compare new Chrome
  screenshots in CI with a small pixel-difference threshold.
