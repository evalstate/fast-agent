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
uv run scripts/docs.py assets
uv run scripts/docs.py social
uv run scripts/docs.py build
uv run scripts/docs.py screenshot
uv run scripts/docs.py assess
```

- `generate` refreshes `_generated/` from the Python source tree.
- `assets` verifies committed interactive docs assets, including vendored asciinema player files.
- `social` renders per-page Open Graph PNGs from HTML using `google-chrome`.
- `build` verifies the committed social PNGs exist, then runs a strict Zensical build.
- `screenshot` captures the built local site and the live site for visual comparison with
  `google-chrome`.
- `assess` runs deterministic screenshot checks for capture dimensions, blank or unstyled pages,
  the designed home-page header, and visible terminal areas.

CI runs `generate`, fails if `_generation_warnings.md` is produced, verifies the committed
`docs/_generated/` snippets are up to date, then runs the strict docs build. Because the docs job is
part of the shared checks workflow, it gates pull requests, `main`, and tag-triggered PyPI
publishing.

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

## Interactive Terminal Assets

Use `scripts/docs_assets.py` through the docs wrapper for interactive asciinema casts:

```bash
uv run scripts/docs.py assets
uv run scripts/docs.py assets-record tui-shell
uv run scripts/docs.py cast-build tui-shell
```

The `tui-shell` scenario records `fast-agent` starting with shell access and a visible model
selection, sends a short prompt, then runs `! git status` from inside the TUI. Recording requires
`asciinema` and `tmux`; normal docs builds only need the committed `.cast` files and the vendored
player assets. `cast-build` is an alias for recording a named cast.

By default, cast recordings stop by killing the tmux session after the final demonstrated action,
so the user-facing recording does not show a trailing `/exit`. Set
`FAST_AGENT_TUI_DEMO_SHOW_EXIT=1` when you explicitly want the exit command included.
Cast recordings also set `TUI__COMPLETION_MENU_RESERVED_LINES=4` by default to keep the prompt
area compact; override it when a recording needs to demonstrate completion menus.
The recording drivers set `FAST_AGENT_KEYRING_NOTICE=0` so the one-time OS keyring probe message
does not appear in committed casts.

The default recording command is:

```bash
uv run fast-agent -x --model deepseek
```

Override it for local refreshes with:

```bash
FAST_AGENT_TUI_DEMO_MODEL=sonnet uv run scripts/docs.py assets-record tui-shell
FAST_AGENT_TUI_DEMO_COMMAND="uv run fast-agent -x --model deepseek" uv run scripts/docs.py assets-record tui-shell
```

A2A recordings use the same shared asciinema invocation, cleanup, and terminal-teardown trimming
helpers, while keeping the deterministic fake-server/snippet generation in the A2A pipeline:

```bash
uv run scripts/a2a_docs_pipeline.py generate
uv run scripts/a2a_docs_pipeline.py check
uv run scripts/a2a_docs_pipeline.py record
```

For provider-backed A2A recordings, export the recording configuration first so the command is
copy/paste-ready and the rendered docs can show exactly which environment was used:

```bash
export HF_TOKEN=...
export OPENAI_API_KEY=...
export A2A_REAL_LLM_MODEL=codexresponses.gpt-5.4-mini
export A2A_HF_MCP_URL=https://hf.co/mcp
export A2A_REAL_LLM_RECORD_SECONDS=70
uv run scripts/a2a_docs_pipeline.py record-real-llm
```

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

- Consider adding `uv run scripts/docs.py assess` to CI after the deterministic screenshot flow is
  stable in hosted runners.
- Add a snippet verifier that scans docs for `--8<--` includes and confirms every referenced file
  exists under an allowed root.
- Add example smoke tests for docs-included examples so pages cannot point at broken sample code.
- Store baseline screenshots for the home page and provider docs page, then compare new Chrome
  screenshots in CI with a small pixel-difference threshold.
