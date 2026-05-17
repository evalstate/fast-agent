# Docs Design Integration Handover

Date: 2026-05-16
Branch: `dev/0.7.5`
Repo: `/home/shaun/source/fast-agent-pr`
Local docs server: `http://127.0.0.1:8000`

## Context

The docs repo is no longer a submodule on `dev/0.7.5`; it is now an ordinary `docs/` folder in the fast-agent repo. The old submodule checkout was moved aside during branch switch, assessed, and removed. No unique source changes were kept from that backup; old docs repo metadata remains under `.git/modules/docs` and was used to restore missing guide content.

Design source provided by user:

```text
/tmp/design/pr/
```

Important source files there:

```text
/tmp/design/pr/README.md
/tmp/design/pr/docs/mkdocs.yml
/tmp/design/pr/docs/overrides/main.html
/tmp/design/pr/docs/docs/stylesheets/fast-agent.css
/tmp/design/pr/docs/docs/assets/brand/*.svg
```

## Build / serve process

Intended helper:

```bash
uv run scripts/docs.py build
```

This runs generation then `zensical build --strict`. On this branch `zensical` is referenced but not installed as a project dependency, so builds/serve have been run with explicit `uv --with`:

```bash
cd docs
uv run --with zensical zensical build --strict
uv run --with zensical zensical serve --dev-addr 127.0.0.1:8000
```

Current running server PID is tracked in:

```text
/tmp/fast-agent-docs-server.pid
```

Log:

```text
/tmp/fast-agent-docs-server.log
```

Restart pattern:

```bash
old=$(cat /tmp/fast-agent-docs-server.pid 2>/dev/null || true)
[ -n "$old" ] && kill "$old" 2>/dev/null || true
cd docs
nohup uv run --with zensical zensical serve --dev-addr 127.0.0.1:8000 \
  > /tmp/fast-agent-docs-server.log 2>&1 &
echo $! > /tmp/fast-agent-docs-server.pid
```

Strict build currently passes.

## Files changed / added

Primary design integration:

```text
docs/docs/stylesheets/fast-agent.css
docs/overrides/main.html
docs/mkdocs.yml
docs/docs/index.md
docs/docs/assets/brand/
```

Restored missing guides from old docs repo metadata:

```text
docs/docs/guides/codex.md
docs/docs/guides/hf-dev.md
docs/docs/guides/privacy_filter.md
docs/docs/guides/structured-outputs.md
docs/docs/guides/2026-03-21-llamacpp.png
docs/docs/guides/ACPX-flow.png
docs/docs/guides/agent-trace.png
docs/docs/guides/image.png
docs/docs/guides/news-output.png
```

Restored missing referenced ref page:

```text
docs/docs/ref/export_command.md
```

Previously generated docs changed after running generation:

```text
docs/docs/_generated/request_params_reference.md
docs/docs/_generated/workflows_reference.md
```

## Design assets

Brand assets added:

```text
docs/docs/assets/brand/fast-agent-anim-dark.svg
docs/docs/assets/brand/fast-agent-anim-light.svg
docs/docs/assets/brand/fast-agent-icon.svg
docs/docs/assets/brand/fast-agent-lockup-dark.svg
docs/docs/assets/brand/fast-agent-lockup-light.svg
docs/docs/assets/brand/fast-agent-social.svg
```

Removed copied Windows `*:Zone.Identifier` sidecar files.

### Animated wordmark

Homepage now uses animated SVG wordmark instead of a plain `h1`:

```html
<span class="fa-hero__brand" aria-label="fast-agent">
  <img class="fa-hero__wordmark fa-hero__wordmark--dark" src="assets/brand/fast-agent-anim-dark.svg" alt="fast-agent">
  <img class="fa-hero__wordmark fa-hero__wordmark--light" src="assets/brand/fast-agent-anim-light.svg" alt="fast-agent">
</span>
```

The original animated SVGs had a clip rect with `width="0"` but no animation. Added SMIL animations:

- reveal width: `0 → 436`, `1.4s`, linear
- cursor x: `92 → 528`, `1.4s`, linear
- cursor blink after typing: opacity animation, `1.6s`, starts at `1.8s`, repeats indefinitely

If browser cache prevents replay, hard refresh. As `<img>`, SVG animation starts on image load.

### Icon centering

`fast-agent-icon.svg` chevron was visually high. Path was shifted down/right to center it better within the `256x256` icon.

## Theme / CSS notes

### Palette toggle

Material/Zensical uses:

```html
data-md-color-scheme="default"  <!-- light -->
data-md-color-scheme="slate"    <!-- dark -->
```

The design CSS initially used `[data-theme="light"]`, which broke light/dark switching. Fixed light-mode token mapping to:

```css
[data-md-color-scheme="default"] { ... }
[data-md-color-scheme="default"] { color-scheme: light; }
[data-md-color-scheme="slate"] { color-scheme: dark; }
```

Also set the first palette entry to `primary: custom` / `accent: custom` to avoid Material emitting `indigo` for system mode.

### Homepage-only hidden generated UI

Material/Zensical still renders edit/raw buttons and page title on homepage. They are hidden visually via homepage-only inline CSS in `docs/overrides/main.html`:

```css
.md-content__inner > .md-content__button { display: none; }
.md-content__inner > h1#__skip { display: none; }
```

This is scoped with:

```jinja
{% if page and page.url in ["", "index.html"] %}
```

Other pages still show edit/raw.

### Buttons

Homepage buttons now use canonical design classes:

```html
<a class="fa-btn fa-btn--primary">...</a>
```

Important contrast token:

```css
--a-fg: #0b0c0f; /* foreground on amber accent */
```

Scoped fixes were needed because `.md-typeset a` overrode button text/underline. Added:

```css
.md-typeset .fa-btn { ... }
.md-typeset .fa-btn--primary { ... color: var(--a-fg); }
```

Buttons now have bottom inset and hover lift/glow.

### Copy confirmation contrast

Material copy feedback uses `.md-dialog`. Its default text color was low contrast against amber. Fixed with:

```css
.md-dialog { background-color: var(--a-0); ... }
.md-dialog__inner { color: var(--a-fg); ... }
```

Also changed:

```css
::selection { background: var(--a-0); color: var(--a-fg); }
--md-accent-bg-color: var(--a-fg);
```

### Terminal block

Homepage terminal uses the designer structure:

```html
<div class="fa-term">
  <div class="fa-term__bar">
    <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    <strong>~/projects/fast-agent</strong>
  </div>
  <pre><code>...</code></pre>
</div>
```

Added `pre code` reset so Material code styling does not interfere:

```css
.fa-term pre code {
  display: block;
  background: transparent !important;
  box-shadow: none;
  color: var(--fg-0);
  font-family: var(--mono);
  padding: 0;
}
```

Terminal shadow uses token:

```css
--shadow-terminal
```

with separate dark/light values. Dark mode now has subtle amber ambient glow.

### Design-system helper classes

User provided inline design reference CSS. Incorporated into shared `fast-agent.css`:

```text
.ds-section
.ds-section h2/h3
.ds-section h3 .meta
.tk-grid
.tk-cell
.ds-type
.ds-spacing
.ds-example
.ds-example--col
.ds-cap
.ds-ico
.ds-jump
.ds-theme-toggle
```

This allows markup like:

```html
<section class="ds-section" id="resources">
  <p class="fa-eyebrow">08 · Resources</p>
  <h3>Terminal block <span class="meta">.fa-term · for hero or feature sections</span></h3>
</section>
```

Section numbering / labels use the existing `fa-eyebrow` class; no separate helper is needed. In the design-system page this pairs with the sticky jump nav:

```html
<!-- Sticky section jumps -->
<nav class="ds-jump">
  <a href="#cover">Cover</a>
  <a href="#foundations">Foundations</a>
  <a href="#brand">Brand</a>
  <a href="#components">Components</a>
  <a href="#patterns">Patterns</a>
  <a href="#docs-demo">Docs demo</a>
  <a href="#voice">Voice</a>
  <a href="#resources">Resources</a>
</nav>
```

## Homepage class migration

Originally homepage used older classes like:

```text
fa-feature-grid
fa-button
fa-terminal
fa-actions
fa-proof__grid
```

These were changed to canonical design classes:

```text
fa-grid fa-grid--4
fa-card
fa-btn
fa-term
fa-hero__actions
fa-step
```

Class audit showed no missing classes after migration.

### Homepage “Get started now” band

The `#try-it-now` section intentionally stays simple: a plain `fa-band` with a scoped `fa-band--start` modifier, an `h2`, and a code block containing the one-line terminal command:

```html
<section id="try-it-now" class="fa-band fa-band--start">
  <div>
    <h2>Get started now</h2>
  </div>
  <pre><code>uvx fast-agent-mcp@latest -x</code></pre>
</section>
```

Current design direction:

- Keep the base `fa-band` visual language: amber wash, left amber rule, no extra gradients.
- Use two columns on desktop so the title and command align.
- Give the command enough width/padding/scale to balance the title.
- Stack on narrow screens.
- Do not add a separate visible copy button here.
- Avoid `fa-eyebrow` in this band; the extra arrow/label felt too noisy.
- Avoid Markdown-in-HTML fenced blocks for this homepage CTA. Although strict build passed, it destabilized the page experience and introduced awkward copy overlay behavior.

The scoped CSS currently lives near the base `fa-band` rules:

```css
.fa-band--start {
  grid-template-columns: minmax(220px, 0.8fr) minmax(460px, 1.2fr);
}
.fa-band--start h2 {
  font-size: var(--t-h1);
}
.fa-band--start > pre {
  margin: 0;
  min-width: 0;
}
.fa-band--start > pre > code {
  font-size: var(--t-body);
  padding: var(--sp-5) var(--sp-6) !important;
}
@media (max-width: 820px) {
  .fa-band--start {
    grid-template-columns: 1fr;
  }
}
```

## Guides nav

Added `Guides` section to `docs/mkdocs.yml`:

```yaml
- Guides:
    - guides/codex.md
    - guides/hf-dev.md
    - Privacy Filter: guides/privacy_filter.md
    - guides/structured-outputs.md
```

Added `Export Command` to Reference because restored guides link to it:

```yaml
- Export Command: ref/export_command.md
```

## Visual/design validation

There is existing docs visual tooling:

```text
scripts/docs.py
scripts/docs_visual_assess.py
docs/visual_assessment.schema.json
docs/docs/ref/docs_automation.md
```

Commands:

```bash
uv run scripts/docs.py screenshot
uv run scripts/docs.py assess
uv run scripts/docs_visual_assess.py --vision --model <model>
```

Caveat: the current visual assessment rubric still references old expectations like a blue header. It should be updated for the new amber/dark design language before relying on it.

## Known follow-ups / questions

1. Consider adding `zensical` to project dev dependencies or updating `scripts/docs.py` to use `uv run --with zensical`.
2. Consider replacing homepage-only CSS hiding of edit/raw/title with a cleaner template block override if DOM removal is preferred.
3. Update `scripts/docs_visual_assess.py` prompt/rubric to remove old “blue brand header” expectation.
4. Review light/dark mode visually after hard refresh; token mapping is fixed, but design tuning may still be desired.

## Current verification

Last checked:

```bash
cd docs && uv run --with zensical zensical build --strict
```

Result:

```text
No issues found
```
