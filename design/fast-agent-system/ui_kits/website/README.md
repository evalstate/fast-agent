# Website UI kit (fast-agent.ai)

A redesign of the Zensical docs site (`evalstate/fast-agent`, `docs/`) in Brand B / Forward. Open `index.html`. It's a click-through: the nav tabs route between the pages, and the Copy buttons fire a toast.

## Screens
- **Home** (`HomePage.jsx`): the hero headline and CTAs, one receipt card for the headline number (with its footnote), the "Get started" band with the pointer mascot and the install commands, and a feature directory as link rows.
- **Benchmarks** (`BenchmarksPage.jsx`, `BenchmarkChart.jsx`): a new page with comparison tabs (Frontier / Value (6hr) / GPT-5.6), an accuracy-vs-cost scatter, a detail panel for the selected run, a results table and the methodology. The presenter mascot sits outside the chart.
- **Docs** (`DocsPage.jsx`): the Getting Started content with section nav, an on-page TOC, and a help card with the sitter mascot. Every docs tab (Guides, Agents, Models and so on) shares this layout.
- `Shell.jsx`: the announce bar, header (wordmark, search, repo), nav tabs, footer and a copyable code block.
- `data.js`: a subset of `docs/docs/javascripts/homepage-benchmark-data.js`.

## Porting notes for Zensical
- Add `{ "Benchmarks" = "benchmarks/index.md" }` to `nav` in `zensical.toml`. Move the `homepage-benchmark*.js` includes and the `data-fa-benchmark` mount from `index.md` to `benchmarks/index.md`. The current methodology text becomes the lower section of that page.
- Swap JetBrains Mono in `overrides/main.html` for Fraunces, Figtree and DM Mono (`tokens/fonts.css`).

## Mascots
The kit uses clean transparent exports from `assets/illustration/`: presenter on Benchmarks, pointer on Home, sitter on Docs. Use one character per page.
