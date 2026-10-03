# fast-agent Design System — Brand B / Forward

Type, colour, marks, motion and tone for fast-agent's new identity, plus React primitives built on those tokens. fast-agent is a developer tool: an agent framework and CLI with MCP support.
Status: **refinement**. The wordmark is typeset in Fraunces until custom lettering is drawn.

**Source:** the local brand folder `fast-agent-brand/` (readme, tokens, assets, guideline specimens, SKILL.md). We received no product UI code, Figma file or slide deck. That's why there are no UI kits or slides yet (see Caveats).

## The universe: "atomic-age salesmanship, lab-grade receipts"

The look borrows from **mid-century American commercial art, roughly 1952–1965**: atomic-age starbursts, Googie optimism, Madison Avenue presenters, sticker-shop "NEW!" bursts and boomerang energy. This is the *mainstream* ad-man confidence of that era, pointed at a developer tool.

The twist is **honesty**. The salesman gets the headline and the burst, while the data stays level, sourced and footnoted. The fun lives in the margins and the proof sits in the middle.

- Confident, warm and a little cheeky. Never smug.
- Flat print colour on ivory stock. The UI has no faked paper grain or halftone, though illustration may use them.
- One salesman per room: one burst, one tilt and one shout per view.

## Content fundamentals

- **Voice:** short declaratives, with the receipt next to the claim. "Same model, better results." is followed by the numbers.
- **Person:** "you" for the reader and "we" for the team. The product is always "fast-agent" (lowercase, hyphenated).
- **Casing:** sentence case for headlines and buttons ("Try it now", "Read the docs"). UPPERCASE with 0.14em tracking only for small labels.
- **Spelling:** British English (flavour, colour, optimise).
- **Numbers:** always give n, date, model and harness version. Round down, never up. If a number has a disclaimer, the disclaimer goes on the same screen.
- **Banned:** supercharge, unleash, seamless, revolutionise, game-changing, "AI-powered". No emoji.
- **Allowed exuberance:** "New!" and "More accurate & efficient" inside a burst. That's where the shouting lives.
- **Microcopy examples:** "Config saved.", "Run failed: provider timed out.", "Delete this run?" / "The transcript and scores go with it."

## Visual foundations

**Colour** (`tokens/colors.css`)
- Ivory #FFF7E8 is the ground and petrol #082C34 is the ink. Dark mode is petrol (#11414B for raised surfaces), never black.
- Amber #FFB52E **means fast-agent**: our series, our CTA, our marker. It's never decoration.
- Orange #F45125 and teal #277C80 are for campaigns and illustration. Teal is also the focus ring and link hover. Competitor chart colours are still being designed.
- Contrast: amber works as a fill on ivory, never as text (use --amber-ink #8A5A00 for text). Teal text only works on ivory. Orange doesn't carry body text.

**Type** (`tokens/typography.css`)
- **Voice:** Fraunces 900, SOFT 100, tracking −0.035em. Used for the wordmark, headlines and section titles.
- **Shout:** Ultra. Campaign bursts, stickers and banners only, one per view.
- **Read:** Figtree. Body 16/1.6, UI 600, names 800, numbers 900 with tabular-nums. Disclaimers go no smaller than 12px.
- **Code:** DM Mono for the CLI and code blocks.

**Spacing and shape** (`tokens/shape.css`)
- Spacing scale: 4, 8, 12, 16, 24, 32, 48, 64, 96.
- Structure is a **2px petrol keyline**. Don't use 1px hairlines. --line (14% petrol) is only for row dividers.
- Radii: 6 (bars, inputs, tags, switches), 10 (buttons, toasts), 14 (cards, dialogs, code), 23% (app icon). No pills, except the chevron tile and the bursts.
- Shadows: none, with one exception. A **solid offset** with no blur goes under stickers (6px amber, or 3px petrol on amber) and under the primary CTA ledge (0 4px 0 amber).
- Cards: paper or ivory fill, 2px keyline, radius 14. No left-border accents and no shadow stacks.

**Backgrounds and imagery**
- Flat colour fields only. The one allowed tint is a flat band marking a chart region (for example, "cheap"). No gradients.
- Illustration is mid-century commercial: a limited palette (petrol line, amber, orange, teal, cream), confident poses and atomic sparkles. Imagery is warm and flat. No full-bleed photography has been defined.
- One mascot per surface, always outside the data area of charts.
- Characters (`assets/illustration/`, transparent PNG, trimmed, ≤900px):
  - presenter: **the default character**. Blonde, teal suit, open-mouth smile, palm out to the left. Place her to the right of the thing she presents. Used on Benchmarks. Variants: presenter-smile, -wink, -smirk.
  - presenter-amber: redhead, amber suit, palm out to the right.
  - Pick the character page by page. When in doubt, use presenter.
  - pointer: points right. Used on the home install band.
  - sitter (and sitter-alt): perches on a card edge. Used on the docs help card.
  - bust-wink: head and shoulders, for avatars and small spots.

**The tilt:** bursts, stickers and badges rest at −4° to −10° (default −8°). Headlines, copy, tables, charts and buttons **never** tilt.

**Motion** (`tokens/motion.css`)
- UI uses --ease-ui cubic-bezier(.2,.7,.2,1): 120ms for hover, 200ms for state changes.
- Bursts pop in with --ease-pop (overshoot) over 420ms, going from scale .55 and −26° to the resting tilt. The pop plays once.
- Loading: the capsule *ratchets* 72° and then pauses. Alternatives are the ❯ caret blink and the chevron chase. Use flat ivory-deep blocks as placeholders, never shimmer skeletons.
- Hover: the primary button's ledge goes 4→2px as the button drops. The secondary button fills amber. A link's underline goes 2→4px and the text turns teal. Chart markers rotate 36° and scale 1.15.
- Press: the button drops fully onto its ledge. No scale-down.
- Focus: 3px teal ring, 2px offset.
- Reduced motion: opacity only.
- No parallax, scroll-jacking or cursor trails.

**Transparency and blur:** none, since nothing on this brand is frosted. The only exception is the Dialog scrim, which is flat petrol at 80% (see Intentional additions).

**Layout:** content sits on an ivory page with ivory-deep insets. Fixed elements aren't specified. Keep navs flat, with a keyline divider.

## Iconography

- **App icon:** the chevron tile (`assets/icon-tile.svg`), a petrol tile with an amber soft ❯. The amber variant is `icon-tile-amber.svg` and the favicon is `icon-tile-16.svg`. The "fa" monogram is retired.
- **Mark universe:**
  - Capsule burst: primary burst, chart marker and footnote marker.
  - Soft star: stickers and merch.
  - Chevron: CLI and docs.
  - Atomic sparkle: decoration only.
  - Score burst: campaign headline numbers.
- **Two sets:** `assets/*.svg` have fixed colours, for img, Figma, email and slides. `assets/mono/*.svg` and `assets/sprite.svg` use currentColor, for the web: `<svg style="color:var(--amber)"><use href="assets/sprite.svg#fa-burst"/></svg>`. The symbols are fa-burst, fa-burst-keyline, fa-star, fa-chevron, fa-sparkle, fa-score, fa-tile and fa-tile-16.
- **UI icons:** not yet chosen. The recommendation is one stroke set with rounded caps at 2px, such as Lucide via CDN. None ships here, and the components use unicode ❯ and × instead.
- No emoji. Unicode ❯ is fine in code and CLI copy.

## De-slop checklist

Remove these: KPI capsules and stat pills, gradients, glass or blur, soft shadows, left-border accent cards, emoji and ✨ "AI" sparkles, more than one burst/tilt/shout per view, tilted data, numbers without receipts, hype verbs, and filler three-up feature cards. Also keep Inter, Arial and JetBrains Mono off brand surfaces.

## Components

The source defines no component library, so this set was authored from the brand rules. Namespace: `window.FastAgentDesignSystem_3898e4`.

- **actions/:** Button (primary, secondary, ghost · sm, md, lg), Link
- **forms/:** Input, Select, Checkbox, Radio, Switch
- **display/:** Card, Tag, Sticker, Burst, FootnoteMark, Table, CodeBlock, Wordmark
- **feedback/:** Loader, Toast, Tooltip, Dialog
- **navigation/:** Tabs

Components that take asset paths (Burst, FootnoteMark, Wordmark, Loader) accept `src`, `sprite` or `assetBase` relative to the consuming page.

### Intentional additions
- **FootnoteMark, Table:** these carry the "receipts" rule, so a number and its disclaimer always travel together.
- **Sticker, Burst, Wordmark, Loader:** they package the brand's marks and motion so consumers don't have to rebuild them.
- **Switch** is squared (radius 6) because the brand forbids pills.
- **Dialog scrim** is flat petrol at 80%. A modal needs a scrim, and there's no blur.

## Fonts

Fraunces, Ultra, Figtree and DM Mono load from Google Fonts in `tokens/fonts.css`. There are no local binaries. Fraunces stands in until the custom wordmark lettering is drawn.

## Index

- `styles.css`: entry point (imports only)
- `tokens/`: colors, typography, shape (spacing, radii, borders, shadows, tilt), motion (easings, durations, keyframes), fonts
- `assets/`: sprite.svg, mono/, icon-tile (and -16, -amber), mark-chevron, burst-capsule (amber, orange, keyline), burst-star, burst-score, sparkle-atomic
- `guidelines/`: foundation specimen cards (colour, type, brand, spacing, motion, de-slop)
- `components/`: React primitives (see above)
- `ui_kits/website/`: fast-agent.ai redesign with Home, Benchmarks and Docs pages (see its README)
- `assets/illustration/`: character art (presenter default + smile/wink/smirk + amber, pointer, sitter ×2, bust-wink)
- `github.md`: source repo link (evalstate/fast-agent, docs/)
- `brand-overview.html`: all original specimen cards on one page
- `thumbnail.html`: project tile
- `SKILL.md`: agent-skill wrapper

## Caveats and open items

- No slides, because no deck was provided.
- Custom wordmark lettering: condensed, soft, vector, with a small-size cut.
- Chart colours for competitor series.
- UI icon set selection.
- Final licence check on fonts (all are Google Fonts, OFL).
