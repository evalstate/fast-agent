#!/usr/bin/env python3
"""Generate per-page Open Graph images for the docs site.

The site build only checks that these committed PNGs exist. Regeneration is a
local authoring step because Cloudflare Pages may not have Chrome available.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

DOCS_DIR = Path(__file__).resolve().parent
CONTENT_DIR = DOCS_DIR / "docs"
OUTPUT_DIR = CONTENT_DIR / "assets" / "social"
SOCIAL_CARDS_DIR = DOCS_DIR / "social_cards"
TEMPLATE_PATH = SOCIAL_CARDS_DIR / "template.html"
STYLES_PATH = SOCIAL_CARDS_DIR / "styles.css"
THEMES_PATH = SOCIAL_CARDS_DIR / "themes.yml"
CONTACT_SHEET_PATH = SOCIAL_CARDS_DIR / "contact-sheet.html"
PREVIEWS_DIR = SOCIAL_CARDS_DIR / "previews"
WORDMARK_PATH = SOCIAL_CARDS_DIR / "wordmark.svg"
PROJECT_ROOT = DOCS_DIR.parent
WIDTH = 1200
HEIGHT = 630
MAX_BYTES = 1_000_000


@dataclass(frozen=True)
class PageCard:
    source: Path
    output: Path
    title: str
    description: str
    section: str
    badge: str
    accent: str
    accent_soft: str
    motif: str
    variant: str
    background: str
    glyph_position: str
    bg_intensity: str
    tagline: str

    @property
    def source_rel(self) -> Path:
        return self.source.relative_to(CONTENT_DIR)

    @property
    def section_key(self) -> str:
        return self.source_rel.parts[0]


def _frontmatter(markdown: str) -> tuple[dict[str, object], str]:
    if not markdown.startswith("---\n"):
        return {}, markdown
    end = markdown.find("\n---", 4)
    if end == -1:
        return {}, markdown
    meta = yaml.safe_load(markdown[4:end]) or {}
    body = markdown[end + 4 :]
    return meta if isinstance(meta, dict) else {}, body


def _plain(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("`", "")
    return " ".join(text.split())


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _theme_value(
    theme: dict[str, object],
    key: str,
    fallback: str = "",
    *,
    allow_blank: bool = False,
) -> str:
    if key in theme:
        value = _plain(theme.get(key))
        if value or allow_blank:
            return value
    return fallback


def load_themes() -> dict[str, object]:
    if not THEMES_PATH.exists():
        return {}
    themes = yaml.safe_load(THEMES_PATH.read_text(encoding="utf-8")) or {}
    return themes if isinstance(themes, dict) else {}


def project_version() -> str:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = _mapping(pyproject.get("project"))
    return "v" + _theme_value(project, "version", "0.0.0")


def _card_theme(themes: dict[str, object], rel: Path, meta: dict[str, object]) -> dict[str, object]:
    default = _mapping(themes.get("default"))
    sections = _mapping(themes.get("sections"))
    pages = _mapping(themes.get("pages"))
    section = _mapping(sections.get(rel.parts[0]))
    page = _mapping(pages.get(rel.as_posix()))
    social = _mapping(meta.get("social"))
    return default | section | page | social


def _title_from_body(body: str, fallback: str) -> str:
    for line in body.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return _plain(match.group(1))
    return fallback


def _description_from_body(body: str) -> str:
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "<", "```", "---", "!", "[")):
            continue
        return _plain(line)
    return "MCP-native agents, workflows, and servers."


def discover_cards() -> list[PageCard]:
    themes = load_themes()
    cards: list[PageCard] = []
    for source in sorted(CONTENT_DIR.rglob("*.md")):
        rel = source.relative_to(CONTENT_DIR)
        if rel.parts[0] in {"_generated", "assets"}:
            continue
        markdown = source.read_text(encoding="utf-8")
        meta, body = _frontmatter(markdown)
        theme = _card_theme(themes, rel, meta)
        default_title = "fast-agent" if rel == Path("index.md") else source.stem.replace("_", " ").replace("-", " ").title()
        title = _theme_value(theme, "title") or _plain(meta.get("title")) or _title_from_body(body, default_title)
        description = _theme_value(theme, "description") or _plain(meta.get("description")) or _description_from_body(body)
        section = "fast-agent" if len(rel.parts) == 1 else rel.parts[0].replace("_", " ")
        is_section_index = rel.name == "index.md" and rel.parent != Path(".")
        default_variant = "hero" if rel == Path("index.md") else "section" if is_section_index else "doc"
        if rel.name == "index.md":
            output_rel = Path("index.png") if rel.parent == Path(".") else rel.parent.with_suffix(".png")
        else:
            output_rel = rel.with_suffix(".png")
        output = OUTPUT_DIR / output_rel
        cards.append(
            PageCard(
                source,
                output,
                title,
                description,
                section,
                _theme_value(theme, "section") or _theme_value(theme, "badge", section.upper()),
                _theme_value(theme, "accent", "#f5a400"),
                _theme_value(theme, "accent_soft", "#ffcf5a"),
                _theme_value(theme, "motif", "protocol-grid"),
                _theme_value(theme, "variant", default_variant),
                _theme_value(theme, "background", "glyph"),
                _theme_value(theme, "glyph_position"),
                _theme_value(theme, "bg_intensity", "12"),
                _theme_value(theme, "tagline", description, allow_blank=True),
            )
        )
    return cards


def _render_template(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace("{{ " + key + " }}", value)
    return template


def _route(card: PageCard) -> str:
    rel = card.source_rel
    if rel == Path("index.md"):
        return "fast-agent.ai"
    if rel.name == "index.md":
        route = rel.parent.as_posix()
    else:
        route = rel.with_suffix("").as_posix()
    return "fast-agent.ai/" + route


def _glyph_position(card: PageCard) -> str:
    if card.glyph_position:
        return card.glyph_position
    if card.variant == "hero":
        return "center"
    if card.variant == "section":
        return "left"
    return "right"


def _background_html(card: PageCard) -> str:
    glyph_position = html.escape(_glyph_position(card))
    return f"""
  <div class="glyph-bg {glyph_position}">
    <svg viewBox="0 0 256 256" aria-hidden="true">
      <path d="M94.340 208L131.300 208L161 128.140L131.300 47.400L94.340 47.400L124.040 127.700L94.340 208Z" />
    </svg>
  </div>
  <div class="tui-bg">
    <div class="inner {glyph_position if glyph_position in {'left', 'center'} else 'bottom-right'}">
<span class="row dimx">~/research → <span class="cyan bold">gpt-5</span> ⇔ (22.1%)</span>
<span class="row"><span class="green bold">▸ commentary</span></span>
<span class="row dimx">  Found the MCP spec — fetching tool list…</span>
<span class="row"><span class="magenta bold">▾ tool call</span> · <span class="cyan">fetch__get_page</span></span>
<span class="row dimx">  ✓ 200 OK · 14.2 KB · 412ms</span>
<span class="row dimx">┌─ MCP ──────────────────  req  resp  notif</span>
<span class="row">│ <span class="blue">◀</span> GET  (SSE)   <span class="dimx">·······</span>  -   12    8</span>
<span class="row">│ <span class="red">▶</span> POST (JSON)  <span class="dimx">·······</span>  3    3    0</span>
<span class="row">│ <span class="green">●</span> HEALTH       <span class="dimx">·······</span>  <span class="green">ok</span></span>
<span class="row dimx">└</span>
    </div>
  </div>
"""


def _content_html(card: PageCard, brand_uri: str, version: str) -> str:
    title = html.escape(card.title)
    section = html.escape(card.badge.upper())
    tagline = html.escape(card.tagline)
    lede = f'<p class="lede">{tagline}</p>' if tagline else ""
    hero_tagline = f'<p class="hero-tagline">{tagline}</p>' if tagline else ""
    url = html.escape(_route(card))
    version = html.escape(version)
    if card.variant == "hero":
        return f"""
  <div class="topstrip">
    <span class="section"><span class="acc">·</span>{section}</span>
  </div>
  <div class="body hero-body">
    <img class="brand-hero phosphor-img subtle" src="{brand_uri}" alt="fast-agent">
    {hero_tagline}
  </div>
  <div class="bottomstrip center">
    <span class="url"><span class="slash">/</span>{url}</span>
    <span class="meta">github.com/evalstate/fast-agent</span>
    <span class="version">{version}</span>
  </div>
"""
    if card.variant == "section":
        return f"""
  <div class="topstrip">
    <span class="section-eyebrow"><span class="bar"></span>{section}</span>
    <img class="brand phosphor-img subtle" src="{brand_uri}" alt="fast-agent" style="margin-left:auto">
  </div>
  <div class="body">
    <h1 class="title">{title}</h1>
    {lede}
  </div>
  <div class="bottomstrip">
    <span class="url"><span class="slash">/</span>{url}</span>
    <span class="spacer"></span>
    <span class="version">{version}</span>
  </div>
"""
    return f"""
  <div class="topstrip">
    <img class="brand phosphor-img subtle" src="{brand_uri}" alt="fast-agent">
    <span class="section"><span class="acc">/</span>{section}</span>
  </div>
  <div class="body">
    <h1 class="title">{title}</h1>
    {lede}
  </div>
  <div class="bottomstrip">
    <span class="url"><span class="slash">/</span>{url}</span>
    <span class="spacer"></span>
    <span class="version">{version}</span>
  </div>
"""


def _bg_intensity(value: str) -> str:
    try:
        number = float(value)
    except ValueError:
        number = 12
    if number > 1:
        number = number / 100
    return str(max(0, min(number, 0.3)))


def _card_html(
    card: PageCard,
    *,
    stylesheet_uri: str | None = None,
    brand_uri: str | None = None,
) -> str:
    variant = card.variant if card.variant in {"doc", "hero", "section"} else "doc"
    background = card.background if card.background in {"glyph", "tui", "none"} else "glyph"
    brand_uri = brand_uri or WORDMARK_PATH.resolve().as_uri()
    version = project_version()
    return _render_template(
        TEMPLATE_PATH.read_text(encoding="utf-8"),
        {
            "accent": card.accent,
            "accent_soft": card.accent_soft,
            "background_html": _background_html(card),
            "badge": html.escape(card.badge.upper()),
            "bg_intensity": _bg_intensity(card.bg_intensity),
            "brand_uri": brand_uri,
            "card_class": f"card v-page v-{variant} bg-{background}",
            "content_html": _content_html(card, brand_uri, version),
            "description": html.escape(card.description),
            "motif": html.escape(card.motif),
            "route": html.escape(_route(card)),
            "section": html.escape(card.section.upper()),
            "stylesheet_uri": stylesheet_uri or STYLES_PATH.resolve().as_uri(),
            "title": html.escape(card.title),
            "variant": html.escape(card.variant),
        },
    )


def chrome_path() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return None


def render(cards: list[PageCard]) -> int:
    chrome = chrome_path()
    if not chrome:
        print("google-chrome/chromium is required to generate social cards", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="fast-agent-social-") as tmp:
        tmpdir = Path(tmp)
        for card in cards:
            card.output.parent.mkdir(parents=True, exist_ok=True)
            html_path = tmpdir / (card.output.relative_to(OUTPUT_DIR).as_posix().replace("/", "__") + ".html")
            html_path.write_text(_card_html(card), encoding="utf-8")
            print(f"Generating {card.output.relative_to(DOCS_DIR)}")
            result = subprocess.run(
                [
                    chrome,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--hide-scrollbars",
                    f"--window-size={WIDTH},{HEIGHT}",
                    f"--screenshot={card.output}",
                    html_path.as_uri(),
                ],
                cwd=DOCS_DIR,
            )
            if result.returncode != 0:
                return result.returncode
            image = Image.open(card.output)
            image.quantize(colors=160).save(card.output, optimize=True)
    return 0


def write_variant_previews(cards: list[PageCard]) -> None:
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    sample = next((card for card in cards if card.source_rel == Path("index.md")), cards[0])
    variants = ["doc", "hero", "section"]
    stylesheet_uri = os.path.relpath(STYLES_PATH, PREVIEWS_DIR)
    brand_uri = os.path.relpath(WORDMARK_PATH, PREVIEWS_DIR)
    links = []
    for variant in variants:
        card = PageCard(
            sample.source,
            sample.output,
            sample.title,
            sample.description,
            sample.section,
            sample.badge,
            sample.accent,
            sample.accent_soft,
            sample.motif,
            variant,
            "glyph",
            "",
            "7" if variant == "hero" else "12",
            "Simple, extendable agents." if variant == "hero" else sample.description,
        )
        path = PREVIEWS_DIR / f"{variant}.html"
        path.write_text(_card_html(card, stylesheet_uri=stylesheet_uri, brand_uri=brand_uri), encoding="utf-8")
        links.append(
            f"""
            <article>
              <div class="preview">
                <iframe src="{html.escape(path.name)}"></iframe>
              </div>
              <h2>{html.escape(variant)}</h2>
              <a href="{html.escape(path.name)}">open full size</a>
            </article>
            """
        )
    (PREVIEWS_DIR / "crt-variants.html").write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>fast-agent social card variants</title>
  <style>
    body {{
      margin: 0;
      padding: 40px;
      background: #080b11;
      color: #eef4ff;
      font: 15px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    h1 {{ margin: 0 0 28px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 24px;
    }}
    article {{
      padding: 16px;
      border: 1px solid rgba(238,244,255,.16);
      border-radius: 18px;
      background: #101722;
    }}
    .preview {{
      container-type: inline-size;
      aspect-ratio: 1200 / 630;
      overflow: hidden;
      border-radius: 12px;
      background: #000;
    }}
    iframe {{
      width: 1200px;
      height: 630px;
      border: 0;
      transform: scale(calc(100cqw / 1200));
      transform-origin: 0 0;
    }}
    h2 {{ margin: 14px 0 4px; color: #f5a400; }}
    a {{ color: #9ed2ff; }}
  </style>
</head>
<body>
  <h1>Social card variants</h1>
  <div class="grid">{"".join(links)}</div>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(f"Wrote {(PREVIEWS_DIR / 'crt-variants.html').relative_to(DOCS_DIR)}")


def _image_status(path: Path) -> tuple[str, str, str]:
    if not path.exists():
        return "missing", "—", "—"
    size = path.stat().st_size
    try:
        with Image.open(path) as image:
            dimensions = f"{image.size[0]}×{image.size[1]}"
            status = "ok" if image.size == (WIDTH, HEIGHT) and size <= MAX_BYTES else "warn"
    except OSError:
        dimensions = "unreadable"
        status = "warn"
    return status, dimensions, f"{size / 1024:.0f} KB"


def write_contact_sheet(cards: list[PageCard]) -> None:
    groups: dict[str, list[PageCard]] = {}
    for card in cards:
        groups.setdefault(card.section, []).append(card)

    sections = []
    for section, section_cards in groups.items():
        rows = []
        for card in section_cards:
            status, dimensions, size = _image_status(card.output)
            image_src = html.escape(os.path.relpath(card.output, SOCIAL_CARDS_DIR))
            source = html.escape(card.source_rel.as_posix())
            output = html.escape(card.output.relative_to(DOCS_DIR).as_posix())
            title = html.escape(card.title)
            badge = html.escape(card.badge)
            theme = html.escape(f"{card.variant} / {card.motif}")
            thumb = (
                f'<img src="{image_src}" alt="{title}">'
                if card.output.exists()
                else '<div class="missing-thumb">missing</div>'
            )
            rows.append(
                f"""
                <article class="card {status}">
                  <a class="thumb" href="{image_src}">{thumb}</a>
                  <div class="meta">
                    <h3>{title}</h3>
                    <dl>
                      <div><dt>Source</dt><dd>{source}</dd></div>
                      <div><dt>Output</dt><dd>{output}</dd></div>
                      <div><dt>Badge</dt><dd>{badge}</dd></div>
                      <div><dt>Theme</dt><dd>{theme}</dd></div>
                      <div><dt>Status</dt><dd><span class="pill">{status}</span></dd></div>
                      <div><dt>Size</dt><dd>{dimensions} · {size}</dd></div>
                    </dl>
                  </div>
                </article>
                """
            )
        sections.append(
            f"""
            <section>
              <h2>{html.escape(section.title())}</h2>
              <div class="grid">{"".join(rows)}</div>
            </section>
            """
        )

    CONTACT_SHEET_PATH.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>fast-agent social cards</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #080b11;
      --panel: #101722;
      --panel-2: #151e2c;
      --text: #eef4ff;
      --muted: #9da9ba;
      --line: rgba(238, 244, 255, .14);
      --accent: #f5a400;
      --warn: #fb7185;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 48px;
      background:
        radial-gradient(circle at top right, rgba(245, 164, 0, .18), transparent 34rem),
        linear-gradient(135deg, #080b11, #0d121b);
      color: var(--text);
      font: 15px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: end;
      margin-bottom: 40px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 24px;
    }}
    h1, h2, h3 {{ margin: 0; line-height: 1.05; }}
    h1 {{ font-size: 42px; letter-spacing: -.04em; }}
    h2 {{ margin: 42px 0 18px; color: var(--accent); font-size: 24px; }}
    h3 {{ font-family: ui-sans-serif, system-ui, sans-serif; font-size: 20px; letter-spacing: -.02em; }}
    .summary {{ color: var(--muted); text-align: right; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 18px;
    }}
    .card {{
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: linear-gradient(180deg, var(--panel), var(--panel-2));
      box-shadow: 0 18px 44px rgba(0,0,0,.25);
    }}
    .card.warn, .card.missing {{ border-color: color-mix(in srgb, var(--warn), transparent 35%); }}
    .thumb {{
      display: block;
      aspect-ratio: 1200 / 630;
      background: #06090f;
      border-bottom: 1px solid var(--line);
      color: inherit;
      text-decoration: none;
    }}
    img {{ display: block; width: 100%; height: 100%; object-fit: cover; }}
    .missing-thumb {{
      display: grid;
      height: 100%;
      place-items: center;
      color: var(--warn);
      font-size: 22px;
      text-transform: uppercase;
      letter-spacing: .16em;
    }}
    .meta {{ padding: 18px; }}
    dl {{ display: grid; gap: 8px; margin: 16px 0 0; }}
    dl div {{
      display: grid;
      grid-template-columns: 76px 1fr;
      gap: 12px;
      min-width: 0;
    }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    .pill {{
      display: inline-block;
      padding: 2px 8px;
      border: 1px solid currentColor;
      border-radius: 999px;
      color: var(--accent);
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: .08em;
    }}
    .warn .pill, .missing .pill {{ color: var(--warn); }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>fast-agent social cards</h1>
      <p>Generated review sheet for committed Open Graph/Twitter images.</p>
    </div>
    <div class="summary">{len(cards)} cards · {WIDTH}×{HEIGHT}px target · {MAX_BYTES // 1000} KB max</div>
  </header>
  {"".join(sections)}
</body>
</html>
""",
        encoding="utf-8",
    )
    print(f"Wrote {CONTACT_SHEET_PATH.relative_to(DOCS_DIR)}")


def _matching_card(cards: list[PageCard], page: str) -> list[PageCard]:
    page_path = Path(page)
    matches = [card for card in cards if card.source_rel == page_path]
    if matches:
        return matches
    matches = [card for card in cards if card.source_rel.as_posix() == page]
    if matches:
        return matches
    print(f"No docs page found for {page}", file=sys.stderr)
    return []


def check(cards: list[PageCard], *, check_stale: bool = True) -> int:
    failures = 0
    expected = {card.output for card in cards}
    missing = [path for path in expected if not path.exists()]
    stale = sorted(OUTPUT_DIR.rglob("*.png")) if check_stale and OUTPUT_DIR.exists() else []
    stale = [path for path in stale if path not in expected]

    for card in cards:
        if not card.output.exists():
            continue
        with Image.open(card.output) as image:
            if image.size != (WIDTH, HEIGHT):
                print(
                    f"Wrong social card size for {card.output.relative_to(DOCS_DIR)}: "
                    f"{image.size[0]}x{image.size[1]}",
                    file=sys.stderr,
                )
                failures += 1
        size = card.output.stat().st_size
        if size > MAX_BYTES:
            print(
                f"Social card exceeds {MAX_BYTES:,} bytes: {card.output.relative_to(DOCS_DIR)} "
                f"({size:,} bytes)",
                file=sys.stderr,
            )
            failures += 1

    if missing:
        failures += len(missing)
        print("Missing social card images. Regenerate locally with:", file=sys.stderr)
        print("  uv run scripts/docs.py social", file=sys.stderr)
        for path in sorted(missing):
            print(f"  - {path.relative_to(DOCS_DIR)}", file=sys.stderr)

    if stale:
        failures += len(stale)
        print("Stale social card images for deleted pages:", file=sys.stderr)
        for path in stale:
            print(f"  - {path.relative_to(DOCS_DIR)}", file=sys.stderr)

    if failures == 0:
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="only verify committed cards exist")
    parser.add_argument("--contact-sheet", action="store_true", help="only write the HTML contact sheet")
    parser.add_argument("--variant-previews", action="store_true", help="write CRT design variant previews")
    parser.add_argument("--page", help="render/check one page, e.g. guides/codex.md")
    args = parser.parse_args()
    all_cards = discover_cards()
    cards = all_cards
    if args.page:
        cards = _matching_card(cards, args.page)
        if not cards:
            return 1
    if args.contact_sheet:
        write_contact_sheet(all_cards)
        return 0
    if args.variant_previews:
        write_variant_previews(all_cards)
        return 0
    if args.check:
        return check(cards, check_stale=args.page is None)
    result = render(cards)
    if result == 0:
        write_contact_sheet(all_cards)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
