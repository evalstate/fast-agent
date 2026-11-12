#!/usr/bin/env python3
"""Generate per-page Open Graph images for the docs site.

The site build only checks that these committed PNGs exist. Regeneration is a
local authoring step because Cloudflare Pages may not have Chrome available.
Cards are drawn from the brand tokens and assets in docs/docs/assets/forward.
"""

from __future__ import annotations

import argparse
import base64
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, get_args

import yaml
from PIL import Image

DOCS_DIR = Path(__file__).resolve().parent
CONTENT_DIR = DOCS_DIR / "docs"
OUTPUT_DIR = CONTENT_DIR / "assets" / "social"
FORWARD_DIR = CONTENT_DIR / "assets" / "forward"
SOCIAL_CARDS_DIR = DOCS_DIR / "social_cards"
TEMPLATE_PATH = SOCIAL_CARDS_DIR / "template.html"
STYLES_PATH = SOCIAL_CARDS_DIR / "styles.css"
THEMES_PATH = SOCIAL_CARDS_DIR / "themes.yml"
CONTACT_SHEET_PATH = SOCIAL_CARDS_DIR / "contact-sheet.html"
PREVIEWS_DIR = SOCIAL_CARDS_DIR / "previews"
WIDTH = 1200
HEIGHT = 630
MAX_BYTES = 1_000_000
REPO = "github.com/evalstate/fast-agent"
INSTALL = "uvx fast-agent-mcp@latest -x"

Design = Literal["hero", "sparkles", "burst", "presenter", "terminal"]
Scheme = Literal["ivory", "petrol"]
Pose = Literal["board", "palm"]
DESIGNS: tuple[Design, ...] = get_args(Design)
SCHEMES: tuple[Scheme, ...] = get_args(Scheme)
POSE_NAMES: tuple[Pose, ...] = get_args(Pose)

type Box = tuple[int, int, int, int]

POSE_SHEET = "illustration/presenter-approved-poses.png"
POSE_SHEET_WIDTH = 1536
# Crops (x0, y0, x1, y1) of the 3×2 approved pose sheet; `patch` hides the panel letter.
POSES: dict[Pose, tuple[Box, Box | None]] = {
    "board": ((58, 8, 508, 508), None),
    "palm": ((532, 8, 1018, 508), (532, 8, 600, 88)),
}
BOARD: Box = (287, 218, 458, 353)
PRESENTER_HEIGHT = 410
# Illustrative frontier on the presenter's board (same as the homepage): no data, no claim.
BOARD_CHART = (
    '<path d="M14 10V86H112" fill="none" stroke="#082C34" stroke-width="3" stroke-linecap="round"/>'
    '<g fill="#277C80" opacity=".5"><circle cx="40" cy="72" r="4"/><circle cx="63" cy="66" r="4"/>'
    '<circle cx="80" cy="48" r="4"/><circle cx="98" cy="57" r="4"/></g>'
    '<path d="M26 65Q39 36 56 27T104 14" fill="none" stroke="#F45125" stroke-width="4" '
    'stroke-linecap="round"/><g fill="#FFB52E" stroke="#082C34" stroke-width="2">'
    '<circle cx="26" cy="65" r="5"/><circle cx="56" cy="27" r="5"/><circle cx="104" cy="14" r="5"/></g>'
)


@dataclass(frozen=True)
class PageCard:
    source: Path
    output: Path
    title: str
    description: str
    section: str
    badge: str
    design: Design
    scheme: Scheme
    pose: Pose
    burst: str
    tagline: str

    @property
    def source_rel(self) -> Path:
        return self.source.relative_to(CONTENT_DIR)


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


def _choice[T: str](theme: dict[str, object], key: str, options: tuple[T, ...], rel: Path) -> T:
    value = _theme_value(theme, key, options[0])
    for option in options:
        if option == value:
            return option
    raise ValueError(f"{rel}: social {key} {value!r} is not one of {', '.join(options)}")


def load_themes() -> dict[str, object]:
    if not THEMES_PATH.exists():
        return {}
    themes = yaml.safe_load(THEMES_PATH.read_text(encoding="utf-8")) or {}
    return themes if isinstance(themes, dict) else {}


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
        default_title = (
            "fast-agent"
            if rel == Path("index.md")
            else source.stem.replace("_", " ").replace("-", " ").title()
        )
        title = (
            _theme_value(theme, "title")
            or _plain(meta.get("title"))
            or _title_from_body(body, default_title)
        )
        description = (
            _theme_value(theme, "description")
            or _plain(meta.get("description"))
            or _description_from_body(body)
        )
        section = "fast-agent" if len(rel.parts) == 1 else rel.parts[0].replace("_", " ")
        if rel.name == "index.md":
            output_rel = (
                Path("index.png") if rel.parent == Path(".") else rel.parent.with_suffix(".png")
            )
        else:
            output_rel = rel.with_suffix(".png")
        badge = _theme_value(theme, "badge", section.upper())
        cards.append(
            PageCard(
                source=source,
                output=OUTPUT_DIR / output_rel,
                title=title,
                description=description,
                section=section,
                badge=badge,
                design=_choice(theme, "design", DESIGNS, rel),
                scheme=_choice(theme, "scheme", SCHEMES, rel),
                pose=_choice(theme, "pose", POSE_NAMES, rel),
                burst=_theme_value(theme, "burst", badge),
                tagline=_theme_value(theme, "tagline", description, allow_blank=True),
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


def _asset(name: str) -> str:
    return (FORWARD_DIR / "assets" / name).as_uri()


def _mark(name: str, colour: str, style: str) -> str:
    # CSS masks are CORS-fetched, which file:// pages can't do; inline the SVG instead.
    svg = base64.b64encode((FORWARD_DIR / "assets" / name).read_bytes()).decode()
    src = f"url('data:image/svg+xml;base64,{svg}')"
    return f'<span class="mark {colour}" style="--src: {src}; {style}"></span>'


def _sparkle(colour: str, size: int, style: str) -> str:
    return _mark("sparkle-atomic.svg", colour, f"width:{size}px;height:{size}px;{style}")


def _presenter_html(pose: Pose) -> str:
    (x0, y0, x1, y1), patch = POSES[pose]
    scale = PRESENTER_HEIGHT / (y1 - y0)

    def rect(box: Box) -> str:
        left, top, right, bottom = box
        return (
            f"left:{(left - x0) * scale:.1f}px;top:{(top - y0) * scale:.1f}px;"
            f"width:{(right - left) * scale:.1f}px;height:{(bottom - top) * scale:.1f}px"
        )

    extras = f'<span class="patch" style="{rect(patch)}"></span>' if patch else ""
    if pose == "board":
        extras += (
            f'<svg class="chart" viewBox="0 0 120 100" style="{rect(BOARD)}">{BOARD_CHART}</svg>'
        )
    return f"""
  <div class="art" style="width:{(x1 - x0) * scale:.0f}px;height:{PRESENTER_HEIGHT}px">
    <img src="{_asset(POSE_SHEET)}" alt="" style="width:{POSE_SHEET_WIDTH * scale:.1f}px;left:{-x0 * scale:.1f}px;top:{-y0 * scale:.1f}px">
    {extras}
  </div>
  {_sparkle("amber", 64, "right:470px;top:150px;transform:rotate(12deg)")}"""


def _art_html(card: PageCard) -> str:
    match card.design:
        case "sparkles":
            return f"""
  <div class="art">
    {_sparkle("teal", 230, "right:110px;top:150px;transform:rotate(12deg)")}
    {_sparkle("amber", 96, "right:380px;top:150px")}
    {_sparkle("ink", 54, "right:84px;top:430px;transform:rotate(-12deg)")}
    {_sparkle("teal", 40, "right:360px;top:430px")}
  </div>"""
        case "burst":
            size = 92 if len(card.burst) <= 3 else 80 if len(card.burst) <= 6 else 54
            return f"""
  <div class="art">
    {_mark("burst-score.svg", "amber", "inset:0;transform:rotate(28deg)")}
    <span class="shout" style="--shout-size:{size}px">{html.escape(card.burst)}</span>
  </div>
  {_sparkle("teal", 56, "right:470px;top:440px")}"""
        case "presenter":
            return _presenter_html(card.pose)
        case "terminal":
            return f"""
  <div class="art"><img src="{_asset("illustration/space-age-terminal-paddles.png")}" alt=""></div>
  {_sparkle("teal", 64, "right:440px;top:140px;transform:rotate(12deg)")}"""
        case "hero":
            return ""


def _title_size(title: str) -> int:
    n = len(title)
    return 88 if n <= 16 else 76 if n <= 28 else 64 if n <= 44 else 54


def _content_html(card: PageCard) -> str:
    tile = "icon-tile-amber.svg" if card.scheme == "petrol" else "icon-tile.svg"
    tagline = html.escape(card.tagline)
    lede = f'<p class="lede">{tagline}</p>' if tagline else ""
    route = _route(card)
    repo = f'<span class="repo">{REPO}</span>' if len(route) <= 48 else ""
    footer = f"""
<footer><span class="url">{html.escape(route)}</span>{repo}</footer>"""
    if card.design == "hero":
        return f"""
<div class="hero">
  <div class="hero-mark">
    {_mark("burst-score.svg", "amber", "inset:0;transform:rotate(28deg)")}
    <h1>fast-<br>agent</h1>
  </div>
  <div class="hero-copy">
    <p class="title">{tagline}</p>
    <p class="lede">{html.escape(card.description)}</p>
    <code class="install">{INSTALL}</code>
  </div>
  {_sparkle("teal", 44, "left:60px;top:70px")}
  {_sparkle("teal", 34, "left:470px;top:420px;transform:rotate(-12deg)")}
</div>{footer}"""
    label = (
        ""
        if card.design == "burst"
        else f'<span class="label"><i></i>{html.escape(card.badge.upper())}</span>'
    )
    return f"""
<header>
  <img class="tile" src="{_asset(tile)}" alt="">
  <span class="wordmark">fast-agent</span>
  {label}
</header>
<div class="body">
  <h1 class="title" style="--title-size:{_title_size(card.title)}px">{html.escape(card.title)}</h1>
  {lede}
</div>{_art_html(card)}{footer}"""


def _card_html(card: PageCard) -> str:
    return _render_template(
        TEMPLATE_PATH.read_text(encoding="utf-8"),
        {
            "assets_uri": FORWARD_DIR.as_uri(),
            "card_class": f"card d-{card.design} scheme-{card.scheme}",
            "content_html": _content_html(card),
            "stylesheet_uri": STYLES_PATH.resolve().as_uri(),
            "title": html.escape(card.title),
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
            html_path = tmpdir / (
                card.output.relative_to(OUTPUT_DIR).as_posix().replace("/", "__") + ".html"
            )
            html_path.write_text(_card_html(card), encoding="utf-8")
            print(f"Generating {card.output.relative_to(DOCS_DIR)}")
            result = subprocess.run(
                [
                    chrome,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--hide-scrollbars",
                    "--virtual-time-budget=10000",
                    f"--window-size={WIDTH},{HEIGHT}",
                    f"--screenshot={card.output}",
                    html_path.as_uri(),
                ],
                cwd=DOCS_DIR,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                sys.stderr.write(result.stderr.decode(errors="replace"))
                return result.returncode
            with Image.open(card.output) as image:
                quantized = image.convert("RGB").quantize(colors=256)
            quantized.save(card.output, optimize=True)
    return 0


def write_design_previews(cards: list[PageCard]) -> None:
    """Write one HTML preview per design × scheme, for choosing themes."""
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    by_rel = {card.source_rel.as_posix(): card for card in cards}
    samples: dict[Design, PageCard] = {
        "hero": by_rel["index.md"],
        "sparkles": by_rel["agents/index.md"],
        "burst": by_rel["mcp/index.md"],
        "presenter": by_rel["benchmarks/index.md"],
        "terminal": by_rel["getting_started.md"],
    }
    variants = [
        replace(card, design=design, scheme=scheme, pose=pose)
        for design, card in samples.items()
        for pose in (POSE_NAMES if design == "presenter" else (card.pose,))
        for scheme in SCHEMES
    ]
    links = []
    for card in variants:
        name = (
            f"{card.design}-{card.pose}-{card.scheme}"
            if card.design == "presenter"
            else (f"{card.design}-{card.scheme}")
        )
        path = PREVIEWS_DIR / f"{name}.html"
        path.write_text(_card_html(card), encoding="utf-8")
        links.append(
            f'<article><div class="preview"><iframe src="{path.name}"></iframe></div>'
            f"<h2>{html.escape(name)}</h2></article>"
        )
    index = PREVIEWS_DIR / "designs.html"
    index.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>fast-agent social card designs</title>
  <style>
    body {{ margin: 0; padding: 40px; background: #F4EAD5; color: #082C34; font: 15px/1.5 system-ui, sans-serif; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 24px; }}
    .preview {{ container-type: inline-size; aspect-ratio: 1200 / 630; overflow: hidden; border: 2px solid #082C34; border-radius: 14px; }}
    iframe {{ width: 1200px; height: 630px; border: 0; transform: scale(calc(100cqw / 1200)); transform-origin: 0 0; }}
    h2 {{ margin: 10px 0 0; font-size: 16px; }}
  </style>
</head>
<body>
  <h1>Social card designs</h1>
  <div class="grid">{"".join(links)}</div>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(f"Wrote {index.relative_to(DOCS_DIR)}")


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
            theme = html.escape(f"{card.design} / {card.scheme}")
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
      --bg: #F4EAD5;
      --panel: #FFFCF4;
      --text: #082C34;
      --muted: rgba(8, 44, 52, .72);
      --line: #082C34;
      --accent: #8A5A00;
      --warn: #F45125;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 48px;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.5 system-ui, sans-serif;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: end;
      margin-bottom: 40px;
      border-bottom: 2px solid var(--line);
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
      border: 2px solid var(--line);
      border-radius: 14px;
      background: var(--panel);
    }}
    .card.warn, .card.missing {{ border-color: color-mix(in srgb, var(--warn), transparent 35%); }}
    .thumb {{
      display: block;
      aspect-ratio: 1200 / 630;
      border-bottom: 2px solid var(--line);
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
    parser.add_argument(
        "--contact-sheet", action="store_true", help="only write the HTML contact sheet"
    )
    parser.add_argument(
        "--variant-previews",
        action="store_true",
        help="write HTML previews of each card design and scheme",
    )
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
        write_design_previews(all_cards)
        return 0
    if args.check:
        return check(cards, check_stale=args.page is None)
    result = render(cards)
    if result == 0:
        write_contact_sheet(all_cards)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
