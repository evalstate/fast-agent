#!/usr/bin/env python3
"""Render Terminal-Bench campaign storyboard assets from the homepage data."""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

DOCS_DIR = Path(__file__).resolve().parent
STORYBOARD_DIR = DOCS_DIR / "social_cards" / "benchmark"
STORYBOARD_PATH = STORYBOARD_DIR / "storyboard.html"
OUTPUT_DIR = STORYBOARD_DIR / "renders"
REVIEW_PATH = STORYBOARD_DIR / "review.html"
WIDTH = 1200
HEIGHT = 630
VARIANTS = {
    "spotlight": "GPT-5.6 Luna max · $19.61 complete run",
    "grok": "Grok 4.6 high · 87.19% with fast-agent",
    "grok-medium": "Grok 4.6 medium · 87.64% with fast-agent",
    "grok-share": "Grok 4.6 high · social-share alternative",
    "grok-share-swapped": "Grok 4.6 high · mirrored social-share alternative",
    "deepseek": "DeepSeek V4 Flash 0731 · $8.74 complete run",
    "pricing-convergence": "Normal pricing / Flex and off-peak convergence",
    "deepseek-cache-cost": "DeepSeek V4 Flash 0731 · cache pricing run cost",
    "task-swings": "Grok 4.6 vs GPT-5.6 Sol · major task swings",
}


def chrome_path() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return None


def render_variant(chrome: str, variant: str, *, high_quality: bool) -> int:
    suffix = "@2x" if high_quality else ""
    output = OUTPUT_DIR / f"terminal-bench-{variant}{suffix}.png"
    url = STORYBOARD_PATH.resolve().as_uri() + f"?variant={variant}"
    print(f"Generating {output.relative_to(DOCS_DIR)}")
    scale_arguments = ["--force-device-scale-factor=2"] if high_quality else []
    result = subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=1500",
            f"--window-size={WIDTH},{HEIGHT}",
            *scale_arguments,
            f"--screenshot={output}",
            url,
        ],
        cwd=DOCS_DIR,
        check=False,
    )
    if result.returncode != 0:
        return result.returncode
    with Image.open(output) as image:
        scale = 2 if high_quality else 1
        expected_size = (WIDTH * scale, HEIGHT * scale)
        if image.size != expected_size:
            print(f"Unexpected image size for {output}: {image.size}", file=sys.stderr)
            return 1
        if high_quality:
            delivery = OUTPUT_DIR / f"terminal-bench-{variant}-hq.png"
            image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(
                delivery,
                optimize=True,
                compress_level=9,
            )
            print(f"Generated {delivery.relative_to(DOCS_DIR)}")
        else:
            image.quantize(colors=192).save(output, optimize=True)
    return 0


def write_review() -> None:
    cards = []
    for variant, title in VARIANTS.items():
        filename = f"terminal-bench-{variant}.png"
        cards.append(
            f"""
      <article>
        <a href="renders/{html.escape(filename)}">
          <img src="renders/{html.escape(filename)}" alt="{html.escape(title)}">
        </a>
        <h2>{html.escape(title)}</h2>
        <p><a href="storyboard.html?variant={html.escape(variant)}">Open live 1200×630 card</a></p>
      </article>
"""
        )
    REVIEW_PATH.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Terminal-Bench social storyboard · fast-agent</title>
  <style>
    :root {{ color-scheme: dark; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 42px;
      background: #090a0d;
      color: #f0ece2;
      font: 15px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    header {{ max-width: 900px; margin-bottom: 32px; }}
    h1 {{ margin: 0; font-size: 38px; letter-spacing: -0.04em; }}
    header p {{ color: #8e897e; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 24px; }}
    article {{ padding: 14px; border: 1px solid #302f2c; background: #101216; }}
    article > a {{ display: block; }}
    img {{ display: block; width: 100%; height: auto; }}
    h2 {{ margin: 14px 0 0; color: #ffc649; font-size: 18px; }}
    article p {{ margin: 4px 0 0; }}
    a {{ color: #bbb5a8; }}
  </style>
</head>
<body>
  <header>
    <h1>Terminal-Bench social storyboard</h1>
    <p>Terminal-Bench benchmark spotlight and comparison campaign cards.</p>
  </header>
  <main>{"".join(cards)}</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(f"Wrote {REVIEW_PATH.relative_to(DOCS_DIR)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument(
        "--high-quality",
        action="store_true",
        help="write a lossless 2x master and supersampled 1200x630 delivery PNG",
    )
    args = parser.parse_args()
    chrome = chrome_path()
    if not chrome:
        print("google-chrome/chromium is required", file=sys.stderr)
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = [args.variant] if args.variant else list(VARIANTS)
    for variant in variants:
        result = render_variant(chrome, variant, high_quality=args.high_quality)
        if result != 0:
            return result
    write_review()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
