#!/usr/bin/env python3
"""Assess docs screenshots with deterministic checks and an optional vision judge."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
SCHEMA = DOCS_DIR / "visual_assessment.schema.json"

MODEL_IDS = {
    "spark": "codexspark",
    "gpt-5.5": "codexresponses.gpt-5.5?reasoning=medium",
    "gpt-5-5": "codexresponses.gpt-5.5?reasoning=medium",
    "sonnet": "sonnet",
    "haiku": "haiku",
}

EXPECTED_SIZES = {
    "live-home.png": (1440, 1200),
    "local-home.png": (1440, 1200),
    "local-home-mobile.png": (390, 900),
    "local-models.png": (1440, 1200),
}


@dataclass(frozen=True)
class Issue:
    screenshot: str
    severity: str
    category: str
    message: str


@dataclass(frozen=True)
class Metrics:
    screenshot: str
    width: int
    height: int
    grayscale_stddev: float
    unique_sample_colors: int
    dark_pixel_ratio: float
    bright_pixel_ratio: float
    blue_header_ratio: float


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots-dir", type=Path, default=DOCS_DIR / "screenshots")
    parser.add_argument("--out-dir", type=Path, default=DOCS_DIR / "visual-assessments")
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--vision", action="store_true", help="run the optional vision judge")
    parser.add_argument("--dry-run", action="store_true", help="write prompt/card without calling a model")
    parser.add_argument("--model", default="gpt-5.5", help="vision judge model id or alias")
    parser.add_argument("--fast-agent-env", type=Path, default=ROOT / ".cdx")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    screenshots = collect_screenshots(args.screenshots_dir)
    if not screenshots:
        raise SystemExit(f"No screenshots found in {args.screenshots_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    deterministic = assess_deterministic(screenshots)
    deterministic_path = args.out_dir / f"docs-visual-{args.run_id}-deterministic.json"
    deterministic_path.write_text(json.dumps(deterministic, indent=2), encoding="utf-8")

    print(f"Deterministic assessment: {deterministic_path}")
    for issue in deterministic["issues"]:
        print(f"{issue['severity']}: {issue['screenshot']}: {issue['message']}")

    if args.vision or args.dry_run:
        prompt = args.out_dir / f"docs-visual-{args.run_id}.md"
        card = args.out_dir / f"docs-visual-judge-{args.run_id}.md"
        prompt.write_text(render_prompt(screenshots, deterministic), encoding="utf-8")
        card.write_text(render_card(args.model), encoding="utf-8")
        print(f"Vision prompt: {prompt}")
        print(f"Vision card: {card}")

        if args.vision and not args.dry_run:
            run_vision_judge(
                prompt=prompt,
                card=card,
                out_dir=args.out_dir,
                run_id=args.run_id,
                fast_agent_env=args.fast_agent_env,
                timeout=args.timeout,
            )

    severe = [
        issue for issue in deterministic["issues"] if issue["severity"] in {"major", "critical"}
    ]
    return 1 if severe else 0


def collect_screenshots(screenshots_dir: Path) -> list[Path]:
    return sorted(screenshots_dir.glob("*.png"), key=screenshot_sort_key)


def screenshot_sort_key(path: Path) -> tuple[int, int, str]:
    name = path.name
    viewport_rank = 0 if "-mobile" not in name else 1
    match = re.search(r"-y(\d+)", name)
    offset = int(match.group(1)) if match else 0
    live_rank = 0 if name.startswith("local-") else 1
    return live_rank, viewport_rank, offset, name


def assess_deterministic(screenshots: list[Path]) -> dict[str, Any]:
    issues: list[Issue] = []
    metrics: list[Metrics] = []

    for screenshot in screenshots:
        item_metrics, item_issues = inspect_screenshot(screenshot)
        metrics.append(item_metrics)
        issues.extend(item_issues)

    names = {path.name for path in screenshots}
    for required in ("local-home.png", "local-home-mobile.png"):
        if required not in names:
            issues.append(
                Issue(
                    screenshot=required,
                    severity="major",
                    category="coverage",
                    message="Required home-page screenshot is missing.",
                )
            )

    return {
        "status": (
            "fail" if any(issue.severity in {"major", "critical"} for issue in issues) else "pass"
        ),
        "screenshots": [str(path.resolve()) for path in screenshots],
        "metrics": [asdict(metric) for metric in metrics],
        "issues": [asdict(issue) for issue in issues],
    }


def inspect_screenshot(path: Path) -> tuple[Metrics, list[Issue]]:
    issues: list[Issue] = []
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        sample = rgb.resize((min(width, 240), min(height, 240)))
        grayscale = sample.convert("L")
        stat = ImageStat.Stat(grayscale)
        grayscale_stddev = float(stat.stddev[0])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            pixels = list(sample.getdata())
        total = len(pixels)
        dark_ratio = sum(1 for r, g, b in pixels if r < 68 and g < 68 and b < 68) / total
        bright_ratio = sum(1 for r, g, b in pixels if r > 245 and g > 245 and b > 245) / total
        unique_colors = len(set(pixels))

        header = rgb.crop((0, 0, width, max(1, min(160, height // 5))))
        header_sample = header.resize((min(width, 240), min(header.height, 80)))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            header_pixels = list(header_sample.getdata())
        blue_header_ratio = (
            sum(
                1
                for r, g, b in header_pixels
                if b > r + 8 and g > r + 4 and 24 < (r + g + b) / 3 < 210
            )
            / len(header_pixels)
        )

    expected_size = EXPECTED_SIZES.get(path.name)
    if expected_size is not None and (width, height) != expected_size:
        issues.append(
            Issue(
                screenshot=path.name,
                severity="major",
                category="capture_dimensions",
                message=f"Expected {expected_size[0]}x{expected_size[1]}, got {width}x{height}.",
            )
        )

    if grayscale_stddev < 18 or unique_colors < 500:
        issues.append(
            Issue(
                screenshot=path.name,
                severity="critical",
                category="blank_or_unstyled",
                message="Screenshot has too little visual variation; it may be blank or unstyled.",
            )
        )

    if bright_ratio > 0.92 and grayscale_stddev < 35:
        issues.append(
            Issue(
                screenshot=path.name,
                severity="major",
                category="blank_or_unstyled",
                message="Screenshot is overwhelmingly white with little structure.",
            )
        )

    if path.name.startswith("local-home") and blue_header_ratio < 0.18:
        issues.append(
            Issue(
                screenshot=path.name,
                severity="major",
                category="brand_header",
                message="Home page header does not show enough blue brand area.",
            )
        )

    if path.name.startswith("local-home") and dark_ratio < 0.025:
        issues.append(
            Issue(
                screenshot=path.name,
                severity="minor",
                category="terminal_panel",
                message="Home page has little dark terminal area; check that CLI examples are visible.",
            )
        )

    return (
        Metrics(
            screenshot=path.name,
            width=width,
            height=height,
            grayscale_stddev=round(grayscale_stddev, 2),
            unique_sample_colors=unique_colors,
            dark_pixel_ratio=round(dark_ratio, 4),
            bright_pixel_ratio=round(bright_ratio, 4),
            blue_header_ratio=round(blue_header_ratio, 4),
        ),
        issues,
    )


def render_prompt(screenshots: list[Path], deterministic: dict[str, Any]) -> str:
    lines = [
        "---USER",
        "Assess these fast-agent documentation screenshots for publication readiness.",
        "",
        "Return only the structured JSON requested by the schema.",
        "",
        "Deterministic screenshot metrics:",
        json.dumps(deterministic, indent=2),
        "",
        "Rubric:",
        "- The home page should feel like a polished developer product, not a raw generated index.",
        "- Confirm the custom brand header is visible and visually integrated with the first viewport.",
        "- Confirm there are no raw Markdown artifacts such as literal `## Getting Started` text.",
        "- Check terminal examples and generated terminal captures for readable command text, "
        "prompt styling, and sufficient contrast.",
        "- Check text, buttons, cards, tables, tabs, and navigation for overlap, clipping, or "
        "incoherent wrapping.",
        "- On mobile, the `fast-agent` wordmark should not split awkwardly around the hyphen, "
        "and CTAs should remain tappable.",
        "- Feature copy should make the product clear quickly: uvx startup, MCP servers, "
        "workflows, model testing, provider aliases, ACP, and packaged examples.",
        "- Penalize pages that look monochrome, unstyled, mostly blank, or dominated by default theme chrome.",
        "- Prefer concrete visible evidence over inferred intent.",
        "",
        "Screenshots in attachment order:",
    ]
    for screenshot in screenshots:
        lines.append(f"- {screenshot.resolve()}")
    for screenshot in screenshots:
        lines.extend(["", "---RESOURCE", str(screenshot.resolve())])
    lines.append("")
    return "\n".join(lines)


def render_card(model: str) -> str:
    model_id = MODEL_IDS.get(model, model)
    return f"""---
type: smart
name: docs_visual_judge
model: {model_id}
shell: false
---

You are a strict but fair visual QA judge for fast-agent documentation screenshots.
Inspect only what is visible in the screenshots and produce structured output.
"""


def run_vision_judge(
    *,
    prompt: Path,
    card: Path,
    out_dir: Path,
    run_id: str,
    fast_agent_env: Path,
    timeout: int,
) -> None:
    raw = out_dir / f"docs-visual-{run_id}.raw.txt"
    structured = out_dir / f"docs-visual-{run_id}.json"
    cmd = [
        "uv",
        "run",
        "fast-agent",
        "go",
        "--env",
        str(fast_agent_env.resolve()),
        "--card",
        str(card),
        "--agent",
        "docs_visual_judge",
        "--no-shell",
        "--prompt-file",
        str(prompt),
        "--json-schema",
        str(SCHEMA),
        "--quiet",
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
        timeout=timeout,
        check=False,
    )
    raw.write_text(completed.stdout + "\n\nSTDERR:\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(f"Vision assessment failed; see {raw}")
    parsed = json.loads(completed.stdout)
    structured.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(f"Vision assessment: {structured}")


if __name__ == "__main__":
    raise SystemExit(main())
