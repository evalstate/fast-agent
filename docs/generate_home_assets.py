"""Export display-sized homepage artwork; keep the original PNGs for authoring.

Run with: uv run docs/generate_home_assets.py
"""

from pathlib import Path

from PIL import Image

ARTWORK = Path(__file__).parent / "docs/assets/forward/assets/illustration"


def main() -> None:
    with Image.open(ARTWORK / "presenter-approved-poses.png") as image:
        # The original CSS showed this 450 × 500 crop of the pose sheet.
        presenter = image.resize((450, 500), Image.Resampling.LANCZOS, box=(58.5, 8, 508.5, 508))
        presenter.save(ARTWORK / "presenter-approved.webp", quality=85, method=6)
    with Image.open(ARTWORK / "space-age-terminal-paddles.png") as image:
        # Twice the maximum CSS width, preserving the lamp overlay coordinates.
        image.thumbnail((580, 580), Image.Resampling.LANCZOS)
        image.save(ARTWORK / "space-age-terminal.webp", quality=85, method=6)


if __name__ == "__main__":
    main()
