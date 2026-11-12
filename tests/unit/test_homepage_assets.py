"""Keep homepage artwork within a cold-load budget without relying on compression."""

from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

CONTENT = Path(__file__).resolve().parents[2] / "docs/docs"


class Homepage(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[dict[str, str | None]] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "img":
            self.images.append(attributes)
        if identifier := attributes.get("id"):
            self.ids.add(identifier)


def homepage() -> Homepage:
    page = Homepage()
    page.feed((CONTENT / "index.md").read_text())
    return page


def test_homepage_artwork_budget_and_intrinsic_dimensions() -> None:
    images = homepage().images
    assert images
    total_bytes = 0
    for attributes in images:
        src = attributes["src"]
        assert src is not None
        path = CONTENT / src
        total_bytes += path.stat().st_size
        if path.suffix == ".svg":
            continue
        with Image.open(path) as image:
            width = int(attributes.get("width") or "0")
            height = int(attributes.get("height") or "0")
            assert width > 0 and height > 0
            assert image.width * height == image.height * width
    # Includes provider/protocol logos, not just the two large illustrations.
    assert total_bytes < 150_000


def test_homepage_skip_link_has_a_target() -> None:
    assert "__skip" in homepage().ids
