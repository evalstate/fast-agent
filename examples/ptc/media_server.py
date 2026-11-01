"""Media MCP server for the programmatic tool calling (PTC) evals.

Each swatch is a solid-colour PNG whose colour appears only in the pixels, never in text, so a
model can only name it if the image actually reached its context.

Run with:
    uv run media_server.py
"""

from __future__ import annotations

from io import BytesIO

from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from PIL import Image as PILImage

# Deliberately unrelated names: the colour must be read from the image.
_SWATCHES = {
    "swatch-north": (20, 160, 40),  # green
    "swatch-east": (130, 20, 170),  # purple
    "swatch-south": (240, 200, 0),  # yellow
}
_RESOURCE_SWATCH = (240, 120, 0)  # orange

app = FastMCP(name="media")


def _png(rgb: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    PILImage.new("RGB", (256, 256), rgb).save(buffer, format="PNG")
    return buffer.getvalue()


@app.tool(description="List the available swatch names.")
def list_swatches() -> list[str]:
    return sorted(_SWATCHES)


@app.tool(description="Render a swatch by name. Returns the swatch as a PNG image.")
def get_swatch(name: str) -> Image:
    if name not in _SWATCHES:
        raise ValueError(f"Unknown swatch {name!r}; call list_swatches")
    return Image(data=_png(_SWATCHES[name]), format="png")


@app.resource("swatch://west", mime_type="image/png", description="A swatch image resource.")
def west_swatch() -> bytes:
    return _png(_RESOURCE_SWATCH)


if __name__ == "__main__":
    app.run()
