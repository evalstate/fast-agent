from __future__ import annotations

from typing import IO, TYPE_CHECKING, cast

from rich.color import Color
from rich.color_triplet import ColorTriplet
from rich.measure import Measurement
from rich.segment import Segment
from rich.style import Style
from textual_image._geometry import ImageSize
from textual_image._pixeldata import PixelData
from textual_image._terminal import CellSize, get_cell_size
from textual_image._utils import StrOrBytesPath, grouped

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from rich.console import Console, ConsoleOptions, RenderResult

_FALLBACK_CELL_SIZE = CellSize(8, 16)
_MIN_PLAUSIBLE_CELL_WIDTH_PX = 4
_MIN_PLAUSIBLE_CELL_HEIGHT_PX = 8


def _effective_cell_size() -> CellSize:
    cell_size = get_cell_size()
    if (
        cell_size.width < _MIN_PLAUSIBLE_CELL_WIDTH_PX
        or cell_size.height < _MIN_PLAUSIBLE_CELL_HEIGHT_PX
    ):
        return _FALLBACK_CELL_SIZE
    return cell_size


def _pixel_color(pixel: tuple[int, int, int]) -> Color:
    return Color.from_triplet(ColorTriplet(*pixel))


class HerdrAwareHalfcellImage:
    """Render half-cell images with sane sizing behind Herdr's terminal proxy."""

    def __init__(
        self,
        image: StrOrBytesPath | IO[bytes] | PILImage.Image,
        width: int | str | None = None,
        height: int | str | None = None,
    ) -> None:
        self._image_data = PixelData(image, mode="rgb")
        self._render_size = ImageSize(
            self._image_data.width,
            self._image_data.height,
            width,
            height,
        )

    def cleanup(self) -> None:
        """No-op."""

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        del console
        width, height = self._render_size.get_cell_size(
            options.max_width,
            options.max_height,
            _effective_cell_size(),
        )
        for upper_row, lower_row in grouped(self._image_data.scaled(width, height * 2), 2):
            for upper_pixel, lower_pixel in zip(upper_row, lower_row, strict=True):
                yield Segment(
                    "▀",
                    style=Style(
                        color=_pixel_color(cast("tuple[int, int, int]", upper_pixel)),
                        bgcolor=_pixel_color(cast("tuple[int, int, int]", lower_pixel)),
                    ),
                )
            yield Segment("\n")

    def __rich_measure__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> Measurement:
        del console
        width, _ = self._render_size.get_cell_size(
            options.max_width,
            options.max_height,
            _effective_cell_size(),
        )
        return Measurement(width, width)
