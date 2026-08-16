from __future__ import annotations

from typing import TYPE_CHECKING

from rich.measure import Measurement
from textual_image._terminal import get_cell_size
from textual_image.renderable.sixel import Image as SixelImage

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult


class ViewportAwareSixelImage(SixelImage):
    """Keep Sixel's placeholder rewind within the terminal viewport."""

    def _fitted_image(self, options: ConsoleOptions) -> SixelImage | None:
        safe_height = options.max_height - 1
        if safe_height < 1:
            return None

        terminal_size = get_cell_size()
        _, height = self._render_size.get_cell_size(
            options.max_width,
            options.max_height,
            terminal_size,
        )
        if height <= safe_height:
            return self

        return SixelImage(
            self._image_data.pil_image,
            width="auto",
            height=safe_height,
            sixel_options=self._sixel_options,
        )

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        fitted = self._fitted_image(options)
        if fitted is None:
            return
        if fitted is self:
            yield from super().__rich_console__(console, options)
            return
        yield from fitted.__rich_console__(console, options)

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        fitted = self._fitted_image(options)
        if fitted is None:
            return Measurement(0, 0)
        if fitted is self:
            return super().__rich_measure__(console, options)
        return fitted.__rich_measure__(console, options)
