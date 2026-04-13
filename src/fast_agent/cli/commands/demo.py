"""Demo commands for UI features."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

import typer
from typer.models import OptionInfo

from fast_agent.cli.command_support import get_settings_or_exit
from fast_agent.cli.shared_options import CommonAgentOptions
from fast_agent.ui import console as shared_console
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.markdown_renderables import build_markdown_renderable
from fast_agent.ui.message_primitives import MESSAGE_CONFIGS, MessageType
from fast_agent.ui.streaming import StreamingMessageHandle

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable


def _build_demo_stream_handle(
    *,
    plain: bool,
    code_word_wrap: bool,
    render_fences_with_syntax: bool,
    metrics_writer: "MetricsWriter | None",
) -> StreamingMessageHandle:
    display = ConsoleDisplay(
        code_word_wrap=code_word_wrap,
        render_fences_with_syntax=render_fences_with_syntax,
    )
    config = MESSAGE_CONFIGS[MessageType.ASSISTANT]
    block_color = config["block_color"]
    arrow = config["arrow"]
    arrow_style = config["arrow_style"]
    header_left = f"[{block_color}]▎[/{block_color}][{arrow_style}]{arrow}[/{arrow_style}] "
    header_right = "[dim]demo[/dim]"
    return StreamingMessageHandle(
        display=display,
        bottom_items=None,
        highlight_index=None,
        max_item_length=None,
        use_plain_text=plain,
        header_left=header_left,
        header_right=header_right,
        progress_display=None,
        performance_hook=metrics_writer.record if metrics_writer else None,
    )


app = typer.Typer(help="Demo commands for UI features.")

_SOURCE_MARKDOWN_DEMO_DIR = Path(__file__).resolve().parents[4] / "examples" / "markdown"
_PACKAGED_MARKDOWN_DEMO_DIR = (
    files("fast_agent").joinpath("resources").joinpath("examples").joinpath("markdown")
)


@app.callback(invoke_without_command=True)
def _demo_root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _chunk_text(text: str, chunk_size: int) -> Iterable[str]:
    for idx in range(0, len(text), chunk_size):
        yield text[idx : idx + chunk_size]


def _read_markdown_demo_asset(filename: str | Path) -> str:
    asset_path = Path(filename).expanduser()

    if asset_path.is_absolute() and asset_path.is_file():
        return asset_path.read_text(encoding="utf-8")

    cwd_path = (Path.cwd() / asset_path).resolve()
    if cwd_path.is_file():
        return cwd_path.read_text(encoding="utf-8")

    source_path = (_SOURCE_MARKDOWN_DEMO_DIR / asset_path).resolve()
    if source_path.is_file():
        return source_path.read_text(encoding="utf-8")

    packaged_path = _PACKAGED_MARKDOWN_DEMO_DIR.joinpath(asset_path.as_posix())
    if packaged_path.is_file():
        return packaged_path.read_text(encoding="utf-8")

    raise FileNotFoundError(f"Markdown demo asset not found: {filename}")


def _resolve_option_value[T](value: T | OptionInfo, default: T) -> T:
    return default if isinstance(value, OptionInfo) else value


@dataclass(frozen=True, slots=True)
class StreamSection:
    scenario_name: str
    content: str


class MetricsWriter:
    def __init__(self, path_template: str, interval: int) -> None:
        self._template = path_template
        self._interval = max(1, interval)
        self._count = 0
        self._scenario = "unknown"
        self._chunk_index = 0
        self._start_time: float | None = None
        self._total_chunks = 0
        self._total_chars = 0
        self._session = time.strftime("%Y%m%d_%H%M%S")
        self._path: Path | None = None
        self._file = None
        if "{scenario}" not in self._template:
            self._open_file(self._resolve_path(self._scenario))

    def _resolve_path(self, scenario: str) -> Path:
        rendered = self._template.format(scenario=scenario, session=self._session)
        return Path(rendered)

    def _open_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")
        self._path = path

    def set_context(self, scenario: str) -> None:
        self._scenario = scenario
        self._chunk_index = 0
        self._total_chunks = 0
        self._total_chars = 0
        self._start_time = time.monotonic()
        if "{scenario}" in self._template:
            new_path = self._resolve_path(scenario)
            if new_path != self._path:
                self.close()
                self._open_file(new_path)

    def update_chunk_index(self, chunk_index: int) -> None:
        self._chunk_index = chunk_index

    def record_chunk(self, char_count: int) -> None:
        if char_count <= 0:
            return
        self._total_chunks += 1
        self._total_chars += char_count

    def record(self, data: dict[str, object]) -> None:
        self._count += 1
        if self._count % self._interval != 0:
            return
        run_elapsed_ms = (
            (time.monotonic() - self._start_time) * 1000
            if self._start_time is not None
            else None
        )
        payload = {
            "ts": time.time(),
            "render_index": self._count,
            "scenario": self._scenario,
            "chunk_index": self._chunk_index,
            "run_elapsed_ms": run_elapsed_ms,
            **data,
        }
        if self._file is not None:
            self._file.write(json.dumps(payload, ensure_ascii=True) + "\n")
            self._file.flush()

    def finalize_context(self) -> None:
        if self._start_time is None:
            return
        payload = {
            "ts": time.time(),
            "event": "summary",
            "scenario": self._scenario,
            "total_elapsed_ms": (time.monotonic() - self._start_time) * 1000,
            "total_chunks": self._total_chunks,
            "total_chars": self._total_chars,
        }
        if self._file is not None:
            self._file.write(json.dumps(payload, ensure_ascii=True) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None


class DemoScenario(str, Enum):
    mixed = "mixed"
    fence_focus = "fence-focus"
    large_code = "large-code"
    many_code = "many-code"
    code_growth = "code-growth"
    viewport_bounce = "viewport-bounce"
    small_tables = "small-tables"
    large_tables = "large-tables"
    long_paragraphs = "long-paragraphs"
    interspersed = "interspersed"
    random_mix = "random-mix"


_SCENARIO_ORDER = [
    DemoScenario.fence_focus,
    DemoScenario.large_code,
    DemoScenario.many_code,
    DemoScenario.code_growth,
    DemoScenario.viewport_bounce,
    DemoScenario.small_tables,
    DemoScenario.large_tables,
    DemoScenario.long_paragraphs,
    DemoScenario.interspersed,
    DemoScenario.random_mix,
]

_SCENARIO_DESCRIPTIONS = {
    DemoScenario.mixed: "A blended workload with lists, tables, code, and paragraphs.",
    DemoScenario.fence_focus: (
        "Focused fence cases for comparing live streaming updates against the final render."
    ),
    DemoScenario.large_code: "One oversized code block to stress markdown height measurement.",
    DemoScenario.many_code: "Many small code blocks to stress fence detection and padding.",
    DemoScenario.code_growth: "Code blocks that grow in size to stress progressive truncation.",
    DemoScenario.viewport_bounce: (
        "Repeated near-threshold waves intended to trigger truncate/fit oscillations."
    ),
    DemoScenario.small_tables: "Repeated short tables to test header preservation.",
    DemoScenario.large_tables: "A large, wide table to stress wrapping and truncation.",
    DemoScenario.long_paragraphs: "Long paragraphs to test soft wrapping across widths.",
    DemoScenario.interspersed: "Interleaved paragraphs, tables, and code blocks.",
    DemoScenario.random_mix: "Random mix of table sizes, paragraphs, and code blocks.",
}


def _repeat_sentence(sentence: str, min_chars: int) -> str:
    chunks: list[str] = []
    total = 0
    while total < min_chars:
        chunks.append(sentence)
        total += len(sentence) + 1
    return " ".join(chunks)


def _build_mixed(lines: int) -> str:
    lines = max(20, lines)
    content: list[str] = [
        "# Streaming Markdown Demo",
        "",
        "This is a local demo to stress the streaming renderer and viewport sizing.",
        "",
        "## Checklist",
        "- Long paragraphs",
        "- Lists",
        "- Tables",
        "- Code fences",
        "",
        "## Table Sample",
        "| Column | Value |",
        "| --- | --- |",
        "| alpha | 1 |",
        "| beta | 2 |",
        "| gamma | 3 |",
        "",
        "## Code Sample",
        "```python",
        "def add(a, b):",
        "    return a + b",
        "```",
        "",
        "## Repeating Section",
    ]
    filler = [
        "### Notes",
        "The quick brown fox jumps over the lazy dog.",
        "A short paragraph keeps wrapping behavior visible across widths.",
        "",
        "- Point A: repeated for layout consistency.",
        "- Point B: streaming should keep the newest content in view.",
        "- Point C: watch for scrollback pollution.",
        "",
        "Inline `code` and **bold** markers should render correctly.",
        "",
    ]
    while len(content) < lines:
        content.extend(filler)
    return "\n".join(content[:lines]) + "\n"


def _build_large_codeblock(scale: int) -> str:
    code_lines = max(40, 40 * scale)
    content = ["### Large Code Block", "```python"]
    for idx in range(code_lines):
        content.append(
            f"line_{idx:03d} = ({idx} * {idx})  # synthetic workload for scrolling"
        )
    content.extend(["```", ""])
    return "\n".join(content)


def _build_fence_focus(scale: int) -> str:
    growth_lines = max(18, 14 * scale)
    content = [
        "### Fence Focus",
        "",
        "Use this scenario to compare the live stream against the final post-stream render.",
        "Smaller chunks make fence transitions easier to inspect.",
        "",
        "#### Case 1 — prose before and after a fence",
        "Before fence marker: the next block should sit flush like Syntax, not with markdown code-block padding.",
        "",
        "```python",
        "def greet(name: str) -> str:",
        "    return f\"hello, {name}\"",
        "```",
        "",
        "After fence marker: reflow here should stay stable when the block above closes.",
        "",
        "#### Case 2 — adjacent fences with different languages",
        "```json",
        '{"alpha": 1, "beta": [2, 3, 4]}',
        "```",
        "```bash",
        "printf 'adjacent fence check\\n'",
        "echo done",
        "```",
        "",
        "#### Case 3 — longer block for repeated repaint pressure",
        "```python",
    ]
    for idx in range(growth_lines):
        content.append(
            f"sample_{idx:02d} = ('visible marker {idx:02d}', {idx} * {idx})"
        )
    content.extend(
        [
            "```",
            "",
            "Trailing prose marker: this paragraph makes it easy to spot whether the final render matches the streamed layout.",
            "",
            "#### Case 4 — fenced patch preview should keep its custom styling",
            "```apply_patch",
            "*** Begin Patch",
            "*** Update File: demo.txt",
            "@@",
            "-old line",
            "+new line",
            "*** End Patch",
            "```",
            "",
            "#### Case 5 — reference definitions around a fenced block",
            "See [renderer notes][render-docs] before the block; the prose above should still resolve the link.",
            "",
            "```python",
            "config = {\"padding\": True, \"reference_defs\": \"preserved\"}",
            "print(config[\"padding\"])",
            "```",
            "",
            "The block above should keep its visual separation without stray blank chunks before this paragraph.",
            "",
            "[render-docs]: https://example.com/rendering \"Renderer notes\"",
            "",
            "Recommended runs:",
            "- `uv run fast-agent demo streaming --scenario fence-focus --chunk-size 7 --delay 0.03`",
            "- `uv run fast-agent demo streaming --scenario fence-focus --chunk-size 3 --delay 0.05`",
            "- `uv run fast-agent demo streaming --scenario fence-focus --wrap-code`",
            "- `uv run fast-agent demo streaming --scenario fence-focus --plain`",
            "",
        ]
    )
    return "\n".join(content)


def _build_many_small_codeblocks(scale: int) -> str:
    count = max(8, 6 * scale)
    content = ["### Many Small Code Blocks", ""]
    for idx in range(count):
        content.extend(
            [
                "```text",
                f"block {idx + 1}: a short snippet",
                "ok",
                "```",
                "",
            ]
        )
    return "\n".join(content)


def _build_code_growth(scale: int) -> str:
    steps = max(4, 3 * scale)
    content = ["### Code Blocks Growing in Size", ""]
    size = 4
    for idx in range(steps):
        content.append(f"#### Block {idx + 1} ({size} lines)")
        content.append("```python")
        for line in range(size):
            content.append(f"row_{line:02d} = {line} + {idx}")
        content.extend(["```", ""])
        size *= 2
    return "\n".join(content)


def _build_viewport_bounce(scale: int) -> str:
    waves = max(5, scale * 3)
    content = [
        "### Viewport Bounce Stress",
        "",
        "This scenario is tuned to hover around the live viewport limit.",
        "Use a narrow terminal and smaller chunk sizes to make the transitions easier to observe.",
        "",
    ]

    for idx in range(waves):
        content.append(f"#### Wave {idx + 1}")
        content.append(
            _repeat_sentence(
                "A near-threshold paragraph should briefly fit, then spill, then fit again after compaction.",
                220 + (idx % 3) * 40,
            )
        )
        content.append("")
        content.extend(
            [
                "| phase | value | note |",
                "| --- | --- | --- |",
                f"| wave | {idx + 1} | paragraph settles before the code block lands |",
                f"| goal | {idx % 4} | watch for overlay stability and stale indicators |",
                "",
                "```python",
            ]
        )
        line_count = 14 + (idx % 4) * 3
        for line in range(line_count):
            content.append(
                f"window_{idx:02d}_{line:02d} = 'segment boundary pressure test {idx}-{line}'"
            )
        content.extend(
            [
                "```",
                "",
                _repeat_sentence(
                    "The tail should stay readable while earlier material gets compacted out of view.",
                    180 + (idx % 2) * 50,
                ),
                "",
            ]
        )

    return "\n".join(content)


def _build_small_tables(scale: int) -> str:
    count = max(6, 5 * scale)
    content = ["### Small Tables", ""]
    for idx in range(count):
        content.extend(
            [
                f"Table {idx + 1}",
                "| Key | Value |",
                "| --- | --- |",
                f"| alpha | {idx} |",
                f"| beta | {idx + 1} |",
                "",
            ]
        )
    return "\n".join(content)


def _build_large_table(scale: int) -> str:
    rows = max(24, 18 * scale)
    content = ["### Large Table", "| Column A | Column B | Column C |", "| --- | --- | --- |"]
    for idx in range(rows):
        content.append(
            f"| row {idx:02d} | some longer value to wrap {idx} | z{idx * 3} |"
        )
    content.append("")
    return "\n".join(content)


def _build_long_paragraphs(scale: int) -> str:
    paragraphs = max(10, 6 * scale)
    content = ["### Long Paragraphs", ""]
    paragraph_starters = [
        "Harbor reports describe a tug easing a damaged ferry toward a foggy pier while passengers count lighthouse flashes and argue about whether the tide is helping or hurting.",
        "A field notebook from a dry plateau lists juniper shade, broken survey stakes, two rusted drums, and a water truck that always seems to arrive five minutes after the crew gives up waiting.",
        "Kitchen staff preparing for a banquet compare copper pans, late herb deliveries, and the exact moment a sauce turns glossy enough to stop stirring without burning the shallots.",
        "Rail dispatch notes mention a stalled freight outside the tunnel, a replacement crew driving in from the coast, and three stations improvising around a schedule that was already unrealistic.",
        "Museum conservators rotate a cracked astrolabe under cool lamps, debating whether the green residue is harmless age, old polish, or evidence of a repair done in haste decades ago.",
        "Storm chasers on a farm road keep revising the map because every ridge hides the cell for a minute, then reveals a darker wall cloud and another set of power lines humming in the wind.",
        "A robotics lab status board mixes calibration warnings, battery temperatures, handwritten arrows, and one stubborn sensor marked with a red circle because nobody trusts its cheerful numbers.",
        "Divers surfacing near a basalt cliff sort tagged samples into orange crates while a support boat radios changing currents, depth readings, and a reminder that daylight is already getting thin.",
    ]
    paragraph_tails = [
        "Watch the commas, emplacements, and long noun phrases here because they create uneven wrap points that should make line movement easier to spot.",
        "This section intentionally mixes short clauses with longer turns of phrase so a tiny rendering shift is visible instead of disappearing into repeated filler.",
        "If the renderer repaints or duplicates a line, the place names and object words in this paragraph should make the glitch much easier to identify at a glance.",
        "The sentence lengths vary on purpose, and the descriptive details are meant to give each paragraph a distinct silhouette once it soft-wraps in a narrow terminal.",
    ]
    for idx in range(paragraphs):
        starter = paragraph_starters[idx % len(paragraph_starters)]
        tail = paragraph_tails[idx % len(paragraph_tails)]
        paragraph = " ".join(
            [
                f"Paragraph {idx + 1:02d}.",
                starter,
                tail,
                f"Marker set {idx + 1:02d}: amber-{idx % 5}, cobalt-{(idx + 2) % 7}, transit-{100 + idx}.",
            ]
        )
        content.append(_repeat_sentence(paragraph, 320 + idx * 30))
        content.append("")
    return "\n".join(content)


def _build_interspersed(scale: int) -> str:
    content = ["### Interspersed Stress Mix", ""]
    for idx in range(max(3, scale + 1)):
        content.append(
            _repeat_sentence(
                "Interleaved content should keep rendering stable while truncating.",
                240 + idx * 40,
            )
        )
        content.append("")
        content.extend(
            [
                "| Metric | Value |",
                "| --- | --- |",
                f"| step | {idx} |",
                f"| score | {idx * 7} |",
                "",
                "```bash",
                f"echo \"round {idx}\"",
                "sleep 1",
                "```",
                "",
            ]
        )
    return "\n".join(content)


def _build_random_mix(scale: int, seed: int | None) -> str:
    rng = Random(seed)
    block_count = max(4, scale * 3)
    content = ["### Randomized Mix", ""]

    for idx in range(block_count):
        choice = rng.choice(["table", "paragraph", "code"])

        if choice == "table":
            rows = rng.randint(2, 60)
            cols = rng.randint(1, 8)
            headers = [f"Col {c + 1}" for c in range(cols)]
            header_line = "| " + " | ".join(headers) + " |"
            separator_line = "| " + " | ".join(["---"] * cols) + " |"
            content.append(f"Table {idx + 1} ({rows}x{cols})")
            content.append(header_line)
            content.append(separator_line)
            for r in range(rows):
                row_values = [f"{r + 1}-{c + 1}" for c in range(cols)]
                content.append("| " + " | ".join(row_values) + " |")
            content.append("")
            continue

        if choice == "paragraph":
            sentence = (
                "Randomized paragraphs should wrap differently across widths to stress truncation."
            )
            content.append(_repeat_sentence(sentence, rng.randint(140, 520)))
            content.append("")
            continue

        lines = rng.randint(3, 40)
        language = rng.choice(["python", "bash", "text"])
        content.append(f"Code block {idx + 1} ({lines} lines)")
        content.append(f"```{language}")
        for line in range(lines):
            if language == "python":
                content.append(f"result_{line:02d} = {line} * {line}")
            elif language == "bash":
                content.append(f"echo \"step {line + 1}\"")
            else:
                content.append(f"line {line + 1}: lorem ipsum")
        content.extend(["```", ""])

    return "\n".join(content)


_SCENARIO_BUILDERS = {
    DemoScenario.mixed: _build_mixed,
    DemoScenario.fence_focus: _build_fence_focus,
    DemoScenario.large_code: _build_large_codeblock,
    DemoScenario.many_code: _build_many_small_codeblocks,
    DemoScenario.code_growth: _build_code_growth,
    DemoScenario.viewport_bounce: _build_viewport_bounce,
    DemoScenario.small_tables: _build_small_tables,
    DemoScenario.large_tables: _build_large_table,
    DemoScenario.long_paragraphs: _build_long_paragraphs,
    DemoScenario.interspersed: _build_interspersed,
}

_SECTION_SEPARATOR = "\n\n---\n\n"


def _validate_streaming_options(
    *,
    chunk_size: int,
    delay: float,
    section_pause: float,
    metrics_interval: int,
) -> None:
    if chunk_size <= 0:
        raise typer.BadParameter("chunk-size must be a positive integer.")
    if delay < 0:
        raise typer.BadParameter("delay must be >= 0.")
    if section_pause < 0:
        raise typer.BadParameter("section-pause must be >= 0.")
    if metrics_interval <= 0:
        raise typer.BadParameter("metrics-interval must be a positive integer.")


def _resolve_demo_scenarios(
    *,
    scenarios: list[DemoScenario] | None,
    cycle: bool,
) -> list[DemoScenario]:
    return list(_SCENARIO_ORDER) if cycle else list(scenarios or [DemoScenario.mixed])


def _build_stream_sections(
    *,
    scenario_list: list[DemoScenario],
    lines: int,
    scale: int,
    seed: int | None,
) -> tuple[list[StreamSection], str]:
    sections = [
        StreamSection(
            scenario_name=scenario.value,
            content=_build_scenario_markdown(scenario, lines=lines, scale=scale, seed=seed),
        )
        for scenario in scenario_list
    ]
    content = _SECTION_SEPARATOR.join(section.content for section in sections)
    return sections, content


def _record_stream_chunk(
    metrics_writer: MetricsWriter | None,
    *,
    chunk_index: int,
    chunk: str,
) -> None:
    if metrics_writer is None:
        return
    metrics_writer.update_chunk_index(chunk_index)
    metrics_writer.record_chunk(len(chunk))


def _record_cache_snapshot(
    *,
    cache_stats: bool,
    cache_snapshots: list[tuple[str, dict[str, int]]],
    scenario_name: str,
    handle: StreamingMessageHandle,
) -> None:
    if not cache_stats:
        return
    cache_snapshots.append((scenario_name, handle._markdown_truncator.cache_sizes()))


def _close_stream_resources(
    *,
    handle: StreamingMessageHandle,
    metrics_writer: MetricsWriter | None,
) -> None:
    handle.close()
    if metrics_writer is not None:
        metrics_writer.close()


async def _pause_async(delay: float) -> None:
    if delay:
        await asyncio.sleep(delay)
    else:
        await asyncio.sleep(0)


async def _pause_sync(delay: float) -> None:
    if delay:
        time.sleep(delay)


async def _run_stream(
    *,
    sections: list[StreamSection],
    content: str,
    plain: bool,
    code_word_wrap: bool,
    render_fences_with_syntax: bool,
    metrics_writer: MetricsWriter | None,
    cache_stats: bool,
    cache_snapshots: list[tuple[str, dict[str, int]]],
    chunk_size: int,
    delay: float,
    section_pause: float,
    pause: Callable[[float], Awaitable[None]],
) -> None:
    handle = _build_demo_stream_handle(
        plain=plain,
        code_word_wrap=code_word_wrap,
        render_fences_with_syntax=render_fences_with_syntax,
        metrics_writer=metrics_writer,
    )
    try:
        for idx, section in enumerate(sections):
            if metrics_writer:
                metrics_writer.set_context(section.scenario_name)

            for chunk_index, chunk in enumerate(_chunk_text(section.content, chunk_size), start=1):
                _record_stream_chunk(metrics_writer, chunk_index=chunk_index, chunk=chunk)
                handle.update(chunk)
                await pause(delay)

            if metrics_writer:
                metrics_writer.finalize_context()
            _record_cache_snapshot(
                cache_stats=cache_stats,
                cache_snapshots=cache_snapshots,
                scenario_name=section.scenario_name,
                handle=handle,
            )
            if idx < len(sections) - 1:
                for chunk in _chunk_text(_SECTION_SEPARATOR, chunk_size):
                    handle.update(chunk)
                    await pause(delay)
                if section_pause:
                    await pause(section_pause)
        handle.finalize(content)
    finally:
        _close_stream_resources(handle=handle, metrics_writer=metrics_writer)


def _build_scenario_markdown(
    scenario: DemoScenario,
    *,
    lines: int,
    scale: int,
    seed: int | None,
) -> str:
    description = _SCENARIO_DESCRIPTIONS.get(scenario, "")
    header = f"## Scenario: {scenario.value.replace('-', ' ').title()}"
    if scenario == DemoScenario.random_mix:
        body = _build_random_mix(scale, seed)
    else:
        body = _SCENARIO_BUILDERS[scenario](lines if scenario == DemoScenario.mixed else scale)
    section = [header, description, "", body]
    return "\n".join(section).strip() + "\n"


@app.command("markdown")
def markdown(
    config_path: str | None = CommonAgentOptions.config_path(),
    sample_file: str = typer.Option(
        "demo_markdown.md",
        "--sample-file",
        "--sample",
        help="Markdown sample to render. Accepts a local path or a bundled asset name.",
    ),
    theme_file: Path | None = typer.Option(
        None,
        "--theme-file",
        help="Optional Rich theme file override for this preview.",
    ),
    wrap_code: bool = typer.Option(
        True,
        "--wrap-code/--crop-code",
        help="Wrap Syntax-rendered code blocks instead of cropping at the viewport edge.",
    ),
    syntax_fences: bool = typer.Option(
        True,
        "--syntax-fences/--markdown-fences",
        help="Render fenced code blocks with Rich Syntax instead of markdown fence blocks.",
    ),
) -> None:
    """Render the bundled markdown style demo."""
    sample_file = _resolve_option_value(sample_file, "demo_markdown.md")
    theme_file = _resolve_option_value(theme_file, None)
    wrap_code = _resolve_option_value(wrap_code, True)
    syntax_fences = _resolve_option_value(syntax_fences, True)

    settings = get_settings_or_exit(config_path)
    if theme_file is not None:
        resolved_theme = theme_file.expanduser()
        if not resolved_theme.is_absolute():
            resolved_theme = (Path.cwd() / resolved_theme).resolve()
        if not resolved_theme.is_file():
            raise typer.BadParameter(
                f"Theme file not found: {theme_file}",
                param_hint="--theme-file",
            )
        settings.logger.theme_file = str(resolved_theme)

    display = ConsoleDisplay(
        config=settings,
        code_word_wrap=wrap_code,
        render_fences_with_syntax=syntax_fences,
    )
    theme_label = settings.logger.theme_file or "default"
    try:
        demo_markdown = _read_markdown_demo_asset(sample_file)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_hint="--sample-file") from exc

    shared_console.console.print(
        f"[dim]theme: {theme_label} • code theme: {display.code_style}[/dim]\n"
    )
    shared_console.console.print(
        build_markdown_renderable(
            demo_markdown,
            code_theme=display.code_style,
            escape_xml=True,
            render_fences_with_syntax=display.render_fences_with_syntax,
            code_word_wrap=display.code_word_wrap,
        ),
        markup=getattr(settings.logger, "enable_markup", True),
    )


@app.command()
def streaming(
    lines: int = typer.Option(
        120,
        "--lines",
        "-l",
        help="Line budget used to scale scenario sizes.",
    ),
    scenarios: list[DemoScenario] | None = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario(s) to stream; repeatable.",
        show_default="mixed",
    ),
    cycle: bool = typer.Option(
        False,
        "--cycle",
        help="Cycle through all pathological scenarios in a fixed order.",
    ),
    section_pause: float = typer.Option(
        0.2,
        "--section-pause",
        help="Pause (seconds) between scenarios when cycling.",
    ),
    seed: int | None = typer.Option(
        0,
        "--seed",
        help="Seed for random-mix scenarios (use -1 for non-deterministic).",
    ),
    metrics_path: str = typer.Option(
        ".fast-agent/streaming-demo-{scenario}-{session}.jsonl",
        "--metrics-path",
        help=(
            "Path template for render timing samples (use empty string to disable). "
            "Supports {scenario} and {session} placeholders."
        ),
    ),
    metrics_interval: int = typer.Option(
        10,
        "--metrics-interval",
        help="Record every Nth render update to reduce overhead.",
    ),
    async_mode: bool = typer.Option(
        True,
        "--async/--sync",
        help="Use async streaming to mimic live model delivery.",
    ),
    chunk_size: int = typer.Option(
        24, "--chunk-size", "-c", help="Character count per streamed chunk."
    ),
    delay: float = typer.Option(
        0.01, "--delay", "-d", help="Delay (seconds) between streamed chunks."
    ),
    plain: bool = typer.Option(False, "--plain", help="Render using plain text streaming."),
    wrap_code: bool = typer.Option(
        True,
        "--wrap-code/--crop-code",
        help="Wrap Syntax-rendered code blocks instead of cropping at the viewport edge.",
    ),
    syntax_fences: bool = typer.Option(
        True,
        "--syntax-fences/--markdown-fences",
        help="Render markdown code fences with Rich Syntax instead of markdown fence blocks.",
    ),
    cache_stats: bool = typer.Option(
        False,
        "--cache-stats",
        help="Print markdown cache sizes after streaming.",
    ),
) -> None:
    """Stream a synthetic markdown document without any model calls."""
    _validate_streaming_options(
        chunk_size=chunk_size,
        delay=delay,
        section_pause=section_pause,
        metrics_interval=metrics_interval,
    )

    scale = max(1, lines // 40)
    scenario_list = _resolve_demo_scenarios(scenarios=scenarios, cycle=cycle)

    if seed == -1:
        seed = None

    sections, content = _build_stream_sections(
        scenario_list=scenario_list,
        lines=lines,
        scale=scale,
        seed=seed,
    )
    cache_snapshots: list[tuple[str, dict[str, int]]] = []
    metrics_writer = MetricsWriter(metrics_path, metrics_interval) if metrics_path else None

    pause = _pause_async if async_mode else _pause_sync
    asyncio.run(
        _run_stream(
            sections=sections,
            content=content,
            plain=plain,
            code_word_wrap=wrap_code,
            render_fences_with_syntax=syntax_fences,
            metrics_writer=metrics_writer,
            cache_stats=cache_stats,
            cache_snapshots=cache_snapshots,
            chunk_size=chunk_size,
            delay=delay,
            section_pause=section_pause,
            pause=pause,
        )
    )

    if cache_stats and cache_snapshots:
        typer.echo("cache stats (entries):")
        for scenario_name, stats in cache_snapshots:
            typer.echo(
                f"{scenario_name}: height={stats['height_entries']} "
                f"truncate={stats['truncate_entries']}"
            )
