"""Demo commands for UI features."""

from __future__ import annotations

import asyncio
import json
import time
from enum import Enum
from pathlib import Path
from random import Random
from typing import Iterable

import typer

from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.message_primitives import MESSAGE_CONFIGS, MessageType
from fast_agent.ui.streaming import StreamingMessageHandle

app = typer.Typer(help="Demo commands for UI features.")


@app.callback(invoke_without_command=True)
def _demo_root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _chunk_text(text: str, chunk_size: int) -> Iterable[str]:
    for idx in range(0, len(text), chunk_size):
        yield text[idx : idx + chunk_size]


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
    large_code = "large-code"
    many_code = "many-code"
    code_growth = "code-growth"
    small_tables = "small-tables"
    large_tables = "large-tables"
    long_paragraphs = "long-paragraphs"
    interspersed = "interspersed"
    random_mix = "random-mix"


_SCENARIO_ORDER = [
    DemoScenario.large_code,
    DemoScenario.many_code,
    DemoScenario.code_growth,
    DemoScenario.small_tables,
    DemoScenario.large_tables,
    DemoScenario.long_paragraphs,
    DemoScenario.interspersed,
    DemoScenario.random_mix,
]

_SCENARIO_DESCRIPTIONS = {
    DemoScenario.mixed: "A blended workload with lists, tables, code, and paragraphs.",
    DemoScenario.large_code: "One oversized code block to stress markdown height measurement.",
    DemoScenario.many_code: "Many small code blocks to stress fence detection and padding.",
    DemoScenario.code_growth: "Code blocks that grow in size to stress progressive truncation.",
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
    paragraphs = max(6, 4 * scale)
    sentence = (
        "This paragraph is intentionally verbose to test wrapping and scrolling performance."
    )
    content = ["### Long Paragraphs", ""]
    for idx in range(paragraphs):
        content.append(_repeat_sentence(sentence, 320 + idx * 30))
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
    DemoScenario.large_code: _build_large_codeblock,
    DemoScenario.many_code: _build_many_small_codeblocks,
    DemoScenario.code_growth: _build_code_growth,
    DemoScenario.small_tables: _build_small_tables,
    DemoScenario.large_tables: _build_large_table,
    DemoScenario.long_paragraphs: _build_long_paragraphs,
    DemoScenario.interspersed: _build_interspersed,
}


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
    cache_stats: bool = typer.Option(
        False,
        "--cache-stats",
        help="Print markdown cache sizes after streaming.",
    ),
) -> None:
    """Stream a synthetic markdown document without any model calls."""
    if chunk_size <= 0:
        raise typer.BadParameter("chunk-size must be a positive integer.")
    if delay < 0:
        raise typer.BadParameter("delay must be >= 0.")
    if section_pause < 0:
        raise typer.BadParameter("section-pause must be >= 0.")
    if metrics_interval <= 0:
        raise typer.BadParameter("metrics-interval must be a positive integer.")

    scale = max(1, lines // 40)
    if cycle:
        scenario_list = _SCENARIO_ORDER
    else:
        scenario_list = scenarios or [DemoScenario.mixed]

    if seed == -1:
        seed = None

    sections = [
        _build_scenario_markdown(scenario, lines=lines, scale=scale, seed=seed)
        for scenario in scenario_list
    ]
    content = "\n\n---\n\n".join(sections)
    cache_snapshots: list[tuple[str, dict[str, int]]] = []
    metrics_writer = None
    if metrics_path:
        metrics_writer = MetricsWriter(metrics_path, metrics_interval)

    async def _run_stream() -> None:
        display = ConsoleDisplay()
        config = MESSAGE_CONFIGS[MessageType.ASSISTANT]
        block_color = config["block_color"]
        arrow = config["arrow"]
        arrow_style = config["arrow_style"]
        header_left = (
            f"[{block_color}]▎[/{block_color}][{arrow_style}]{arrow}[/{arrow_style}] "
        )
        header_right = "[dim]demo[/dim]"

        handle = StreamingMessageHandle(
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
        try:
            for idx, section in enumerate(sections):
                scenario_name = scenario_list[idx].value
                chunk_index = 0
                if metrics_writer:
                    metrics_writer.set_context(scenario_name)

                for chunk in _chunk_text(section, chunk_size):
                    chunk_index += 1
                    if metrics_writer:
                        metrics_writer.update_chunk_index(chunk_index)
                        metrics_writer.record_chunk(len(chunk))
                    handle.update(chunk)
                    if delay:
                        await asyncio.sleep(delay)
                    else:
                        await asyncio.sleep(0)
                if metrics_writer:
                    metrics_writer.finalize_context()
                if cache_stats:
                    cache_snapshots.append(
                        (scenario_name, handle._markdown_truncator.cache_sizes())
                    )
                if idx < len(sections) - 1:
                    for chunk in _chunk_text("\n\n---\n\n", chunk_size):
                        handle.update(chunk)
                        if delay:
                            await asyncio.sleep(delay)
                        else:
                            await asyncio.sleep(0)
                    if section_pause:
                        await asyncio.sleep(section_pause)
            handle.finalize(content)
        finally:
            handle.close()
            if metrics_writer:
                metrics_writer.close()

    def _run_sync() -> None:
        display = ConsoleDisplay()
        config = MESSAGE_CONFIGS[MessageType.ASSISTANT]
        block_color = config["block_color"]
        arrow = config["arrow"]
        arrow_style = config["arrow_style"]
        header_left = (
            f"[{block_color}]▎[/{block_color}][{arrow_style}]{arrow}[/{arrow_style}] "
        )
        header_right = "[dim]demo[/dim]"

        handle = StreamingMessageHandle(
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
        try:
            for idx, section in enumerate(sections):
                scenario_name = scenario_list[idx].value
                chunk_index = 0
                if metrics_writer:
                    metrics_writer.set_context(scenario_name)

                for chunk in _chunk_text(section, chunk_size):
                    chunk_index += 1
                    if metrics_writer:
                        metrics_writer.update_chunk_index(chunk_index)
                        metrics_writer.record_chunk(len(chunk))
                    handle.update(chunk)
                    if delay:
                        time.sleep(delay)
                if metrics_writer:
                    metrics_writer.finalize_context()
                if cache_stats:
                    cache_snapshots.append(
                        (scenario_name, handle._markdown_truncator.cache_sizes())
                    )
                if idx < len(sections) - 1:
                    for chunk in _chunk_text("\n\n---\n\n", chunk_size):
                        handle.update(chunk)
                        if delay:
                            time.sleep(delay)
                    if section_pause:
                        time.sleep(section_pause)
            handle.finalize(content)
        finally:
            handle.close()
            if metrics_writer:
                metrics_writer.close()

    if async_mode:
        asyncio.run(_run_stream())
    else:
        _run_sync()

    if cache_stats and cache_snapshots:
        typer.echo("cache stats (entries):")
        for scenario_name, stats in cache_snapshots:
            typer.echo(
                f"{scenario_name}: height={stats['height_entries']} "
                f"truncate={stats['truncate_entries']}"
            )
