from __future__ import annotations

import io
import re

from rich.console import Console, Group
from rich.syntax import Syntax

from fast_agent.ui.markdown_renderables import build_markdown_renderable
from fast_agent.ui.markdown_truncator import MarkdownTruncator


def _find_renderable_containing(renderable: object, needle: str) -> object | None:
    """Return the first leaf renderable whose text contains ``needle``."""
    if isinstance(renderable, Group):
        for child in renderable.renderables:
            found = _find_renderable_containing(child, needle)
            if found is not None:
                return found
        return None
    if isinstance(renderable, Syntax):
        return renderable if needle in (renderable.code or "") else None
    # Rich Markdown / Text etc. expose the source via .markup or str()
    source = getattr(renderable, "markup", None) or str(renderable)
    return renderable if needle in source else None


def test_streaming_truncation_reinserts_code_fence() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.5)
    test_console = Console(width=80)

    code_body = "\n".join(f"print({i})" for i in range(40))
    text = "intro\n```python\n" + code_body + "\n```\nsummary line"

    truncated = truncator.truncate(
        text,
        terminal_height=10,
        console=test_console,
        code_theme="native",
        prefer_recent=True,
    )

    assert truncated.startswith("```python\n")
    assert truncated.count("```") >= 2


def test_streaming_truncation_handles_untyped_code_block() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.5)
    test_console = Console(width=80)

    code_body = "\n".join(f"line {i}" for i in range(50))
    text = "preface\n```\n" + code_body + "\n```\npostface"

    truncated = truncator.truncate(
        text,
        terminal_height=12,
        console=test_console,
        code_theme="native",
        prefer_recent=True,
    )

    assert truncated.startswith("```\n")
    assert truncated.count("```") >= 2


def test_measure_rendered_height_matches_padded_code_block_rendering() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.5)
    test_console = Console(file=io.StringIO(), force_terminal=False, width=80)
    text = "```bash\necho hi\n```\n\n```python\nprint(1)\n```"

    measured = truncator.measure_rendered_height(text, test_console, code_theme="native")
    rendered = test_console.render_lines(
        build_markdown_renderable(
            text,
            code_theme="native",
            escape_xml=False,
            close_incomplete_fences=True,
            render_fences_with_syntax=True,
        ),
        options=test_console.options.update(width=test_console.size.width),
        pad=False,
    )

    assert measured == len(rendered)


def test_streaming_truncation_tracks_latest_code_block_language() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.5)
    test_console = Console(width=80)

    second_block = "\n".join(f"print({i})" for i in range(80))
    text = (
        f'header\n```json\n{{ "example": true }}\n```\nmiddle\n```python\n{second_block}\n```\ntail'
    )

    truncated = truncator.truncate(
        text,
        terminal_height=10,
        console=test_console,
        code_theme="native",
        prefer_recent=True,
    )

    assert truncated.startswith("```python\n")
    assert "```json" not in truncated.splitlines()[0]
    assert truncated.count("```python") == 1
    assert truncated.count("```") >= 2


def test_streaming_truncation_consistency_across_sliding_window() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.6)
    test_console = Console(width=80)

    segments = [
        "intro paragraph",  # plain text
        '```json\n{\n  "alpha": 1\n}\n```',  # short code block
        "more context text",  # plain text
        "```python\n" + "\n".join(f"print({i})" for i in range(30)) + "\n```",  # long block
        "closing remarks",  # plain text
    ]
    full_text = "\n".join(segments)

    for height in range(8, 20):
        truncated = truncator.truncate(
            full_text,
            terminal_height=height,
            console=test_console,
            code_theme="native",
            prefer_recent=True,
        )

        assert truncated.strip(), f"no content produced for height={height}"

        trailing_source = full_text[-len(truncated) :]
        json_open = trailing_source.count("```json")
        python_open = trailing_source.count("```python")

        if python_open > 0:
            assert truncated.startswith("```python"), "python fence not preserved"
        elif json_open > 0:
            assert truncated.startswith("```json"), "json fence not preserved"

        if truncated.startswith("```json"):
            assert "```python" not in truncated.splitlines()[0]
        if truncated.startswith("```python"):
            assert "```json" not in truncated.splitlines()[0]

    assert truncated.count("```") >= 2, f"missing closing fence for height={height}"


def test_streaming_truncation_many_small_blocks() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.6)
    test_console = Console(width=80)

    snippets = []
    code_blocks: dict[int, str] = {}
    for idx in range(10):
        snippets.append(f"Paragraph {idx}\n\nThis is some filler text for block {idx}.")
        block = "```lang{}\n{}\n```".format(
            idx,
            "\n".join(f"value_{idx}_{n}" for n in range(3)),
        )
        snippets.append(block)
        code_blocks[idx] = block

    full_text = "\n\n".join(snippets)

    for height in range(6, 18):
        truncated = truncator.truncate(
            full_text,
            terminal_height=height,
            console=test_console,
            code_theme="native",
            prefer_recent=True,
        )

        assert truncated.strip(), f"no content produced for height={height}"

        # Parse the truncated output directly to validate structure
        opening_fence_match = re.match(r"```(lang\d+)?\n", truncated)
        content_matches = list(re.finditer(r"value_(\d+)_(\d+)", truncated))

        if opening_fence_match:
            lang_spec = opening_fence_match.group(1)  # e.g., "lang9" or None

            if lang_spec:
                # Specific language fence - verify content from that block is present
                block_idx = int(lang_spec.replace("lang", ""))
                block_content = [m for m in content_matches if int(m.group(1)) == block_idx]
                assert block_content, (
                    f"fence ```{lang_spec} present but no content from block {block_idx} at height={height}"
                )
            else:
                # Generic fence - just verify some content exists
                assert content_matches, f"generic fence present but no content at height={height}"

            # Verify closing fence exists
            assert truncated.count("```") >= 2, f"missing closing fence for height={height}"
        elif content_matches:
            # Content present but no opening fence
            # Check if there are any fences at all
            fence_count = truncated.count("```")
            if fence_count > 0:
                assert fence_count >= 2, f"unbalanced fences at height={height}"


def test_streaming_truncation_preserves_table_header() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.6)
    test_console = Console(width=80)

    header = "| name | value |"
    separator = "|------|-------|"
    rows = [f"| row{i} | {i} |" for i in range(50)]

    table_text = "\n".join([header, separator, *rows])
    text = "\n\n".join(
        [
            "Intro paragraph explaining the table.",
            table_text,
            "Closing remarks with summary.",
        ]
    )

    for height in range(6, 18):
        truncated = truncator.truncate(
            text,
            terminal_height=height,
            console=test_console,
            code_theme="native",
            prefer_recent=True,
        )

        assert truncated.strip(), f"no content produced for height={height}"

        lines = [line for line in truncated.splitlines() if line.strip()]
        data_indices = [i for i, line in enumerate(lines) if line.startswith("| row")]
        if not data_indices:
            continue

        first_data_index = data_indices[0]
        assert first_data_index >= 2, f"missing header before table rows at height={height}"
        assert lines[first_data_index - 2] == header, (
            f"expected header at height={height}, got {lines[first_data_index - 2]!r}"
        )
        assert lines[first_data_index - 1] == separator, (
            f"expected separator at height={height}, got {lines[first_data_index - 1]!r}"
        )


def test_streaming_truncation_handles_lists_and_code() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.6)
    test_console = Console(width=80)

    sections = []
    for idx in range(12):
        sections.append(f"- item {idx}\n  continuation for item {idx}")
        sections.append("```bash\n" + "\n".join(f"echo item_{idx}_{n}" for n in range(4)) + "\n```")

    text = "\n\n".join(sections)

    for height in range(6, 16):
        truncated = truncator.truncate(
            text,
            terminal_height=height,
            console=test_console,
            code_theme="native",
            prefer_recent=True,
        )

        assert truncated.strip(), f"no content produced for height={height}"

        if "```bash" in truncated:
            fence_index = truncated.rfind("```bash")
            assert fence_index != -1
            nearest_echo = truncated.find("echo item_", fence_index)
            assert nearest_echo != -1, "code block truncated without content"

        bullet_lines = [line for line in truncated.splitlines() if line.startswith("- item")]
        if bullet_lines:
            assert bullet_lines[0].startswith("- item"), "bullet prefix lost after truncation"


def test_streaming_truncation_indented_code_block() -> None:
    """Indented (4-space) code blocks must not gain a synthetic ``` fence.

    Earlier versions prepended an opening fence when truncation landed inside
    an indented code block. Because indented blocks have no closing delimiter
    in the source, the downstream ``close_incomplete_code_blocks`` pass in the
    live render path would then append a closing ``` at the very end of the
    truncated text — sweeping any paragraphs that followed the indented block
    into a spurious fenced code region. The retained 4-space indent on every
    kept line is sufficient for markdown-it to re-detect the block as code,
    so no synthetic fence is needed and trailing prose stays prose.
    """
    truncator = MarkdownTruncator(target_height_ratio=0.6)
    test_console = Console(width=80)

    indented_block = "\n".join(f"    indented line {i}" for i in range(60))
    text = "lead-in paragraph\n\n" + indented_block + "\n\nclosing text"

    for height in range(6, 14):
        truncated = truncator.truncate(
            text,
            terminal_height=height,
            console=test_console,
            code_theme="native",
            prefer_recent=True,
        )

        assert truncated.strip(), f"no content produced for height={height}"
        # No synthetic fence should be inserted for indented blocks.
        assert not truncated.lstrip().startswith("```"), (
            f"unexpected synthetic fence for indented block at height={height}"
        )
        # Any retained indented lines keep their 4-space prefix so markdown-it
        # still recognises them as an indented code block without help.
        retained_indented_lines = [
            line for line in truncated.splitlines() if "indented line" in line
        ]
        for line in retained_indented_lines:
            assert line.startswith("    "), (
                f"indented line lost its 4-space prefix at height={height}: {line!r}"
            )
        # If trailing prose made it into the window, it must not be dragged
        # inside a synthetic fence (i.e. it must render via Markdown, not Syntax).
        if "closing text" in truncated:
            renderable = build_markdown_renderable(
                truncated,
                code_theme="native",
                escape_xml=False,
                close_incomplete_fences=True,
                render_fences_with_syntax=True,
            )
            closing_renderable = _find_renderable_containing(renderable, "closing text")
            assert closing_renderable is not None, (
                f"closing text vanished from renderables at height={height}"
            )
            assert not isinstance(closing_renderable, Syntax), (
                "closing prose was rendered as Syntax (inside a spurious fence) "
                f"at height={height}"
            )


def test_streaming_truncation_does_not_wrap_nested_list_as_code() -> None:
    """Deeply-indented list continuations must not be rendered as code.

    When a viewport truncation lands just before a line indented by 4+ spaces
    (e.g. a nested bullet like ``    - child``), markdown-it parses that line
    in isolation as an indented code block. Earlier versions responded by
    prepending a synthetic ``` fence, which then got auto-closed at end of
    text by ``close_incomplete_code_blocks`` — dragging the following list
    items and any trailing paragraphs into a single syntax-highlighted region.
    The effect was prose suddenly "looking like code" during a live stream and
    self-correcting once the viewport scrolled past the indented line.
    """
    truncator = MarkdownTruncator(target_height_ratio=0.7)
    test_console = Console(width=80)

    text = (
        "Here is the plan for the new model:\n\n"
        "- Item has its own model-db entry/spec, not just an alias.\n"
        "  - It uses the adaptive reasoning spec with:\n"
        "    - low\n"
        "    - medium\n"
        "    - high\n"
        "    - xhigh\n"
        "    - max\n"
        "  - Default effort is medium.\n"
        "  - Router prefers adaptive reasoning by default.\n\n"
        "Next up we should verify the router configuration, confirm that "
        "defaults load, and decide on the alias policy.\n"
    )

    # Sweep a range of heights so at least some truncations land just before
    # a 4-space-indented line (the failure mode). For every height, no prose
    # should end up inside a Syntax renderable.
    prose_markers = (
        "Next up",
        "Router prefers",
        "Default effort",
        "defaults load",
    )

    saw_group = False
    for height in range(6, 18):
        truncated = truncator.truncate_to_height(
            text, terminal_height=height, console=test_console
        )
        renderable = build_markdown_renderable(
            truncated,
            code_theme="native",
            escape_xml=False,
            close_incomplete_fences=True,
            render_fences_with_syntax=True,
        )
        if isinstance(renderable, Group):
            saw_group = True
        # Walk every Syntax leaf and make sure none contain prose.
        pending: list[object] = [renderable]
        while pending:
            node = pending.pop()
            if isinstance(node, Group):
                pending.extend(node.renderables)
                continue
            if isinstance(node, Syntax):
                code = node.code or ""
                for marker in prose_markers:
                    assert marker not in code, (
                        f"prose containing {marker!r} was rendered as Syntax at "
                        f"height={height}; truncated text was:\n{truncated!r}"
                    )

    # Guard: the sweep really did exercise the fenced/grouped paths too, so
    # the loop isn't vacuously passing by only hitting pure-Markdown frames.
    assert saw_group, "expected at least one height to produce a Group renderable"


def test_streaming_truncation_avoids_duplicate_table_header() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.5)
    original = (
        "Intro\n"
        "| Mission | Date |\n"
        "| --- | --- |\n"
        "| Apollo 11 | 1969 |\n"
        "| Apollo 12 | 1969 |\n"
    )

    truncated = (
        "| Mission | Date |\n"
        "| --- | --- |\n"
        "| Apollo 12 | 1969 |\n"
    )

    result = truncator._ensure_table_header_if_needed(original, truncated)
    assert result.count("| Mission | Date |") == 1


def test_streaming_table_scrolls_latest_rows() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.75)
    test_console = Console(width=200)

    header = (
        "| Rank | Airport Name | IATA | ICAO | City/Region | Country | Elevation (m) | "
        "Elevation (ft) |"
    )
    separator = (
        "|------|--------------|------|------|-------------|---------|---------------|"
        "----------------|"
    )
    rows = [
        "| 1 | Daocheng Yading Airport | DCY | ZUDC | Daocheng | China | 4,411 | 14,472 |",
        "| 2 | Qamdo Bamda Airport | BPX | ZUBD | Qamdo | China | 4,334 | 14,219 |",
        "| 3 | Kangding Airport | KGT | ZUKD | Kangding | China | 4,280 | 14,042 |",
        "| 4 | Ngari Gunsa Airport | NGQ | ZUAS | Ngari | China | 4,274 | 14,022 |",
        "| 5 | El Alto International Airport | LPB | SLLP | La Paz | Bolivia | 4,061 | 13,325 |",
        "| 6 | Yushu Batang Airport | YUS | ZLYS | Yushu | China | 3,890 | 12,762 |",
        "| 7 | Inca Manco Capac International Airport | JUL | SPJL | Juliaca | Peru | 3,826 | 12,552 |",
        "| 8 | Shigatse Peace Airport | RKZ | ZURK | Shigatse | China | 3,782 | 12,408 |",
        "| 9 | Lhasa Gonggar Airport | LXA | ZULS | Lhasa | China | 3,570 | 11,710 |",
        "| 10 | Leh Kushok Bakula Rimpochee Airport | IXL | VILH | Leh | India | 3,256 | 10,682 |",
        "| 11 | Alejandro Velasco Astete International Airport | CUZ | SPZO | Cusco | Peru | 3,199 | 10,489 |",
        "| 12 | Tenzing-Hillary Airport | LUA | VNLK | Lukla | Nepal | 2,860 | 9,383 |",
        "| 13 | Alcantari Airport (Sucre) | SRE | SLET | Sucre | Bolivia | 2,834 | 9,301 |",
        "| 14 | Toluca International Airport | TLC | MMTO | Toluca | Mexico | 2,580 | 8,465 |",
        "| 15 | Arequipa Airport | AQP | SPQU | Arequipa | Peru | 2,560 | 8,400 |",
        "| 16 | Jorge Wilstermann International Airport | CBB | SLCB | Cochabamba | Bolivia | 2,548 | 8,360 |",
        "| 17 | El Dorado International Airport | BOG | SKBO | Bogota | Colombia | 2,548 | 8,360 |",
        "| 18 | Mariscal Sucre International Airport | UIO | SEQM | Quito | Ecuador | 2,400 | 7,873 |",
        "| 19 | Addis Ababa Bole International Airport | ADD | HAAB | Addis Ababa | Ethiopia | 2,334 | 7,625 |",
        "| 20 | Mexico City International Airport | MEX | MMMX | Mexico City | Mexico | 2,230 | 7,316 |",
        "| 21 | Puebla International Airport | PBC | MMPB | Puebla | Mexico | 2,204 | 7,230 |",
        "| 22 | Kunming Changshui International Airport | KMG | ZPPP | Kunming | China | 2,103 | 6,896 |",
        "| 23 | Sanaa International Airport | SAH | OYSN | Sanaa | Yemen | 2,200 | 7,218 |",
        "| 24 | Lanzhou Zhongchuan International Airport | LHW | ZLLL | Lanzhou | China | 1,967 | 6,450 |",
        "| 25 | Kabul International Airport | KBL | OAKB | Kabul | Afghanistan | 1,791 | 5,877 |",
        "| 26 | Denver International Airport | DEN | KDEN | Denver | USA | 1,655 | 5,431 |",
        "| 27 | O.R. Tambo International Airport | JNB | FAOR | Johannesburg | South Africa | 1,694 | 5,558 |",
        "| 28 | Tehran Imam Khomeini International Airport | IKA | OIIE | Tehran | Iran | 1,007 | 3,305 |",
        "| 29 | Urumqi Diwopu International Airport | URC | ZWWW | Urumqi | China | 648 | 2,126 |",
        "| 30 | Silao International Airport (Bajio) | BJX | MMLO | Silao | Mexico | 1,815 | 5,955 |",
    ]

    table_text = "\n".join([header, separator, *rows])
    text = "Here is a table of the highest elevation airports worldwide:\n\n" + table_text

    total_rows = len(rows)
    total_lines = total_rows + 2

    def expected_start_row(height: int, ratio: float) -> int:
        target_lines = max(1, int(height * ratio))
        if target_lines >= total_lines:
            return 1
        start_line = total_lines - target_lines + 1
        return max(1, start_line - 2)

    for height in (12, 16, 20):
        truncated = truncator.truncate(
            text,
            terminal_height=height,
            console=test_console,
            code_theme="native",
            prefer_recent=True,
        )

        lines = [line for line in truncated.splitlines() if line.strip()]
        assert header in lines
        assert separator in lines

        row_numbers = []
        for line in lines:
            if line.startswith("|"):
                parts = [part.strip() for part in line.split("|")]
                if len(parts) > 1 and parts[1].isdigit():
                    row_numbers.append(int(parts[1]))

        assert row_numbers, "expected table rows to be present in truncated output"
        start_row = expected_start_row(height, 0.75)
        assert row_numbers[0] == start_row
        assert row_numbers[-1] == total_rows
        assert row_numbers == list(range(start_row, total_rows + 1))
