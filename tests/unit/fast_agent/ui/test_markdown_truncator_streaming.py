from __future__ import annotations

import re

from fast_agent.ui import console
from fast_agent.ui.markdown_truncator import MarkdownTruncator


def test_streaming_truncation_reinserts_code_fence() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.5)

    code_body = "\n".join(f"print({i})" for i in range(40))
    text = "intro\n```python\n" + code_body + "\n```\nsummary line"

    truncated = truncator.truncate(
        text,
        terminal_height=10,
        console=console.console,
        code_theme="native",
        prefer_recent=True,
    )

    assert truncated.startswith("```python\n")
    assert truncated.count("```") >= 2


def test_streaming_truncation_handles_untyped_code_block() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.5)

    code_body = "\n".join(f"line {i}" for i in range(50))
    text = "preface\n```\n" + code_body + "\n```\npostface"

    truncated = truncator.truncate(
        text,
        terminal_height=12,
        console=console.console,
        code_theme="native",
        prefer_recent=True,
    )

    assert truncated.startswith("```\n")
    assert truncated.count("```") >= 2


def test_streaming_truncation_tracks_latest_code_block_language() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.5)

    second_block = "\n".join(f"print({i})" for i in range(80))
    text = (
        f'header\n```json\n{{ "example": true }}\n```\nmiddle\n```python\n{second_block}\n```\ntail'
    )

    truncated = truncator.truncate(
        text,
        terminal_height=10,
        console=console.console,
        code_theme="native",
        prefer_recent=True,
    )

    assert truncated.startswith("```python\n")
    assert "```json" not in truncated.splitlines()[0]
    assert truncated.count("```python") == 1
    assert truncated.count("```") >= 2


def test_streaming_truncation_consistency_across_sliding_window() -> None:
    truncator = MarkdownTruncator(target_height_ratio=0.6)

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
            console=console.console,
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
            console=console.console,
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
