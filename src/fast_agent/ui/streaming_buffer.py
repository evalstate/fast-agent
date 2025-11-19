"""Streaming buffer for markdown content with intelligent truncation.

This module provides a simple, robust streaming buffer that:
1. Accumulates streaming chunks from LLM responses
2. Truncates to fit terminal height (keeps most recent content)
3. Preserves markdown context when truncating:
   - Code blocks: retains opening ```language fence
   - Tables: retains header + separator rows
   - Code blocks: adds closing ``` if unclosed

Design Philosophy
=================
KISS (Keep It Simple, Stupid):
- No binary search (streaming is linear)
- No dual modes (streaming always keeps recent content)
- Parse once per truncation (not per chunk)
- Position-based tracking (clear, testable)
"""

from dataclasses import dataclass
from math import ceil
from typing import Generator, List, Optional

from markdown_it import MarkdownIt
from markdown_it.token import Token


@dataclass
class CodeBlock:
    """Position and metadata for a code block."""

    start_pos: int  # Character position where block starts
    end_pos: int  # Character position where block ends
    language: str  # Language identifier (e.g., "python")


@dataclass
class Table:
    """Position and metadata for a table."""

    start_pos: int  # Character position where table starts
    end_pos: int  # Character position where table ends
    header_lines: List[str]  # Header row + separator (e.g., ["| A | B |", "|---|---|"])


class StreamBuffer:
    """Buffer for streaming markdown content with smart truncation.

    Usage:
        buffer = StreamBuffer()
        for chunk in stream:
            buffer.append(chunk)
            display_text = buffer.get_display_text(terminal_height)
            render(display_text)
    """

    def __init__(self):
        """Initialize the stream buffer."""
        self._chunks: List[str] = []
        self._parser = MarkdownIt().enable("table")

    def append(self, chunk: str) -> None:
        """Add a chunk to the buffer.

        Args:
            chunk: Text chunk from streaming response
        """
        if chunk:
            self._chunks.append(chunk)

    def get_full_text(self) -> str:
        """Get the complete buffered text.

        Returns:
            Full concatenated text from all chunks
        """
        return "".join(self._chunks)

    def get_display_text(
        self,
        terminal_height: int,
        target_ratio: float = 0.7,
        terminal_width: Optional[int] = None,
    ) -> str:
        """Get text for display, truncated to fit terminal.

        This applies intelligent truncation when content exceeds terminal height:
        1. Keeps most recent content (last N lines)
        2. Preserves code block fences if truncated mid-block
        3. Preserves table headers if truncated in table data
        4. Adds closing fence if code block is unclosed

        Args:
            terminal_height: Height of terminal in lines
            target_ratio: Keep this multiple of terminal height (default 1.5)
            terminal_width: Optional terminal width for estimating wrapped lines

        Returns:
            Text ready for display (truncated if needed)
        """
        full_text = self.get_full_text()
        if not full_text:
            return full_text
        return self._truncate_for_display(
            full_text, terminal_height, target_ratio, terminal_width
        )

    def clear(self) -> None:
        """Clear the buffer."""
        self._chunks.clear()

    def _truncate_for_display(
        self,
        text: str,
        terminal_height: int,
        target_ratio: float,
        terminal_width: Optional[int],
    ) -> str:
        """Truncate text to fit display with context preservation.

        Algorithm:
        1. If text fits, return as-is
        2. Otherwise, keep last N lines (where N = terminal_height * target_ratio)
        3. Parse markdown to find code blocks and tables
        4. If we truncated mid-code-block, prepend opening fence
        5. If we truncated mid-table-data, prepend table header
        6. If code block is unclosed, append closing fence

        Args:
            text: Full markdown text
            terminal_height: Terminal height in lines
            target_ratio: Multiplier for target line count

        Returns:
            Truncated text with preserved context
        """
        lines = text.split("\n")

        if target_ratio <= 1:
            extra_lines = 0
        else:
            extra_lines = int(ceil(terminal_height * (target_ratio - 1)))
        raw_target_lines = terminal_height + extra_lines

        # Estimate how many rendered lines the text will occupy
        if terminal_width and terminal_width > 0:
            # Treat each logical line as taking at least one row, expanding based on width
            display_counts = self._estimate_display_counts(lines, terminal_width)
            total_display_lines = sum(display_counts)
        else:
            display_counts = None
            total_display_lines = len(lines)

        # Fast path: no truncation needed if content still fits the viewport
        if total_display_lines <= terminal_height:
            # Still need to check for unclosed code blocks
            return self._add_closing_fence_if_needed(text)

        # Determine how many display lines we want to keep after truncation
        desired_display_lines = min(total_display_lines, raw_target_lines)
        if desired_display_lines > terminal_height:
            window_lines = max(1, terminal_height // 5)  # keep ~20% headroom
            desired_display_lines = max(terminal_height, desired_display_lines - window_lines)
        else:
            desired_display_lines = terminal_height

        # Determine how many logical lines we can keep based on estimated display rows
        if display_counts:
            running_total = 0
            start_index = len(lines) - 1
            for idx in range(len(lines) - 1, -1, -1):
                running_total += display_counts[idx]
                start_index = idx
                if running_total >= desired_display_lines:
                    break
        else:
            start_index = max(len(lines) - desired_display_lines, 0)

        # Compute character position where truncation occurs
        truncation_pos = sum(len(line) + 1 for line in lines[:start_index])
        truncated_text = text[truncation_pos:]

        if terminal_width and terminal_width > 0:
            truncated_text, truncation_pos = self._trim_within_line_if_needed(
                text=text,
                truncated_text=truncated_text,
                truncation_pos=truncation_pos,
                terminal_width=terminal_width,
                max_display_lines=desired_display_lines,
            )

        # Parse markdown structures once
        code_blocks = self._find_code_blocks(text)
        tables = self._find_tables(text)

        # Preserve code block context if needed
        truncated_text = self._preserve_code_block_context(
            text, truncated_text, truncation_pos, code_blocks
        )

        # Preserve table context if needed
        truncated_text = self._preserve_table_context(
            text, truncated_text, truncation_pos, tables
        )

        # Add closing fence if code block is unclosed
        truncated_text = self._add_closing_fence_if_needed(truncated_text)

        return truncated_text

    def _find_code_blocks(self, text: str) -> List[CodeBlock]:
        """Find all code blocks in text using markdown-it parser.

        Args:
            text: Markdown text to analyze

        Returns:
            List of CodeBlock objects with position information
        """
        tokens = self._parser.parse(text)
        lines = text.split("\n")
        blocks = []

        for token in self._flatten_tokens(tokens):
            if token.type in ("fence", "code_block") and token.map:
                start_line = token.map[0]
                end_line = token.map[1]
                start_pos = sum(len(line) + 1 for line in lines[:start_line])
                end_pos = sum(len(line) + 1 for line in lines[:end_line])
                language = getattr(token, "info", "") or ""

                blocks.append(
                    CodeBlock(start_pos=start_pos, end_pos=end_pos, language=language)
                )

        return blocks

    def _find_tables(self, text: str) -> List[Table]:
        """Find all tables in text using markdown-it parser.

        Args:
            text: Markdown text to analyze

        Returns:
            List of Table objects with position and header information
        """
        tokens = self._parser.parse(text)
        lines = text.split("\n")
        tables = []

        for i, token in enumerate(tokens):
            if token.type == "table_open" and token.map:
                # Find tbody within this table to extract header
                tbody_start_line = None

                # Look ahead for tbody
                for j in range(i + 1, len(tokens)):
                    if tokens[j].type == "tbody_open" and tokens[j].map:
                        tbody_start_line = tokens[j].map[0]
                        break
                    elif tokens[j].type == "table_close":
                        break

                if tbody_start_line is not None:
                    table_start_line = token.map[0]
                    table_end_line = token.map[1]

                    # Calculate positions
                    start_pos = sum(len(line) + 1 for line in lines[:table_start_line])
                    end_pos = sum(len(line) + 1 for line in lines[:table_end_line])

                    # Header lines = everything before tbody (header row + separator)
                    header_lines = lines[table_start_line:tbody_start_line]

                    tables.append(
                        Table(start_pos=start_pos, end_pos=end_pos, header_lines=header_lines)
                    )

        return tables

    def _preserve_code_block_context(
        self, original_text: str, truncated_text: str, truncation_pos: int, code_blocks: List[CodeBlock]
    ) -> str:
        """Prepend code block opening fence if truncation removed it.

        When we truncate mid-code-block, we need to preserve the opening fence
        so the remaining code still renders with syntax highlighting.

        Args:
            original_text: Full original text
            truncated_text: Text after truncation
            truncation_pos: Character position where truncation happened
            code_blocks: List of code blocks in original text

        Returns:
            Truncated text with fence prepended if needed
        """
        for block in code_blocks:
            # Check if we truncated within this code block
            if block.start_pos < truncation_pos < block.end_pos:
                # We're inside this block - did we remove the opening fence?
                if truncation_pos > block.start_pos:
                    fence = f"```{block.language}\n"
                    # Avoid duplicates
                    if not truncated_text.startswith(fence):
                        return fence + truncated_text
                # Found the relevant block, no need to check others
                break

        return truncated_text

    def _preserve_table_context(
        self, original_text: str, truncated_text: str, truncation_pos: int, tables: List[Table]
    ) -> str:
        """Prepend table header if truncation removed it.

        When we truncate table data rows, we need to preserve the header
        (header row + separator) so the remaining rows have context.

        Design Point #4: Keep the 3 lines marking beginning of table:
        - Newline before table (if present)
        - Header row (e.g., "| Name | Size |")
        - Separator (e.g., "|------|------|")

        Args:
            original_text: Full original text
            truncated_text: Text after truncation
            truncation_pos: Character position where truncation happened
            tables: List of tables in original text

        Returns:
            Truncated text with header prepended if needed
        """
        for table in tables:
            # Check if we truncated within this table
            if table.start_pos < truncation_pos < table.end_pos:
                # Check if we removed the header (header is at start of table)
                # If truncation happened after the header, we need to restore it
                lines = original_text.split("\n")
                table_start_line = sum(
                    1 for line in original_text[:table.start_pos].split("\n")
                ) - 1

                # Find where the data rows start (after separator)
                # Header lines include header row + separator
                data_start_line = table_start_line + len(table.header_lines)
                data_start_pos = sum(len(line) + 1 for line in lines[:data_start_line])

                # If we truncated in the data section, restore header
                if truncation_pos >= data_start_pos:
                    header_text = "\n".join(table.header_lines) + "\n"
                    return header_text + truncated_text

                # Found the relevant table, no need to check others
                break

        return truncated_text

    def _add_closing_fence_if_needed(self, text: str) -> str:
        """Add closing ``` fence if code block is unclosed.

        Design Point #5: Add closing fence to bottom if we detect unclosed block.
        This ensures partial code blocks render correctly during streaming.

        Args:
            text: Markdown text to check

        Returns:
            Text with closing fence added if needed
        """
        # Count opening vs closing fences
        import re

        opening_fences = len(re.findall(r"^```", text, re.MULTILINE))
        closing_fences = len(re.findall(r"^```\s*$", text, re.MULTILINE))

        # If odd number of fences, we have an unclosed block
        if opening_fences > closing_fences:
            # Check if text already ends with a closing fence
            if not re.search(r"```\s*$", text):
                return text + "\n```\n"

        return text

    def _flatten_tokens(self, tokens: List[Token]) -> Generator[Token, None, None]:
        """Flatten nested token tree.

        Args:
            tokens: List of tokens from markdown-it

        Yields:
            Flattened tokens
        """
        for token in tokens:
            is_fence = token.type == "fence"
            is_image = token.tag == "img"
            if token.children and not (is_image or is_fence):
                yield from self._flatten_tokens(token.children)
            else:
                yield token

    def _estimate_display_counts(self, lines: List[str], terminal_width: int) -> List[int]:
        """Estimate how many terminal rows each logical line will occupy."""
        return [
            max(1, ceil(len(line) / terminal_width)) if line else 1
            for line in lines
        ]

    def _estimate_display_lines(self, text: str, terminal_width: int) -> int:
        """Estimate how many terminal rows the given text will occupy."""
        if not text:
            return 0
        lines = text.split("\n")
        return sum(self._estimate_display_counts(lines, terminal_width))

    def _trim_within_line_if_needed(
        self,
        text: str,
        truncated_text: str,
        truncation_pos: int,
        terminal_width: int,
        max_display_lines: int,
    ) -> tuple[str, int]:
        """Trim additional characters when a single line exceeds the viewport."""
        current_pos = truncation_pos
        current_text = truncated_text
        estimated_lines = self._estimate_display_lines(current_text, terminal_width)

        while estimated_lines > max_display_lines and current_pos < len(text):
            excess_display = estimated_lines - max_display_lines
            chars_to_trim = excess_display * terminal_width
            if chars_to_trim <= 0:
                break

            candidate_pos = min(len(text), current_pos + chars_to_trim)

            # Prefer trimming at the next newline to keep markdown structures intact
            newline_pos = text.find("\n", current_pos, candidate_pos)
            if newline_pos != -1:
                candidate_pos = newline_pos + 1

            if candidate_pos <= current_pos:
                break

            current_pos = candidate_pos
            current_text = text[current_pos:]
            estimated_lines = self._estimate_display_lines(current_text, terminal_width)

        return current_text, current_pos
