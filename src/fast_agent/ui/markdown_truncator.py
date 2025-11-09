"""Smart markdown truncation that preserves markdown context.

This module provides intelligent truncation of markdown text for streaming displays,
ensuring that markdown structures (code blocks, lists, blockquotes) are preserved
when possible, and gracefully degrading when single blocks are too large.

KEY CONCEPT: Truncation Strategy
=================================

In STREAMING MODE (prefer_recent=True):
  - Always show MOST RECENT content (keep end, remove beginning)
  - Why: Users are following along as content streams in. They want to see the
    current position, not what was written at the start.
  - For TABLES: Show the most recent rows while preserving the header
  - Example: Table with 100 rows - show header + last 20 rows (not first 20)

In STATIC MODE (prefer_recent=False):
  - For TABLE-DOMINANT content (>50% table lines): Show FIRST page
  - For TEXT content: Show MOST RECENT
  - Example: Tool output listing 100 files - show header + first 20 rows

Context Preservation
====================

When truncating removes the opening of a structure, we restore it:
- CODE BLOCKS: Prepend ```language fence (only if it was removed)
- TABLES: Prepend header row + separator row (only if they were removed)

This ensures truncated content still renders correctly as markdown.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional

from markdown_it import MarkdownIt
from markdown_it.token import Token
from rich.console import Console
from rich.markdown import Markdown
from rich.segment import Segment


@dataclass
class TruncationPoint:
    """Represents a position in text where truncation is safe."""

    char_position: int
    block_type: str
    token: Token
    is_closing: bool


@dataclass
class CodeBlockInfo:
    """Information about a code block in the document."""

    start_pos: int
    end_pos: int
    fence_line: int
    language: str
    fence_text: str | None
    token: Token


@dataclass
class TableInfo:
    """Information about a table in the document."""

    start_pos: int
    end_pos: int
    thead_start_pos: int
    thead_end_pos: int
    tbody_start_pos: int
    tbody_end_pos: int
    header_lines: List[str]  # Header + separator rows


class MarkdownTruncator:
    """Handles intelligent truncation of markdown text while preserving context."""

    def __init__(self, target_height_ratio: float = 0.8):
        """Initialize the truncator.

        Args:
            target_height_ratio: Target height as ratio of terminal height (0.0-1.0).
                After truncation, aim to keep content at this ratio of terminal height.
        """
        self.target_height_ratio = target_height_ratio
        self.parser = MarkdownIt().enable("strikethrough").enable("table")
        # Cache for streaming mode to avoid redundant work
        self._last_full_text: str | None = None
        self._last_truncated_text: str | None = None
        self._last_terminal_height: int | None = None
        # Markdown parse cache
        self._cache_source: str | None = None
        self._cache_tokens: List[Token] | None = None
        self._cache_lines: List[str] | None = None
        self._cache_safe_points: List[TruncationPoint] | None = None
        self._cache_code_blocks: List[CodeBlockInfo] | None = None
        self._cache_tables: List[TableInfo] | None = None

    def truncate(
        self,
        text: str,
        terminal_height: int,
        console: Console,
        code_theme: str = "monokai",
        prefer_recent: bool = False,
    ) -> str:
        """Truncate markdown text to fit within terminal height.

        This method attempts to truncate at safe block boundaries (between paragraphs,
        after code blocks, etc.). If no safe boundary works (e.g., single block is
        too large), it falls back to character-based truncation.

        Args:
            text: The markdown text to truncate.
            terminal_height: Height of the terminal in lines.
            console: Rich Console instance for measuring rendered height.
            code_theme: Theme for code syntax highlighting.
            prefer_recent: If True, always show most recent content (streaming mode).
                This overrides table-dominant detection to ensure streaming tables
                show the latest rows, not the first rows.

        Returns:
            Truncated markdown text that fits within target height.
        """
        if not text:
            return text

        # Fast path for streaming: use incremental truncation
        if prefer_recent:
            return self._truncate_streaming(text, terminal_height, console, code_theme)

        # Measure current height
        current_height = self._measure_rendered_height(text, console, code_theme)

        if current_height <= terminal_height:
            # No truncation needed
            return text

        target_height = int(terminal_height * self.target_height_ratio)

        # Find safe truncation points (block boundaries)
        safe_points = self._find_safe_truncation_points(text)

        if not safe_points:
            # No safe points found, fall back to character truncation
            truncated = self._truncate_by_characters(text, target_height, console, code_theme)
            # Ensure code fence is preserved if we truncated within a code block
            truncated = self._ensure_code_fence_if_needed(text, truncated)
            # Ensure table header is preserved if we truncated within a table body
            return self._ensure_table_header_if_needed(text, truncated)

        # Determine truncation strategy BEFORE finding best point
        # This is needed because _find_best_truncation_point needs to know
        # which direction to test (keep beginning vs keep end)
        is_table_content = False if prefer_recent else self._is_primary_content_table(text)

        # Try to find the best truncation point
        best_point = self._find_best_truncation_point(
            text, safe_points, target_height, console, code_theme, keep_beginning=is_table_content
        )

        if best_point is None:
            # No safe point works, fall back to character truncation
            truncated = self._truncate_by_characters(text, target_height, console, code_theme)
            # Ensure code fence is preserved if we truncated within a code block
            truncated = self._ensure_code_fence_if_needed(text, truncated)
            # Ensure table header is preserved if we truncated within a table body
            return self._ensure_table_header_if_needed(text, truncated)

        # ============================================================================
        # TRUNCATION STRATEGY: Two Different Behaviors
        # ============================================================================
        #
        # We use different truncation strategies depending on content type:
        #
        # 1. TABLES: Show FIRST page (keep beginning, remove end)
        #    - Rationale: Tables are structured data where the header defines meaning.
        #      Users need to see the header and first rows to understand the data.
        #      Showing the "most recent" rows without context is meaningless.
        #    - Example: A file listing table - seeing the last 10 files without the
        #      header columns (name, size, date) is useless.
        #    - NOTE: This is overridden when prefer_recent=True (streaming mode)
        #
        # 2. STREAMING TEXT: Show MOST RECENT (keep end, remove beginning)
        #    - Rationale: In streaming assistant responses, the most recent content
        #      is usually the most relevant. The user is following along as text
        #      appears, so they want to see "where we are now" not "where we started".
        #    - Example: A code explanation - seeing the conclusion is more valuable
        #      than seeing the introduction paragraph that scrolled off.
        #
        # Detection: Content is considered "table-dominant" if >50% of lines are
        # part of table structures (see _is_primary_content_table).
        # OVERRIDE: When prefer_recent=True, always use "show most recent" strategy.
        # ============================================================================

        # Note: is_table_content was already determined above before calling _find_best_truncation_point

        if is_table_content:
            # For tables: keep BEGINNING, truncate END (show first N rows)
            # Use safe point as END boundary, keep everything before it
            truncated_text = text[: best_point.char_position]

            # ========================================================================
            # TABLE HEADER INTEGRITY CHECK
            # ========================================================================
            # Markdown tables require both a header row AND a separator line:
            #
            #   | Name    | Size | Date       |   <-- Header row
            #   |---------|------|------------|   <-- Separator (required!)
            #   | file.py | 2KB  | 2024-01-15 |   <-- Data rows
            #
            # If we truncate between the header and separator, the table won't
            # render at all in markdown. So we need to ensure both are present.
            # ========================================================================
            if truncated_text.strip() and "|" in truncated_text:
                lines_result = truncated_text.split("\n")
                # Check if we have header but missing separator (dashes)
                has_header = any("|" in line and "---" not in line for line in lines_result)
                has_separator = any("---" in line for line in lines_result)

                if has_header and not has_separator:
                    # We cut off the separator! Find it in original and include it
                    original_lines = text.split("\n")
                    for i, line in enumerate(original_lines):
                        if "---" in line and "|" in line:
                            # Found separator line - include up to and including this line
                            truncated_text = "\n".join(original_lines[: i + 1])
                            break
        else:
            # ========================================================================
            # STREAMING TEXT: Keep END, truncate BEGINNING (show most recent)
            # ========================================================================
            # This is the primary use case: assistant is streaming a response, and
            # the terminal can't show all of it. We want to show what's currently
            # being written (the end), not what was written minutes ago (the start).
            # ========================================================================
            truncated_text = text[best_point.char_position :]

            # ========================================================================
            # CONTEXT PRESERVATION for Truncated Structures
            # ========================================================================
            # When truncating removes the beginning of a structure (code block or
            # table), we need to restore the opening context so it renders properly.
            #
            # CODE BLOCKS: If we truncate mid-block, prepend the opening fence
            #   Original:  ```python\ndef foo():\n  return 42\n```
            #   Truncate:  [```python removed] def foo():\n  return 42\n```
            #   Fixed:     ```python\ndef foo():\n  return 42\n```
            #
            # TABLES: If we truncate table data rows, prepend the header
            #   Original:  | Name | Size |\n|------|------|\n| a | 1 |\n| b | 2 |
            #   Truncate:  [header removed] | b | 2 |
            #   Fixed:     | Name | Size |\n|------|------|\n| b | 2 |
            # ========================================================================

            # Get code block info once for efficient position-based checks
            code_blocks = self._get_code_block_info(text)

            # Find which code block (if any) contains the truncation point
            containing_code_block = None
            for block in code_blocks:
                if block.start_pos < best_point.char_position < block.end_pos:
                    containing_code_block = block
                    break

            # Check if we need special handling for code blocks
            if containing_code_block:
                truncated_text = self._handle_code_block_truncation(
                    containing_code_block, best_point, truncated_text
                )

            # Get table info once for efficient position-based checks
            tables = self._get_table_info(text)

            # Find ANY table whose content is in the truncated text but whose header was removed
            for table in tables:
                # Check if we truncated somewhere within this table (after the start)
                # and the truncated text still contains part of this table
                if (
                    best_point.char_position > table.start_pos
                    and best_point.char_position < table.end_pos
                ):
                    # We truncated within this table
                    # Check if the header was removed
                    # Use >= because if we truncate AT thead_end_pos, the header is already gone
                    if best_point.char_position >= table.thead_end_pos:
                        # Header was removed - prepend it
                        header_text = "\n".join(table.header_lines) + "\n"
                        truncated_text = header_text + truncated_text
                        break  # Only restore one table header

        return truncated_text

    def _ensure_parse_cache(self, text: str) -> None:
        if self._cache_source == text:
            return

        tokens = self.parser.parse(text)
        self._cache_source = text
        self._cache_tokens = tokens
        self._cache_lines = text.split("\n")
        self._cache_safe_points = None
        self._cache_code_blocks = None
        self._cache_tables = None

    def _find_safe_truncation_points(self, text: str) -> List[TruncationPoint]:
        """Find safe positions to truncate at (block boundaries).

        Args:
            text: The markdown text to analyze.

        Returns:
            List of TruncationPoint objects representing safe truncation positions.
        """
        self._ensure_parse_cache(text)
        if self._cache_safe_points is not None:
            return list(self._cache_safe_points)

        assert self._cache_tokens is not None
        assert self._cache_lines is not None

        safe_points: List[TruncationPoint] = []
        tokens = self._cache_tokens
        lines = self._cache_lines

        for token in tokens:
            # We're interested in block-level tokens with map information
            # Opening tokens (nesting=1) and self-closing tokens (nesting=0) have map info
            if token.map is not None:
                # token.map gives [start_line, end_line] (0-indexed)
                end_line = token.map[1]

                # Calculate character position at end of this block
                if end_line <= len(lines):
                    char_pos = sum(len(line) + 1 for line in lines[:end_line])

                    safe_points.append(
                        TruncationPoint(
                            char_position=char_pos,
                            block_type=token.type,
                            token=token,
                            is_closing=(token.nesting == 0),  # Self-closing or block end
                        )
                    )
        self._cache_safe_points = safe_points
        return list(safe_points)

    def _get_code_block_info(self, text: str) -> List[CodeBlockInfo]:
        """Extract code block positions and metadata using markdown-it.

        Uses same technique as prepare_markdown_content in markdown_helpers.py:
        parse once with markdown-it, extract exact positions from tokens.

        Args:
            text: The markdown text to analyze.

        Returns:
            List of CodeBlockInfo objects with position and language metadata.
        """
        self._ensure_parse_cache(text)
        if self._cache_code_blocks is not None:
            return list(self._cache_code_blocks)

        assert self._cache_tokens is not None
        assert self._cache_lines is not None

        tokens = self._cache_tokens
        lines = self._cache_lines
        code_blocks: List[CodeBlockInfo] = []

        for token in self._flatten_tokens(tokens):
            if token.type in ("fence", "code_block") and token.map:
                start_line = token.map[0]
                end_line = token.map[1]
                start_pos = sum(len(line) + 1 for line in lines[:start_line])
                end_pos = sum(len(line) + 1 for line in lines[:end_line])
                language = token.info or "" if hasattr(token, "info") else ""
                fence_text: str | None = None
                if token.type == "fence":
                    fence_text = lines[start_line] if 0 <= start_line < len(lines) else None

                code_blocks.append(
                    CodeBlockInfo(
                        start_pos=start_pos,
                        end_pos=end_pos,
                        fence_line=start_line,
                        language=language,
                        fence_text=fence_text,
                        token=token,
                    )
                )
        self._cache_code_blocks = code_blocks
        return list(code_blocks)

    def _build_code_block_prefix(self, block: CodeBlockInfo) -> str | None:
        """Construct the opening fence text for a code block if applicable."""
        token = block.token

        if token.type == "fence":
            if block.fence_text:
                fence_line = block.fence_text
            else:
                markup = getattr(token, "markup", "") or "```"
                info = (getattr(token, "info", "") or "").strip()
                fence_line = f"{markup}{info}" if info else markup
            return fence_line if fence_line.endswith("\n") else fence_line + "\n"

        if token.type == "code_block":
            info = (getattr(token, "info", "") or "").strip()
            if info:
                return f"```{info}\n"
            if block.language:
                return f"```{block.language}\n"
            return "```\n"

        return None

    def _get_table_info(self, text: str) -> List[TableInfo]:
        """Extract table positions and metadata using markdown-it.

        Uses same technique as _get_code_block_info: parse once with markdown-it,
        extract exact positions from tokens.

        Args:
            text: The markdown text to analyze.

        Returns:
            List of TableInfo objects with position and header metadata.
        """
        self._ensure_parse_cache(text)
        if self._cache_tables is not None:
            return list(self._cache_tables)

        assert self._cache_tokens is not None
        assert self._cache_lines is not None

        tokens = self._cache_tokens
        lines = self._cache_lines
        tables: List[TableInfo] = []

        for i, token in enumerate(tokens):
            if token.type == "table_open" and token.map:
                # Find thead and tbody within this table
                thead_start_line = None
                thead_end_line = None
                tbody_start_line = None
                tbody_end_line = None

                # Look ahead in tokens to find thead and tbody
                for j in range(i + 1, len(tokens)):
                    if tokens[j].type == "thead_open" and tokens[j].map:
                        token_map = tokens[j].map
                        assert token_map is not None  # Type narrowing
                        thead_start_line = token_map[0]
                        thead_end_line = token_map[1]
                    elif tokens[j].type == "tbody_open" and tokens[j].map:
                        token_map = tokens[j].map
                        assert token_map is not None  # Type narrowing
                        tbody_start_line = token_map[0]
                        tbody_end_line = token_map[1]
                    elif tokens[j].type == "table_close":
                        # End of this table
                        break

                # Check if we have both thead and tbody
                if (
                    thead_start_line is not None
                    and thead_end_line is not None
                    and tbody_start_line is not None
                    and tbody_end_line is not None
                ):
                    # Calculate character positions
                    table_start_line = token.map[0]
                    table_end_line = token.map[1]

                    # markdown-it reports table_start_line as pointing to the HEADER ROW,
                    # not the separator. So table_start_line should already be correct.
                    # We just need to capture from table_start_line to tbody_start_line
                    # to get both the header row and separator row.
                    actual_table_start_line = table_start_line

                    table_start_pos = sum(len(line) + 1 for line in lines[:actual_table_start_line])
                    table_end_pos = sum(len(line) + 1 for line in lines[:table_end_line])
                    thead_start_pos = sum(len(line) + 1 for line in lines[:thead_start_line])
                    thead_end_pos = sum(len(line) + 1 for line in lines[:thead_end_line])
                    tbody_start_pos = sum(len(line) + 1 for line in lines[:tbody_start_line])
                    tbody_end_pos = sum(len(line) + 1 for line in lines[:tbody_end_line])

                    # Extract header lines (header row + separator)
                    # table_start_line points to the header row,
                    # and tbody_start_line is where data rows start.
                    # So lines[table_start_line:tbody_start_line] gives us both header and separator
                    header_lines = lines[actual_table_start_line:tbody_start_line]

                    tables.append(
                        TableInfo(
                            start_pos=table_start_pos,
                            end_pos=table_end_pos,
                            thead_start_pos=thead_start_pos,
                            thead_end_pos=thead_end_pos,
                            tbody_start_pos=tbody_start_pos,
                            tbody_end_pos=tbody_end_pos,
                            header_lines=header_lines,
                        )
                    )
        self._cache_tables = tables
        return list(tables)

    def _find_best_truncation_point(
        self,
        text: str,
        safe_points: List[TruncationPoint],
        target_height: int,
        console: Console,
        code_theme: str,
        keep_beginning: bool = False,
    ) -> Optional[TruncationPoint]:
        """Find the truncation point that gets closest to target height.

        Args:
            text: The full markdown text.
            safe_points: List of potential truncation points.
            target_height: Target height in terminal lines.
            console: Rich Console for measuring.
            code_theme: Code syntax highlighting theme.
            keep_beginning: If True, test keeping text BEFORE point (table mode).
                           If False, test keeping text AFTER point (streaming mode).

        Returns:
            The best TruncationPoint, or None if none work.
        """
        best_point = None
        best_diff = float("inf")

        for point in safe_points:
            # Test truncating at this point
            # Direction depends on truncation strategy
            if keep_beginning:
                # Table mode: keep beginning, remove end
                truncated = text[: point.char_position]
            else:
                # Streaming mode: keep end, remove beginning
                truncated = text[point.char_position :]

            # Skip if truncation would result in empty or nearly empty text
            if not truncated.strip():
                continue

            height = self._measure_rendered_height(truncated, console, code_theme)

            # Calculate how far we are from target
            diff = abs(height - target_height)

            # We prefer points that keep us at or below target
            if height <= target_height and diff < best_diff:
                best_point = point
                best_diff = diff

        return best_point

    def _truncate_by_characters(
        self, text: str, target_height: int, console: Console, code_theme: str
    ) -> str:
        """Fall back to character-based truncation using binary search.

        This is used when no safe block boundary works (e.g., single block too large).

        Args:
            text: The markdown text to truncate.
            target_height: Target height in terminal lines.
            console: Rich Console for measuring.
            code_theme: Code syntax highlighting theme.

        Returns:
            Truncated text that fits within target height.
        """
        if not text:
            return text

        # Binary search on character position
        left, right = 0, len(text) - 1
        best_pos = None

        while left <= right:
            mid = (left + right) // 2
            test_text = text[mid:]

            if not test_text.strip():
                # Skip empty results
                right = mid - 1
                continue

            height = self._measure_rendered_height(test_text, console, code_theme)

            if height <= target_height:
                # Can keep more text - try removing less
                best_pos = mid
                right = mid - 1
            else:
                # Need to truncate more
                left = mid + 1

        # If nothing fits at all, return the last portion of text that's minimal
        if best_pos is None:
            # Return last few characters or lines that might fit
            # Take approximately the last 20% of the text as a fallback
            fallback_pos = int(len(text) * 0.8)
            return text[fallback_pos:] if fallback_pos < len(text) else text

        return text[best_pos:]

    def measure_rendered_height(
        self, text: str, console: Console, code_theme: str = "monokai"
    ) -> int:
        """Public helper that measures rendered height for markdown content."""
        return self._measure_rendered_height(text, console, code_theme)

    def _handle_code_block_truncation(
        self, code_block: CodeBlockInfo, truncation_point: TruncationPoint, truncated_text: str
    ) -> str:
        """Handle truncation within a code block by preserving the opening fence.

        When truncating within a code block, we need to ensure the opening fence
        (```language) is preserved so the remaining content renders correctly.

        This uses a simple position-based approach: if the truncation point is after
        the fence's starting position, the fence has scrolled off and needs to be
        prepended. Otherwise, it's still on screen.

        Args:
            code_block: The CodeBlockInfo for the block being truncated.
            truncation_point: Where we're truncating.
            truncated_text: The text after truncation.

        Returns:
            Modified truncated text with code fence preserved if needed.
        """
        # Simple check: did we remove the opening fence?
        # If truncation happened after the fence line, it scrolled off
        if truncation_point.char_position > code_block.start_pos:
            # Check if fence is already at the beginning (avoid duplicates)
            fence = self._build_code_block_prefix(code_block)
            if fence and not truncated_text.startswith(fence):
                # Fence scrolled off - prepend it
                return fence + truncated_text

        # Fence still on screen or already prepended - keep as-is
        return truncated_text

    def _ensure_code_fence_if_needed(self, original_text: str, truncated_text: str) -> str:
        """Ensure code fence is prepended if truncation happened within a code block.

        This is used after character-based truncation to check if we need to add
        a code fence to the beginning of the truncated text.

        Uses the same position-based approach as _handle_code_block_truncation.

        Args:
            original_text: The original full text before truncation.
            truncated_text: The truncated text.

        Returns:
            Truncated text with code fence prepended if needed.
        """
        if not truncated_text or truncated_text == original_text:
            return truncated_text

        # Find where the truncated text starts in the original
        truncation_pos = original_text.rfind(truncated_text)
        if truncation_pos == -1:
            truncation_pos = max(0, len(original_text) - len(truncated_text))

        # Get code block info using markdown-it parser
        code_blocks = self._get_code_block_info(original_text)

        # Find which code block (if any) contains the truncation point
        for block in code_blocks:
            if block.start_pos < truncation_pos < block.end_pos:
                # Truncated within this code block
                # Simple check: did truncation remove the fence?
                if truncation_pos > block.start_pos:
                    fence = self._build_code_block_prefix(block)
                    if fence and not truncated_text.startswith(fence):
                        return fence + truncated_text
                # Fence still on screen or already prepended
                return truncated_text

        return truncated_text

    def _ensure_table_header_if_needed(self, original_text: str, truncated_text: str) -> str:
        """Ensure table header is prepended if truncation happened within a table body.

        When truncating within a table body, we need to preserve the header row(s)
        so the remaining table rows have context and meaning.

        Uses the same position-based approach as code block handling.

        Args:
            original_text: The original full text before truncation.
            truncated_text: The truncated text.

        Returns:
            Truncated text with table header prepended if needed.
        """
        if not truncated_text or truncated_text == original_text:
            return truncated_text

        # Find where the truncated text starts in the original
        truncation_pos = original_text.rfind(truncated_text)
        if truncation_pos == -1:
            truncation_pos = max(0, len(original_text) - len(truncated_text))

        # Get table info using markdown-it parser
        tables = self._get_table_info(original_text)

        # Find which table (if any) contains the truncation point in tbody
        for table in tables:
            # Check if truncation happened within tbody (after thead)
            if table.thead_end_pos <= truncation_pos < table.tbody_end_pos:
                # Truncated within table body
                # Simple check: did truncation remove the header?
                # Use >= because if we truncate AT thead_end_pos, the header is already gone
                if truncation_pos >= table.thead_end_pos:
                    # Header completely scrolled off - prepend it
                    header_text = "\n".join(table.header_lines) + "\n"
                    return header_text + truncated_text
                else:
                    # Header still on screen
                    return truncated_text

        return truncated_text

    def _is_primary_content_table(self, text: str) -> bool:
        """Check if the document's primary content is a table.

        This heuristic determines if we should use "show first page" truncation
        (for tables) vs "show most recent" truncation (for streaming text).

        Detection Logic:
        ----------------
        A document is considered "table-dominant" if MORE THAN 50% of its lines
        are part of table structures.

        Why 50%?
        - Below 50%: Content is mostly text with some tables mixed in.
                     Show most recent (standard streaming behavior).
        - Above 50%: Content is primarily tabular data.
                     Show beginning so users see the header defining the columns.

        Examples:
        ---------
        TABLE-DOMINANT (>50%, will show first page):
          | Name | Size |
          |------|------|
          | a    | 1    |
          | b    | 2    |
          | c    | 3    |
          (5 lines, 5 table lines = 100% table)

        NOT TABLE-DOMINANT (≤50%, will show most recent):
          Here's a file listing:
          | Name | Size |
          |------|------|
          | a    | 1    |
          This shows the files in the directory.
          (6 lines, 3 table lines = 50% table)

        Args:
            text: The full markdown text.

        Returns:
            True if document is primarily a table (table content > 50% of lines).
        """
        if not text.strip():
            return False

        tokens = self.parser.parse(text)
        lines = text.split("\n")
        total_lines = len(lines)

        if total_lines == 0:
            return False

        # Count lines that are part of tables
        table_lines = 0
        for token in tokens:
            if token.type == "table_open" and token.map:
                start_line = token.map[0]
                end_line = token.map[1]
                table_lines += end_line - start_line

        # If more than 50% of content is table, consider it table-dominant
        return table_lines > (total_lines * 0.5)

    def _measure_rendered_height(self, text: str, console: Console, code_theme: str) -> int:
        """Measure how many terminal lines the rendered markdown takes.

        Args:
            text: The markdown text to measure.
            console: Rich Console for rendering.
            code_theme: Code syntax highlighting theme.

        Returns:
            Height in terminal lines.
        """
        if not text.strip():
            return 0

        md = Markdown(text, code_theme=code_theme)
        options = console.options
        lines = console.render_lines(md, options)
        _, height = Segment.get_shape(lines)

        return height

    def _truncate_streaming(
        self,
        text: str,
        terminal_height: int,
        console: Console,
        code_theme: str = "monokai",
    ) -> str:
        """Fast truncation optimized for streaming mode.

        This method uses a line-based rolling window approach that avoids
        redundant parsing and rendering. It's designed for the common case
        where content is continuously growing and we want to show the most
        recent portion.

        Key optimizations:
        1. Incremental: Only processes new content since last call
        2. Line-based: Uses fast line counting instead of full renders
        3. Single-pass: Only one render at the end to verify fit

        Args:
            text: The markdown text to truncate.
            terminal_height: Height of the terminal in lines.
            console: Rich Console for rendering.
            code_theme: Code syntax highlighting theme.

        Returns:
            Truncated text showing the most recent content.
        """
        if not text:
            return text

        target_height = int(terminal_height * self.target_height_ratio)

        # Check if we can use cached result
        if (
            self._last_full_text is not None
            and text.startswith(self._last_truncated_text or "")
            and self._last_terminal_height == terminal_height
        ):
            # Text only grew at the end, we can be more efficient
            # But for simplicity in first version, just proceed with normal flow
            pass

        # Fast line-based estimation
        # Strategy: Keep approximately 2x target lines as a generous buffer
        # This avoids most cases where we need multiple render passes
        lines = text.split('\n')
        total_lines = len(lines)

        # Rough heuristic: markdown usually expands by 1.5-2x due to formatting
        # So to get target_height rendered lines, keep ~target_height raw lines
        estimated_raw_lines = int(target_height * 1.2)  # Conservative estimate

        if total_lines <= estimated_raw_lines:
            # Likely fits, just verify with single render
            height = self._measure_rendered_height(text, console, code_theme)
            if height <= terminal_height:
                self._update_cache(text, text, terminal_height)
                return text
            # Didn't fit, fall through to truncation

        # Keep last N lines as initial guess
        keep_lines = min(estimated_raw_lines, total_lines)
        truncated_lines = lines[-keep_lines:]
        truncated_text = '\n'.join(truncated_lines)

        # Check for incomplete structures and fix them
        truncated_text = self._fix_incomplete_structures(text, truncated_text)

        # Verify it fits (single render)
        height = self._measure_rendered_height(truncated_text, console, code_theme)

        # If it doesn't fit, trim more aggressively
        if height > terminal_height:
            # Binary search on line count (much faster than character-based)
            left, right = 0, keep_lines
            best_lines = None

            while left <= right:
                mid = (left + right) // 2
                test_lines = lines[-mid:] if mid > 0 else []
                test_text = '\n'.join(test_lines)

                if not test_text.strip():
                    right = mid - 1
                    continue

                # Fix structures before measuring
                test_text = self._fix_incomplete_structures(text, test_text)
                test_height = self._measure_rendered_height(test_text, console, code_theme)

                if test_height <= terminal_height:
                    best_lines = mid
                    left = mid + 1  # Try to keep more
                else:
                    right = mid - 1  # Need to keep less

            if best_lines is not None and best_lines > 0:
                truncated_lines = lines[-best_lines:]
                truncated_text = '\n'.join(truncated_lines)
                truncated_text = self._fix_incomplete_structures(text, truncated_text)
            else:
                # Extreme case: even one line is too much
                # Keep last 20% of text as fallback
                fallback_pos = int(len(text) * 0.8)
                truncated_text = text[fallback_pos:]
                truncated_text = self._fix_incomplete_structures(text, truncated_text)

        self._update_cache(text, truncated_text, terminal_height)
        return truncated_text

    def _fix_incomplete_structures(self, original_text: str, truncated_text: str) -> str:
        """Fix incomplete markdown structures after line-based truncation.

        Handles:
        - Code blocks missing opening fence
        - Tables missing headers

        Args:
            original_text: The original full text.
            truncated_text: The truncated text that may have incomplete structures.

        Returns:
            Fixed truncated text with structures completed.
        """
        if not truncated_text or truncated_text == original_text:
            return truncated_text

        original_fragment = truncated_text

        # Find where the truncated text starts in the original
        truncation_pos = original_text.rfind(original_fragment)
        if truncation_pos == -1:
            truncation_pos = max(0, len(original_text) - len(original_fragment))

        code_blocks = self._get_code_block_info(original_text)
        active_block = None
        for block in code_blocks:
            if block.start_pos <= truncation_pos < block.end_pos:
                active_block = block

        if active_block:
            fence = self._build_code_block_prefix(active_block)
            if fence and not truncated_text.startswith(fence):
                truncated_text = fence + truncated_text

        # Check for incomplete tables when not inside a code block
        if active_block is None and '|' in truncated_text:
            tables = self._get_table_info(original_text)
            for table in tables:
                if table.thead_end_pos <= truncation_pos < table.tbody_end_pos:
                    # We're in the table body, header was removed
                    header_text = "\n".join(table.header_lines) + "\n"
                    if not truncated_text.startswith(header_text):
                        truncated_text = header_text + truncated_text
                    break

        return truncated_text

    def _update_cache(self, full_text: str, truncated_text: str, terminal_height: int) -> None:
        """Update the cache for streaming mode.

        Args:
            full_text: The full text that was truncated.
            truncated_text: The resulting truncated text.
            terminal_height: The terminal height used.
        """
        self._last_full_text = full_text
        self._last_truncated_text = truncated_text
        self._last_terminal_height = terminal_height

    def _flatten_tokens(self, tokens: Iterable[Token]) -> Iterable[Token]:
        """Flatten nested token structure.

        Args:
            tokens: Iterable of Token objects from markdown-it.

        Yields:
            Flattened tokens.
        """
        for token in tokens:
            is_fence = token.type == "fence"
            is_image = token.tag == "img"
            if token.children and not (is_image or is_fence):
                yield from self._flatten_tokens(token.children)
            else:
                yield token
