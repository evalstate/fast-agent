# fast-agent UI Styling Guide

This report summarizes how the fast-agent console UI is styled, where the styling rules live, and how they are applied across chat output, tool output, progress indicators, and auxiliary views.

## 1. Core primitives and configuration

### Message types and palette
- **File:** `src/fast_agent/ui/message_primitives.py`
- **MessageType** enum defines the canonical types: `USER`, `ASSISTANT`, `SYSTEM`, `TOOL_CALL`, `TOOL_RESULT`.
- **MESSAGE_CONFIGS** assigns **block colors**, **arrow glyphs**, and **highlight colors**:
  - `USER`: blue, arrow `▶`
  - `ASSISTANT`: green, arrow `◀`
  - `SYSTEM`: yellow, bullet `●`
  - `TOOL_CALL`/`TOOL_RESULT`: magenta, arrows `◀`/`▶`

### Logger settings (UI switches)
- **File:** `src/fast_agent/config.py` (`LoggerSettings`)
- Key toggles:
  - `show_chat`, `show_tools`
  - `truncate_tools`
  - `enable_markup`
  - `streaming` (markdown/plain/none)
  - `message_style` (`classic` or `a3`)
  - `progress_display`

### Console wiring
- **File:** `src/fast_agent/ui/console.py`
- Central `Console` instance (`console`) is used across UI modules.
- `ensure_blocking_console()` safeguards Rich output on event-loop managed TTYs.
- Optional stderr routing and dedicated `error_console`/`server_console` are defined here.

## 2. ConsoleDisplay – the main renderer

### Ownership and responsibilities
- **File:** `src/fast_agent/ui/console_display.py`
- This is the hub for:
  - Chat headers and separators
  - Content rendering (markdown vs JSON vs XML)
  - Bottom metadata bars
  - Assistant streaming display
  - Mermaid and MCP-UI link rendering

### Message header style
- Header left is constructed from `MESSAGE_CONFIGS`: `▎` + arrow + optional name.
- Header right is `right_info` (model/turn/metadata), styled dim.
- **Style variants:**
  - **`classic`**: header uses a horizontal rule + right info bracketed.
  - **`a3`**: header is compact, no horizontal rule, right info inline.

Relevant methods:
- `_format_header_line()` – chooses classic vs a3 layout.
- `_create_combined_separator_status()` – prints the header line and separator.

### Content rendering
- `display_message()` is the unified entry point.
- `_display_content()` adapts rendering:
  - JSON → `rich.pretty.Pretty`
  - XML-ish content → `rich.syntax.Syntax` (xml)
  - Markdown → `rich.markdown.Markdown` (with `prepare_markdown_content`)
  - Tool content defaults to dim style; user/assistant/system default to normal white.

### Markdown sanitization
- **File:** `src/fast_agent/ui/markdown_helpers.py`
- `prepare_markdown_content()` escapes HTML/XML outside code blocks to keep markdown readable and safe.

### Bottom metadata bars
- `_render_bottom_metadata()` handles the footer bars.
- **`a3` style:** compact, prefixed with `▎•` and bullet separators.
- **`classic` style:** `─| ... |───` line with pipe separators.
- `_format_bottom_metadata_compact()` and `_format_bottom_metadata()` handle layout and truncation.

## 3. Chat message display rules

### Assistant messages
- **File:** `src/fast_agent/ui/console_display.py` (`show_assistant_message`)
- Right info shows model via `format_model_display()`.
- Mermaid blocks are extracted and displayed below the message as clickable links.
- Streaming uses `streaming_assistant_message()` which renders with the same header primitives.

### User messages
- **Files:**
  - `src/fast_agent/agents/llm_agent.py` (grouping logic)
  - `src/fast_agent/ui/console_display.py` (`show_user_message`)
- Consecutive user messages are grouped in `LlmAgent._display_user_messages()` and shown once.
- Header right info can include:
  - `turn (N)` for grouped parts
  - `N parts` when multiple user messages are grouped
- Attachment prefaces appear as a dim `🔗` line before message content.

### System messages
- **File:** `src/fast_agent/ui/console_display.py` (`show_system_message`)
- Displays server counts and uses the SYSTEM style (yellow bullet).

## 4. Tool calls and tool results

### Tool display
- **File:** `src/fast_agent/ui/tool_display.py`
- Tool results are rendered with either:
  - `display_message()` for standard results
  - Manual header/metadata for structured content (e.g., skybridge)
- Status text includes error state and content characterization (text blocks, structured content, etc.).
- Bottom metadata can include transport channel and tool timing.

### Tool call formatting
- Tool call output is dimmed (uses `MessageType.TOOL_CALL`) and includes JSON/args formatting logic.

## 5. Streaming UI

- **Files:**
  - `src/fast_agent/ui/streaming.py`
  - `src/fast_agent/ui/stream_segments.py`
  - `src/fast_agent/ui/streaming_buffer.py`
- The streaming pipeline segments incremental text, supports markdown/plain modes, and gracefully truncates long outputs.
- `ConsoleDisplay.streaming_assistant_message()` hands off to `_StreamingMessageHandle` with shared header styling.

## 6. Progress display

- **Files:**
  - `src/fast_agent/ui/progress_display.py`
  - `src/fast_agent/ui/rich_progress.py`
- `RichProgressDisplay` uses a Rich progress bar with:
  - Spinner column
  - Status column (styled action labels and arrows)
  - Target column (bold blue)
  - Details column (white)
- Progress action → style mapping is centralized in `_get_action_style()`.

## 7. History and usage views

### Conversation history overview
- **File:** `src/fast_agent/ui/history_display.py`
- Provides compact timeline bars with block shading representing message volume.
- Colors reflect role: user/assistant/tool (with error variants).
- Context bar uses filled blocks with caution thresholds.

### Usage report
- **File:** `src/fast_agent/ui/usage_display.py`
- Emits a dim, table-like report with:
  - Header rule
  - `▎` glyph header
  - Column formatting with bold totals
- Uses `format_model_display()` for consistent model labeling.

## 8. Shell tool exit-bar styling

- **File:** `src/fast_agent/tools/shell_runtime.py`
- Exit codes are shown below tool output.
- Styling depends on `ConsoleDisplay._use_a3_style()`:
  - **a3:** `▎• exit code N` (blank line before/after)
  - **classic:** `─| exit code N |────`
- Exit code label color:
  - `0` → `white reverse dim`
  - `1` → `red reverse dim`
  - otherwise → `red reverse bold`

## 9. Supporting helpers

- **Model name formatting:** `src/fast_agent/ui/model_display.py`
- **Console-friendly XML/JSON:** handled in `ConsoleDisplay._display_content()` with rich syntax/pretty formatting.
- **Mermaid diagrams & MCP UI links:** `console_display.py` adds clickable link sections with dim bullets.

## 10. Style conventions at a glance

- **Headers:** `▎` + arrow glyphs (direction indicates speaker/tool direction).
- **Right-side metadata:** dim text, inline for a3, bracketed for classic.
- **Bottom bars:**
  - a3 → `▎•` + bullet separators
  - classic → `─| ... |────`
- **Tool content:** dimmed by default; errors use red tones.
- **System content:** yellow bullet/labeling.

---

If you want additional sections (ACP terminal views, TUI widgets, or prompt-loader visualizations), let me know and I can append them.
