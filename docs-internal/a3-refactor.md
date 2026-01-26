# A3 Styling Refactor Assessment (Style Strategy)

Date: 2026-01-24

## Goal
Adopt a **style strategy** to eliminate scattered `if message_style == "a3"` branches, and to
separate **application concerns** (what to show) from **display concerns** (how to render it).

## Current Status (Implemented)
The style strategy **has been implemented** for the core console and tool rendering paths:

- **Strategy module:** `src/fast_agent/ui/message_styles.py`
  - `MessageStyle` protocol
  - `A3MessageStyle` / `ClassicMessageStyle`
  - `resolve_message_style()`
- **Console rendering:** `ConsoleDisplay` now holds `self._style` and delegates header, bottom
  metadata, shell exit lines, and spacing to the strategy.
- **Tool rendering:** `ToolDisplay` reads `self._display.style` and uses the strategy for
  structured tool output and tool updates.
- **Streaming:** `StreamingMessageHandle` uses `ConsoleDisplay._format_header_line()` which
  delegates to the active style strategy.

The old `_use_a3_style()` branching has been removed from the primary rendering pipeline.

## Remaining Work

1. **Elicitation UI alignment**
   - `elicitation_style.py` and `elicitation_form.py` still hold A3-specific styles inline.
   - Consider a shared style adapter (or a smaller prompt-toolkit-specific style strategy).

2. **Ancillary surfaces (optional)**
   - Usage display, MCP status, progress, and link renderers remain independent of the
     message style strategy. These can stay as-is unless full UI parity is desired.

## Strategy Overview (Implemented Design)

### MessageStyle protocol (implemented)

```python
class MessageStyle(Protocol):
    name: str

    def header_line(self, left: str, right: str, width: int) -> Text: ...
    def bottom_metadata(
        self,
        items: list[str],
        highlight_index: int | None,
        highlight_color: str,
        max_item_length: int | None,
        width: int,
    ) -> Text | None: ...
    def shell_exit_line(self, exit_code: int, width: int) -> Text: ...
    def after_header_spacing(self) -> int: ...
```

### ConsoleDisplay integration

```python
class ConsoleDisplay:
    def __init__(...):
        self._style = resolve_message_style(self._logger_settings.message_style)

    def _create_combined_separator_status(self, left: str, right: str = "") -> None:
        line = self._style.header_line(left, right, console.console.size.width)
        console.console.print()
        console.console.print(line, markup=self._markup)
        for _ in range(self._style.after_header_spacing()):
            console.console.print()
```

Bottom metadata:

```python
line = self._style.bottom_metadata(...)
if line is not None:
    console.console.print()
    console.console.print(line, markup=self._markup)
    console.console.print()
```

Shell exit:

```python
line = self._style.shell_exit_line(exit_code, console.console.size.width)
console.console.print()
console.console.print(line)
if self._style.after_header_spacing():
    console.console.print()
```

## Risk & Compatibility Notes
- Preserve behavior for **classic** and **a3**; ensure unit snapshots or visual tests remain
  unchanged.
- Do not change `ConsoleDisplay` public API.
- Keep compact A3 bullet rendering consistent with `docs/a3-styling.md`.

## Benefits (Realized)
- Removed conditional clutter from display flow.
- Centralized styling rules for A3/classic.
- Enabled future styles without refactoring `ConsoleDisplay` or tool display logic again.

## Display Code Path Sweep (Current Flow)

### Entry points (application -> display)
- **LLM Agent** (`src/fast_agent/agents/llm_agent.py`)
  - `show_user_message()` -> `ConsoleDisplay.show_user_message()`
  - `show_assistant_message()` -> `ConsoleDisplay.show_assistant_message()`
  - `streaming_assistant_message()` -> `StreamingMessageHandle` (via `ConsoleDisplay`)
  - Tool calls/results are routed through `ConsoleDisplay.show_tool_call()` / `show_tool_result()`
- **History review** (`ui/history_actions.py`) replays turns through `ConsoleDisplay` helpers.
- **Shell runtime** (`tools/shell_runtime.py`) uses `ConsoleDisplay` for tool output.
- **MCP URL elicitation** (`mcp/elicitation_handlers.py`) uses `ConsoleDisplay.show_url_elicitation()`.

### Rendering pipeline (core chat output)
- `ConsoleDisplay.display_message()`
  - Builds header and delegates to `_create_combined_separator_status()`.
  - Renders main body via `_display_content()`.
  - Renders bottom metadata via `_render_bottom_metadata()`.
- `ConsoleDisplay.show_assistant_message()`
  - Adds reasoning pre-content and forwards to `display_message()`.
- `ConsoleDisplay.show_user_message()`
  - Formats attachments/turn metadata then forwards to `display_message()`.

### Streaming
- `ConsoleDisplay.streaming_assistant_message()` creates `StreamingMessageHandle`.
- `StreamingMessageHandle._build_header()` uses `ConsoleDisplay._format_header_line()`.
- Streaming does **not** use bottom metadata; it only renders the header + stream body.

### Tool output paths (strategy-driven)
- `ToolDisplay.show_tool_result()`
  - Structured tool results use the style strategy instead of branching on a3/classic.
  - Non-structured results reuse `ConsoleDisplay.display_message()`.
- `ToolDisplay.show_tool_update()`
  - Updates use the style strategy for headers and separators.

### Other display surfaces
- **Mermaid links** and **MCP-UI links** render inline bullet lists (`ConsoleDisplay`).
- **Usage display** and **MCP status** render their own tables with Rich (not style-aware).
- **Progress display** (`ui/rich_progress.py`) uses a `` prefix and arrows, but no style flag.
- **Elicitation forms** use a static prompt-toolkit style (`elicitation_style.py`).

## Outcome
The style strategy refactor is complete for core chat rendering, tool rendering, and streaming.
Remaining work is limited to optional UI surfaces (elicitation and ancillary displays) if we
want full parity across all output modes.
