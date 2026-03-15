# Reversal Instructions: Streaming Clear/Reprint Optimizations

This document explains how to revert the two UX optimizations introduced for short markdown streams:

1. **No initial empty Live refresh** (Live starts with `None` renderable)
2. **Skip final clear+reprint when safe** (preserve streamed final frame)
3. **Reduce extra blank lines at reasoning transitions** (merge reasoning gap into markdown segment)
4. **Remove extra blank lines on preserved final frame + startup banner**
5. **Clear regular prompt echo line before formatted user message**

---

## 1) Revert "no initial empty Live refresh"

File: `src/fast_agent/ui/streaming.py`

- In `StreamingMessageHandle.__init__`:
  - Restore `initial_renderable` creation:
    - `Text("", style=...)` for plain mode
    - `Markdown("")` for markdown mode
  - Pass `initial_renderable` back into `Live(...)` instead of `None`.

This restores Rich's initial refresh behavior when entering Live mode.

---

## 2) Revert "preserve streamed final frame" handoff

### A. Remove preserve-final-frame controls from streaming handle

File: `src/fast_agent/ui/streaming.py`

- Remove state flag:
  - `self._preserve_final_frame`
- Remove methods:
  - `has_scrolled()`
  - `preserve_final_frame()`
- In `_shutdown_live_resources()`:
  - Remove the logic that sets `self._live.transient = False`
  - Keep default transient close behavior
- In `finalize(...)`:
  - Remove the preserve-path tweak that zeros `_max_render_height` / `_height_fudge`
- In `NullStreamingHandle`:
  - Remove `has_scrolled()` / `preserve_final_frame()` no-op methods
- In `StreamingHandle` protocol:
  - Remove `has_scrolled()` / `preserve_final_frame()` requirements

### B. Restore unconditional final assistant reprint

File: `src/fast_agent/agents/llm_agent.py`

- Remove `_can_preserve_streamed_final_frame(...)`
- In `generate_impl(...)` streaming path:
  - Remove `preserve_streamed_frame` decision logic
  - Remove call to `stream_handle.preserve_final_frame()`
  - Restore unconditional:

  ```python
  await self.show_assistant_message(
      result,
      additional_message=summary_text,
      render_markdown=render_markdown,
  )
  ```

- Remove now-unused imports (if any)

---

## 3) Revert reasoning-transition newline fix

File: `src/fast_agent/ui/stream_segments.py`

- In `StreamSegmentBuffer.consume_reasoning_gap()`:
  - Restore the old behavior that appends reasoning gap to a **plain** segment:

  ```python
  self._append_to_segment("plain", gap)
  ```

  instead of targeting markdown when `base_kind == "markdown"`.

This reverts the newline-spacing behavior back to the previous transition model.

---

## 4) Revert startup banner newline trim

File: `src/fast_agent/ui/prompt/session.py`

- In the first-help startup message, restore the trailing `"\n"` at the end
  of the `rich_print("...[/dim]\n")` string if you want the extra spacer line.

---

## 5) Revert prompt-echo line clear behavior

File: `src/fast_agent/ui/prompt/session_runtime.py`

- Remove `_clear_prompt_echo_line(...)` and `_ERASE_PREVIOUS_LINE_SEQ`.
- Remove the call to `_clear_prompt_echo_line(result)` in `run_prompt_once(...)`.

This restores the previous behavior where the raw prompt line remains visible
before formatted user output.

---

## 6) Remove tests added for these optimizations

- `tests/unit/fast_agent/agents/test_llm_agent_streaming_handoff.py` (delete)
- `tests/unit/fast_agent/ui/test_prompt_session_runtime.py` (delete)
- In `tests/unit/fast_agent/ui/test_streaming_mode_switch.py` remove tests for:
  - initial renderable being `None`
  - preserve-final-frame behavior
  - reasoning gap merge behavior
  - preserve-final-frame final padding behavior

---

## 7) Validate after rollback

Run:

```bash
uv run scripts/lint.py --fix
uv run scripts/typecheck.py
uv run pytest tests/unit
```

Ask operator to run e2e if needed.
