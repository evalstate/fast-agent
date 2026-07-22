<!--
  GENERATED FILE — DO NOT EDIT.
  Source: generate_reference_docs.py
-->

#### TUI environment variables

| Symbol | Variable | Purpose |
| --- | --- | --- |
| `FAST_AGENT_SHELL_CHILD_ENV` | `FAST_AGENT_SHELL_CHILD` | Set to `1` in child shells opened from the TUI with `!`. |

#### TUI-related settings

| Setting | Environment variable | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `logger.streaming` | `LOGGER__STREAMING` | `Literal['markdown', 'plain', 'none']` | `markdown` | Streaming renderer for assistant responses. |
| `logger.enable_prompt_marks` | `LOGGER__ENABLE_PROMPT_MARKS` | `bool` | `True` | Emit OSC 133 prompt marks for supported terminals. |
| `logger.progress_display` | `LOGGER__PROGRESS_DISPLAY` | `bool` | `True` | Enable or disable progress display. |
| `logger.show_chat` | `LOGGER__SHOW_CHAT` | `bool` | `True` | Show user and assistant messages on the console. |
| `logger.show_tools` | `LOGGER__SHOW_TOOLS` | `bool` | `True` | Show MCP server tool calls on the console. |
| `logger.truncate_tools` | `LOGGER__TRUNCATE_TOOLS` | `bool` | `True` | Truncate long tool call display. |
| `logger.theme_file` | `LOGGER__THEME_FILE` | `str \| None` | `None` | Optional Rich theme file for console styles. |
| `logger.code_theme` | `LOGGER__CODE_THEME` | `str` | `native` | Pygments/Rich syntax theme for Markdown code rendering. |
| `logger.render_fences_with_syntax` | `LOGGER__RENDER_FENCES_WITH_SYNTAX` | `bool` | `True` | Render Markdown code fences with Rich Syntax. |
| `logger.code_word_wrap` | `LOGGER__CODE_WORD_WRAP` | `bool` | `True` | Wrap Syntax-rendered code blocks instead of cropping. |
| `logger.apply_patch_preview_max_lines` | `LOGGER__APPLY_PATCH_PREVIEW_MAX_LINES` | `int \| None` | `120` | Maximum lines to show in apply_patch previews. |
| `logger.terminal_images.enabled` | `LOGGER__TERMINAL_IMAGES__ENABLED` | `bool` | `True` | Render image content in capable terminals. |
| `logger.terminal_images.backend` | `LOGGER__TERMINAL_IMAGES__BACKEND` | `Literal['auto', 'textual-image', 'kitty', 'sixel', 'halfcell', 'unicode', 'none']` | `auto` | Terminal image backend to use. |
| `logger.terminal_images.width` | `LOGGER__TERMINAL_IMAGES__WIDTH` | `TerminalImageSize` | `80%` | Image render width. |
| `logger.terminal_images.height` | `LOGGER__TERMINAL_IMAGES__HEIGHT` | `TerminalImageSize` | `auto` | Image render height. |
| `shell_execution.tool_profile` | `SHELL_EXECUTION__TOOL_PROFILE` | `ShellToolProfile` | `minimal_process` | Model-facing Bash/Process contract. |
| `shell_execution.output_display_lines` | `SHELL_EXECUTION__OUTPUT_DISPLAY_LINES` | `int \| None` | `5` | Maximum shell/read_text_file lines to display. |
| `shell_execution.show_bash` | `SHELL_EXECUTION__SHOW_BASH` | `bool` | `True` | Show shell command output on the console. |
| `shell_execution.interactive_use_pty` | `SHELL_EXECUTION__INTERACTIVE_USE_PTY` | `bool` | `True` | Use a PTY for interactive prompt shell commands. |
| `shell_execution.timeout_seconds` | `SHELL_EXECUTION__TIMEOUT_SECONDS` | `int` | `90` | Maximum seconds without command output before termination. |
| `shell_execution.warning_interval_seconds` | `SHELL_EXECUTION__WARNING_INTERVAL_SECONDS` | `int` | `30` | Show timeout warnings every N seconds. |
| `shell_execution.process_poll_max_wait_seconds` | `SHELL_EXECUTION__PROCESS_POLL_MAX_WAIT_SECONDS` | `int` | `250` | Maximum managed-process wait. |
| `shell_execution.managed_process_poll_history_folding` | `SHELL_EXECUTION__MANAGED_PROCESS_POLL_HISTORY_FOLDING` | `Literal['auto', 'on', 'off']` | `auto` | Fold repetitive quiet managed-process polling. |
| `tui.completion_menu_reserved_lines` | `TUI__COMPLETION_MENU_RESERVED_LINES` | `int` | `6` | Prompt-toolkit lines reserved below the input for completion menus. |
