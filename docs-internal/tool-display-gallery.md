# Tool display gallery

Generated with `COLUMNS=100 uv run scripts/tool_display_gallery.py --layout compare`.

## Compact

### MCP call/result

```text
▎◀ agent tool (MCP) huggingface hf_fs · id: call_…abcdef
▎▶ agent tool (MCP) huggingface hf_fs · text only 33 chars · 12.0s · id: call_…abcdef
```

### Parallel same-tool calls

```text
▎◀ agent tool (MCP) huggingface hf_fs · 5 requests
▎▶ agent tool (MCP) huggingface hf_fs · 5 results, 40 chars · 8.50s
```

### File reads

```text
▎▶ dev file read · ERROR
```

### Shell lifecycle

```text
▎◀ dev bash (/bin/bash) | yield 10s idle / 30s total · id: call_…456789
$ uv run scripts/lint.py
▎ exit code 0  1 line id: call_…456789
```

### Built-in edit

```text
▎◀ dev edit edit_file · id: call_…456789
edit_file preview: src/fast_agent/ui/tool_display.py
--- src/fast_agent/ui/tool_display.py
+++ src/fast_agent/ui/tool_display.py
@@ -1 +1 @@
-layout = "full"
+layout = "compact"
▎▶ dev edit edit_file · text only 26 chars · 34ms · id: call_…456789
```

## Full

### MCP call/result

```text

▎◀ agent tool call - huggingface__hf_fs · id: call_…abcdef
{'path': 'datasets/evalstate/tool-fixtures', 'detail': True}

▎▶ agent tool result - text only 33 chars · id: call_…abcdef
{"kind":"directory","entries":12}
```

### Parallel same-tool calls

```text

▎◀ agent tool call - huggingface__hf_fs · id: call_…lel_01
{'path': 'datasets/example/1'}

▎◀ agent tool call - huggingface__hf_fs · id: call_…lel_02
{'path': 'datasets/example/2'}

▎◀ agent tool call - huggingface__hf_fs · id: call_…lel_03
{'path': 'datasets/example/3'}

▎◀ agent tool call - huggingface__hf_fs · id: call_…lel_04
{'path': 'datasets/example/4'}

▎◀ agent tool call - huggingface__hf_fs · id: call_…lel_05
{'path': 'datasets/example/5'}

▎▶ agent tool result - text only 8 chars · id: call_…lel_01
entry-1


▎▶ agent tool result - text only 8 chars · id: call_…lel_02
entry-2


▎▶ agent tool result - text only 8 chars · id: call_…lel_03
entry-3


▎▶ agent tool result - text only 8 chars · id: call_…lel_04
entry-4


▎▶ agent tool result - text only 8 chars · id: call_…lel_05
entry-5
```

### File reads

```text

▎▶ dev file read - llm/sampling_converter.py (offset 90, 60 lines)

line 90
line 91


▎▶ dev file read - llm/sampling_converter.py (offset 90, 60 lines)

line 90
line 91
line 92
line 93

(+4 more lines)

▎▶ dev file read - ERROR
Permission denied: llm/private.py
```

### Shell lifecycle

```text

▎◀ dev bash (/bin/bash) | yield 10s idle / 30s total · id: call_…456789
uv run scripts/lint.py

▎▶ dev tool result - text only 42 chars · id: call_…456789
All checks passed!
▎ exit code 0  1 line id: call_…456789
```

### Built-in edit

```text

▎◀ dev tool call - edit_file · id: call_…456789
edit_file preview: src/fast_agent/ui/tool_display.py
--- src/fast_agent/ui/tool_display.py
+++ src/fast_agent/ui/tool_display.py
@@ -1 +1 @@
-layout = "full"
+layout = "compact"

▎▶ dev tool result - text only 26 chars · id: call_…456789
Success. Replaced 1 match.
```
