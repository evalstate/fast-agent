# A3 Styling

The **A3** display style is a compact, inline console layout used by the interactive UI. It minimizes horizontal rules and keeps message metadata close to the header line.

## Enable A3

Set the logger style in your config:

```yaml
logger:
  message_style: a3
```

The style defaults to `a3` if not specified.

## Visual elements

### Message header

A3 uses a single inline header line:

```
▎◀ assistant_name [dim right-info]
```

- **Left block:** `▎` colored by message type (user/assistant/system/tool).
- **Arrow glyph:** indicates message direction (`▶` for user/tool-result, `◀` for assistant/tool-call, `●` for system).
- **Right info:** appended inline and dimmed (no brackets or filler rules).

### Bottom metadata (compact)

When metadata is shown, A3 uses a bullet list line:

```
▎• item • item • item
```

- Prefix: `▎• ` (dim)
- Separator: ` • ` (dim)
- Highlighted items use the message highlight color.

### Shell exit codes

Exit code banners use the compact bullet layout:

```
▎•  exit code 1
```

### Prompt listing and selection

`/prompt` and `/prompts` list prompts in a compact, A3-friendly format:

```
[ 1] server•prompt_name Title
     Description line
     args: required* , optional
```

## Classic vs A3

| Element | Classic | A3 |
| --- | --- | --- |
| Header | Horizontal rule + bracketed right info | Inline, no rule |
| Metadata | `─| item |────` | `▎• item • item` |
| Spacing | Extra blank lines | Tighter vertical spacing |

## Notes for implementers

- Use `ConsoleDisplay.display_message()` and related helpers to ensure A3 is applied consistently.
- For tool results with structured content, A3 uses the compact bullet bar rather than the classic `─|` separator.
- Prompt listing UI should be rendered through `ConsoleDisplay` to match A3.


## Tool update notifications

When the client receives a tool list changed notification, A3 shows a compact update line:

```
▎• Updating tools for server <name>
```

If the UI is running without prompt_toolkit, the fallback path still renders the compact line and avoids classic full-width separators.

