## Ripgrep Usage

> ⚠️ **IMPORTANT: ripgrep (`rg`) does NOT support `-R` or `--recursive`.**
>
> Ripgrep is recursive by default. Using `-R` will cause an error. Just run `rg pattern path/`.

### Core Rules

- **Always execute** rg commands—don't just suggest them.
- **Don't infer** behavior beyond retrieved lines. If you need more detail, run another search.
- **Exit code 1** = no matches (not an error).
- **Narrow early**: use `-t` or `-g` to limit file types/paths.
- **Count first** for broad terms: `rg -c 'pattern' path/` and summarize if >50 matches.
- **Path hygiene**: if the repo root isn't explicit, check `pwd`/`ls` and stop if the expected repo isn't present.
- **Attempt budget**: max 3 discovery attempts, then conclude "not found in workspace."

### Standard Exclusions (always include)

```bash
-g '!.git/*' -g '!node_modules/*' -g '!__pycache__/*' -g '!*.pyc' -g '!.venv/*' -g '!venv/*'
```

For doc/concept searches, also consider:
```bash
-g '!ripgrep_search*.json' -g '!stream-debug/*' -g '!fastagent.jsonl'
```

### File Discovery

- **Filename discovery**: `rg --files -g '*pattern*'` (no search pattern needed)
- **Content discovery**: `rg -l 'pattern'` (requires search pattern)
- **Never** use `rg -l` without a search pattern
- **Never** use `ls -R`; prefer ripgrep for all discovery
- **Never** use full paths as globs (`-g 'src/foo.py'`)
- **Never** pipe `rg --files` to `grep`; use multiple `-g` patterns instead

### Useful Flags

| Flag | Purpose |
|------|---------|
| `-F` | Literal (fixed-string) match |
| `-S` | Smart case |
| `-i` | Case-insensitive |
| `-w` | Whole word match |
| `-l` | List files with matches |
| `-c` | Count matches per file |
| `-t <type>` | Filter by type: `py`, `js`, `md`, `json`, etc. |
| `-g '<glob>'` | Glob pattern, e.g., `-g '*.py'` or `-g '!node_modules/*'` |
| `-n` | Line numbers |
| `--heading` | Group by file |
| `-C N` | Context lines (before and after) |
| `-A N` / `-B N` | Context lines after/before only |
| `--max-count=N` | Limit matches per file |
| `-U` | Multiline search |
| `-P` | PCRE2 regex |
| `-a` | Treat binary as text |
| `--hidden` | Include hidden files |
| `--no-ignore` | Don't respect ignore files |
| `-uuu` | Shorthand for `--hidden --no-ignore --no-ignore-parent` |

### Literal Safety

Use `-F` or escape regex metacharacters for literal searches:
```bash
rg -F '.fast-agent'
rg '\.fast-agent'
```

### Output Control

- Prefer `rg -l` for content discovery over `rg -c` (avoid log explosions).
- Use `--max-count 1`, `--stats`, or `head -n 50` to limit output.
- Never use `rg -c ''` for structure (it just counts lines).

### Handling Large Results

When a search might return many matches, **count first**:

```bash
rg -c 'pattern' path/
```

If >50 matches, summarize for the user:
- Total match count
- Top files by match count
- Suggestions to narrow the search

### Docs/Spec Search Pattern

1. List docs files:
```bash
rg --files -g '*.md' -g '*.mdx' -g '*.rst'
```
2. Search within docs:
```bash
rg -n 'pattern' -g '*.md' -g '*.mdx' -g '*.rst'
```

### File Content Requests

If the user asks to "show" a file:
1) Confirm existence:
```bash
rg --files -g 'name'
```
2) Then show content:
```bash
rg -n '.' path/to/file | head -n 200
```
