## Ripgrep Usage

> ⚠️ **IMPORTANT: ripgrep (`rg`) does NOT support `-R` or `--recursive`.**
>
> Ripgrep is recursive by default. Using `-R` will cause an error. Just run `rg pattern path/`.

### Useful Flags

| Flag | Purpose |
|------|---------|
| `-i` | Case-insensitive |
| `-w` | Whole word match |
| `-l` | List files only |
| `-c` | Count matches per file |
| `-t <type>` | Filter by type: `py`, `js`, `md`, `json`, etc. |
| `-g '<glob>'` | Glob pattern, e.g., `-g '*.py'` or `-g '!node_modules/*'` |
| `-n` | Line numbers |
| `--heading` | Group by file |
| `-C N` | Context lines (before and after) |
| `-A N` / `-B N` | Context lines after/before only |
| `--max-count=N` | Limit matches per file |

### Standard Exclusions

Always exclude noise directories:
```bash
-g '!.git/*' -g '!node_modules/*' -g '!__pycache__/*' -g '!*.pyc' -g '!.venv/*' -g '!venv/*'
```

### Handling Large Results

When a search might return many matches (>50 lines), **count first**:

```bash
rg -c 'pattern' path/
```

Then drill into specific files if needed. Summarize for the user:
- Total match count
- Top files by match count
- Suggestions to narrow the search
