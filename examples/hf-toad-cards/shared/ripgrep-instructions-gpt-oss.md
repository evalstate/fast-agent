## Ripgrep Usage

> ⚠️ **IMPORTANT: ripgrep (`rg`) does NOT support `-R` or `--recursive`.**
>
> Ripgrep is recursive by default. Using `-R` will cause an error. Just run `rg pattern path/`.

### Core Search Practices

- **Narrow early**: use `-t` or `-g` to limit file types/paths.
- **Count first for broad terms**: `rg -c 'pattern' path/` and summarize if large.
- **Docs/spec queries**: list docs files first, then search within them.
- **Avoid `ls -R`**: use `rg --files` or `rg -l` for discovery.
- **Path hygiene**: if the repo root isn’t explicit, check `pwd`/`ls` and stop if the expected repo isn’t present.
- **Attempt budget**: max 3 discovery attempts, then conclude “not found in workspace.”

### Useful Flags

| Flag | Purpose |
|------|---------|
| `-F` | Literal (fixed-string) match |
| `-S` | Smart case |
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
| `-U` | Multiline search |
| `-P` | PCRE2 regex |
| `-a` | Treat binary as text |

### File Discovery Rules

- Find files by name with: `rg --files -g '*pattern*'`
- **Never** use `rg -l` without a search pattern
- **Never** use full paths as globs (`-g 'src/foo.py'`)
- **Never** pipe `rg --files` to `grep`; use multiple `-g` patterns instead

### Literal Safety

Use `-F` or escape regex metacharacters for literal searches, e.g.:
```bash
rg -F '.fast-agent'
rg '\.fast-agent'
```

### Standard Exclusions

For broad or repo-wide searches, exclude noise directories and JSON/JSONL logs:
```bash
-g '!.git/*' -g '!node_modules/*' -g '!__pycache__/*' -g '!*.pyc' -g '!.venv/*' -g '!venv/*' -g '!*.json' -g '!*.jsonl' -g '!stream-debug/*'
```

If you need to search JSON or JSONL content, remove the JSON exclusions.

If you are targeting a specific file (explicit path or a single `rg --files -g 'name'` lookup), complex exclusions are optional.

### Output Control (Avoid Log Explosions)

- Prefer `rg -l` for discovery over `rg -c`.
- Use `--max-count 1`, `--stats`, or `head -n 50` to limit output.
- Never use `rg -c ''` for structure (it just counts lines).

### Handling Large Results

When a search might return many matches (>50 lines), **count first**:

```bash
rg -c 'pattern' path/
```

Then drill into specific files if needed. Summarize for the user:
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

If the user asks to “show” a file:
1) Confirm existence:
```bash
rg --files -g 'name'
```
2) Then show content:
```bash
rg -n '.' path/to/file | head -n 200
```
