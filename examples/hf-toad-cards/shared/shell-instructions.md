## Shell Execution Model

> ⚠️ **Each shell command runs in a fresh, isolated environment.**

This means:
- **Variables don't persist** — `VAR=x` in one call is gone in the next
- **Directory changes don't persist** — `cd` has no effect on subsequent calls
- **Multi-step operations must be chained** — use `&&` to combine dependent commands

**❌ Wrong** (fails with "missing operand" or "no such file"):
```bash
MY_DIR="/some/path"
```
```bash
mkdir -p "$MY_DIR"   # $MY_DIR is empty here!
```

**✅ Correct** (single command with inline literal or chaining):
```bash
mkdir -p /some/path && cd /some/path && git clone ...
```

**Always use literal paths** rather than variables across commands.
