# Ripgrep Search Agent Improvements

## Problem Analysis

The ripgrep_search agent was experiencing two types of errors:

1. **Invalid `-R` flag usage** (2 occurrences in trace)
   - Model was adding `-R` flag despite instructions
   - Caused: `error: Found argument '-R' which wasn't expected`
   - Ripgrep doesn't support `-R` (it's recursive by default)

2. **Tool hallucination** (4 occurrences across traces)
   - Model trying to call non-existent tools: `commentary`, `exec`, `executescript`
   - Caused: `Tool call validation failed: attempted to call tool 'X' which was not in request.tools`
   - Model has only ONE tool (`execute`) but tried to call others

## Solutions Implemented

### 1. ✅ Strengthened Instructions (.fast-agent/tool-cards/ripgrep-search.md)

**Added Critical Rule #3:**
```markdown
3. **You have ONLY ONE tool: `execute`** (for running shell commands)
   - After running ripgrep commands, RETURN RESULTS AS TEXT
   - DO NOT attempt to call tools like 'commentary', 'exec', 'executescript', etc.
   - Use the Output Format specified below to present findings
```

**Enhanced Critical Rule #1:**
```markdown
1. **NEVER use `-R` or `--recursive` flags** - ripgrep does NOT support them and will error.
   - ❌ WRONG: `rg pattern -R path/`
   - ✅ RIGHT: `rg pattern path/`
   - Ripgrep is ALWAYS recursive by default when given a directory.
```

**Added "Valid Flags Only" section:**
- Whitelist of allowed flags
- Explicit blacklist: `DO NOT USE: -R, --recursive, -r`

### 2. ✅ Updated Few-Shot Examples (.fast-agent/tool-cards/ripgrep-tuning.json)

**Before:**
```json
"tool_calls": {
  "t1": { ... }
}
```

**After:**
```json
"tool_calls": {
  "fc_399c6f22-9992-4ab4-9d5b-134fbe7d47ae": { ... }
}
```

All 6 tool call/result pairs now use realistic UUID format matching actual runtime behavior.

### 3. ✅ NEW: Safety Hook (.fast-agent/hooks/strip_ripgrep_r_flag.py)

**Purpose:** Intercept and fix `-R` flags before they cause errors

**How it works:**
- Registers as `before_tool_call` hook
- Inspects all `execute` tool calls
- Detects ripgrep commands with `-R` flag
- Strips the flag before execution
- Logs a warning about the correction

**Test results:**
```
✅ 'rg -n -R 'pattern' src/' -> 'rg -n 'pattern' src/'
✅ 'rg -R -n 'pattern' src/' -> 'rg -n 'pattern' src/'
✅ 'rg 'pattern' -R' -> 'rg 'pattern''
✅ 'rg -c -R -n 'pattern'' -> 'rg -c -n 'pattern''
```

**Hook configuration:**
```yaml
tool_hooks:
  before_tool_call: ../hooks/strip_ripgrep_r_flag.py:strip_r_flag_from_ripgrep
  after_turn_complete: ../hooks/save_history.py:save_history_to_file
```

## Results

### Before Improvements (trace 22:41:54)
- Total errors: 4
- `-R` flag errors: 2
- Commands executed: 11
- Final outcome: Blocked by syntax errors

### After Improvements (traces 23:52:54, 23:52:59)
- Total errors: 0 (ripgrep related)
- `-R` flag errors: 0
- Commands executed: 3 and 1 (more efficient)
- Final outcome: Successfully found matches

## Architecture: Defense in Depth

1. **Instructions** - Tell the model not to use `-R`
2. **Examples** - Show correct usage patterns
3. **Hook** - Automatically fix if model still uses `-R`

This layered approach ensures:
- Best case: Model learns not to use `-R`
- Fallback: Hook catches and fixes mistakes
- Result: Zero user-facing errors

## Files Changed

- `.fast-agent/tool-cards/ripgrep-search.md` (76→99 lines)
- `.fast-agent/tool-cards/ripgrep-tuning.json` (tool IDs updated)
- `.fast-agent/hooks/strip_ripgrep_r_flag.py` (NEW)
- Backups saved with `.backup` extension

## Testing

Run the hook test:
```bash
uv run python test_strip_r_hook.py
```

Expected output:
```
✅ All tests passed!
```

## Next Steps

Monitor traces for:
1. Reduced `-R` flag usage (instructions working)
2. Hook activations (logged as warnings)
3. Tool hallucination reduction (instructions addressing this)

If tool hallucination persists, consider:
- Switching to a more instruction-adherent model
- Adding validation hook for tool names
