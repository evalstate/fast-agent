---
name: session-investigator
description: Investigate fast-agent session and history files to diagnose issues. Use when a session ended unexpectedly, when debugging tool loops, when correlating sub-agent traces with main sessions, or when analyzing conversation flow and timing. Covers session.json metadata, history JSON format, message structure, tool call/result correlation, and common failure patterns.
---

# Session Investigator

Diagnose fast-agent session issues by examining session and history files.

## Session Directory Structure

Sessions are stored in `.fast-agent/sessions/<session-id>/`:

```
2601181023-Kob2h3/
├── session.json              # Session metadata
├── history_<agent>.json      # Current agent history  
└── history_<agent>_previous.json  # Previous save (rotation backup)
```

Session IDs encode creation time: `YYMMDDHHMM-<random>` (e.g., `2601181023` = 2026-01-18 10:23).

## Key Files

### session.json

```json
{
  "name": "2601181023-Kob2h3",
  "created_at": "2026-01-18T10:23:24.116526",
  "last_activity": "2026-01-18T10:39:42.873467",
  "history_files": ["history_dev_previous.json", "history_dev.json"],
  "metadata": {
    "agent_name": "dev",
    "first_user_preview": "is it possible to override..."
  }
}
```

### history_<agent>.json

```json
{
  "messages": [
    {
      "role": "user|assistant",
      "content": [{"type": "text", "text": "..."}],
      "tool_calls": {"<id>": {"method": "tools/call", "params": {"name": "...", "arguments": {}}}},
      "tool_results": {"<id>": {"content": [...], "isError": false}},
      "channels": {
        "fast-agent-timing": [{"type": "text", "text": "{\"start_time\": ..., \"end_time\": ..., \"duration_ms\": ...}"}],
        "fast-agent-tool-timing": [{"type": "text", "text": "{\"<tool_id>\": {\"timing_ms\": ..., \"transport_channel\": ...}}"}],
        "reasoning": [{"type": "text", "text": "..."}]
      },
      "stop_reason": "endTurn|toolUse|error",
      "is_template": false
    }
  ]
}
```

## Investigation Commands

### Basic inspection

```bash
# Message count
jq '.messages | length' history_dev.json

# Last N messages overview
jq '.messages[-5:] | .[] | {role, stop_reason, has_tool_calls: (.tool_calls != null), has_tool_results: (.tool_results != null)}' history_dev.json

# View specific message
jq '.messages[227]' history_dev.json
```

### Tool call correlation

Tool calls and results are linked by correlation ID. Valid pattern: assistant with `tool_calls` → user with matching `tool_results`.

```bash
# Check tool call/result pairing
jq '.messages[-10:] | to_entries | .[] | {
  index: .key, 
  role: .value.role,
  tool_calls: (if .value.tool_calls then (.value.tool_calls | keys) else [] end),
  tool_results: (if .value.tool_results then (.value.tool_results | keys) else [] end)
}' history_dev.json
```

### Find specific tool calls

```bash
# Find all calls to a specific tool
jq '.messages | to_entries | .[] | 
  select(.value.tool_calls != null) | 
  select(.value.tool_calls | to_entries | .[0].value.params.name == "agent__ripgrep_search") | 
  {index: .key, timing: (.value.channels."fast-agent-timing"[0].text)}' history_dev.json
```

## Session Statistics

### LLM Call Stats

```bash
# Total LLM time and call count
jq '[.messages[] | select(.role == "assistant") | 
  select(.channels."fast-agent-timing") | 
  .channels."fast-agent-timing"[0].text | fromjson | .duration_ms] | 
  {count: length, total_ms: add, avg_ms: (add/length), max_ms: max, min_ms: min}' history_dev.json

# LLM calls sorted by duration (slowest first)
jq '[.messages | to_entries | .[] | 
  select(.value.role == "assistant") | 
  select(.value.channels."fast-agent-timing") |
  {index: .key, duration_ms: (.value.channels."fast-agent-timing"[0].text | fromjson | .duration_ms)}] | 
  sort_by(-.duration_ms) | .[0:10]' history_dev.json
```

### Tool Execution Stats

```bash
# All tool timings aggregated
jq '[.messages[] | select(.channels."fast-agent-tool-timing") | 
  .channels."fast-agent-tool-timing"[0].text | fromjson | to_entries | .[].value.timing_ms] |
  {count: length, total_ms: add, avg_ms: (add/length), max_ms: max, min_ms: min}' history_dev.json

# Tool calls by name with timing
jq '[.messages | to_entries | .[] |
  select(.value.tool_calls) |
  (.value.tool_calls | to_entries | .[0]) as $tc |
  {index: .key, tool: $tc.value.params.name, 
   llm_ms: (.value.channels."fast-agent-timing"[0].text | fromjson | .duration_ms)}] |
  group_by(.tool) | 
  map({tool: .[0].tool, count: length, total_llm_ms: (map(.llm_ms) | add)}) |
  sort_by(-.count)' history_dev.json
```

### Session Timeline

```bash
# Session duration from first to last timing
jq '.messages | [
  (map(select(.channels."fast-agent-timing")) | first | .channels."fast-agent-timing"[0].text | fromjson | .start_time),
  (map(select(.channels."fast-agent-timing")) | last | .channels."fast-agent-timing"[0].text | fromjson | .end_time)
] | {start: .[0], end: .[1], duration_sec: ((.[1] - .[0]) | round)}' history_dev.json

# Message rate over time (messages per minute estimate)
jq '{
  messages: (.messages | length),
  llm_calls: [.messages[] | select(.role == "assistant" and .channels."fast-agent-timing")] | length,
  total_llm_ms: [.messages[] | select(.channels."fast-agent-timing") | .channels."fast-agent-timing"[0].text | fromjson | .duration_ms] | add,
  total_tool_ms: [.messages[] | select(.channels."fast-agent-tool-timing") | .channels."fast-agent-tool-timing"[0].text | fromjson | to_entries | .[].value.timing_ms] | add
} | . + {llm_sec: (.total_llm_ms/1000), tool_sec: ((.total_tool_ms//0)/1000)}' history_dev.json
```

### Sub-agent Stats

```bash
# Sub-agent calls (tools starting with "agent__")
jq '[.messages | to_entries | .[] |
  select(.value.tool_calls) |
  (.value.tool_calls | to_entries | .[0]) as $tc |
  select($tc.value.params.name | startswith("agent__")) |
  {index: .key, agent: $tc.value.params.name, 
   llm_ms: (.value.channels."fast-agent-timing"[0].text | fromjson | .duration_ms)}] |
  group_by(.agent) |
  map({agent: .[0].agent, calls: length, total_ms: (map(.llm_ms) | add), avg_ms: ((map(.llm_ms) | add) / length)})' history_dev.json
```

## Common Failure Patterns

### Unanswered Tool Call

**Symptom**: API error "No tool output found for function call"

**Pattern**: History ends with `assistant` message having `tool_calls` and `stop_reason: "toolUse"`, followed by `user` message WITHOUT matching `tool_results`.

```bash
# Check last message for pending tool call
jq '.messages[-1] | {role, has_tool_calls: (.tool_calls != null), stop_reason}' history_dev.json
```

**Cause**: Session interrupted mid-tool-loop, then resumed with new user input before tool completed.

**Fix**: Truncate history to last valid tool result:
```bash
# Find last user message with tool_results
jq '.messages | to_entries | map(select(.value.role == "user" and .value.tool_results != null)) | last | .key' history_dev.json

# Truncate (keep messages 0 to N inclusive, so use N+1)
jq '.messages = .messages[0:227]' history_dev.json > /tmp/fixed.json && mv /tmp/fixed.json history_dev.json
```

### Duplicate User Messages

**Pattern**: Two consecutive `user` messages before assistant response.

**Cause**: Often from `before_llm_call` hooks appending instructions. Check agent card's `tool_hooks` configuration.

## Sub-agent Trace Correlation

Sub-agent traces are saved as `<agent_name>-<timestamp>.json` in the working directory.

```bash
# List traces around session time
ls -la ripgrep_search*2026-01-18-10-3*.json

# Correlate via timing - match monotonic clock values
jq '.messages[-1].channels."fast-agent-timing"[0].text' ripgrep_search*.json
```

Compare `start_time`/`end_time` values between main session and sub-agent traces to correlate which sub-agent call corresponds to which main session tool call.

## Log File

Check `fastagent.jsonl` for errors during the session timeframe:

```bash
# Filter by timestamp range
cat fastagent.jsonl | while read line; do
  ts=$(echo "$line" | jq -r '.timestamp // empty' 2>/dev/null)
  if [[ "$ts" > "2026-01-18T10:20" && "$ts" < "2026-01-18T10:45" ]]; then
    echo "$line" | jq -c '{timestamp, level, message}'
  fi
done
```
