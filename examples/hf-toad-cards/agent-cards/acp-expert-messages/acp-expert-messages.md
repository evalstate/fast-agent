---USER
How does an Agent communicate its plan to the Client?

---ASSISTANT
Let me search for the plan documentation.

```bash
rg -n --heading -C 5 'session_update.*plan\|sessionUpdate.*plan' agent-client-protocol/docs/protocol/ -g '*.mdx'
```

---USER
agent-client-protocol/docs/protocol/agent-plan.mdx
1----
2-title: "Agent Plan"
3-description: "How Agents communicate their execution plans"
4----
5-
6-Plans are execution strategies for complex tasks that require multiple steps.
7-
8:Agents may share plans with Clients through [`session/update`](./prompt-turn#3-agent-reports-output) notifications, providing real-time visibility into their thinking and progress.
9-
10-## Creating Plans
11-
12-When the language model creates an execution plan, the Agent **SHOULD** report it to the Client:
13-
14-```json
15-{
16-  "jsonrpc": "2.0",
17-  "method": "session/update",
18-  "params": {
19-    "sessionId": "sess_abc123def456",
20-    "update": {
21:      "sessionUpdate": "plan",

---ASSISTANT
From `agent-client-protocol/docs/protocol/agent-plan.mdx:6-21`:

Agents communicate plans via **`session/update` notifications** with `sessionUpdate: "plan"`. Here's the JSON-RPC format:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123def456",
    "update": {
      "sessionUpdate": "plan",
      "entries": [
        {
          "content": "Analyze the existing codebase structure",
          "priority": "high",
          "status": "pending"
        },
        {
          "content": "Identify components that need refactoring",
          "priority": "high", 
          "status": "pending"
        }
      ]
    }
  }
}
```

Let me find the Python SDK types:

```bash
rg -n --heading -A 15 'class PlanEntry' python-sdk/src/acp/schema.py
```

---USER
python-sdk/src/acp/schema.py
1375:class PlanEntry(BaseModel):
1376-    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
1377-    content: Annotated[str, Field(description="Human-readable description of what this task aims to accomplish.")]
1378-    priority: Annotated[PlanEntryPriority, Field(description="The relative importance of this task.")]
1379-    status: Annotated[PlanEntryStatus, Field(description="Current execution status of this task.")]

---ASSISTANT
From `python-sdk/src/acp/schema.py:1375-1379`:

```python
class PlanEntry(BaseModel):
    content: str      # Human-readable task description
    priority: PlanEntryPriority  # "high" | "medium" | "low"
    status: PlanEntryStatus      # "pending" | "in_progress" | "completed"
```

**Key points:**
1. Send plans via `session/update` with `sessionUpdate: "plan"`
2. Each entry has `content`, `priority`, and `status`
3. Plans are **dynamic** — send updated entries as the Agent progresses
4. Each update **replaces** the entire plan (send all entries, not just changes)

---USER
What options does an agent have to update the status of tool progress?

---ASSISTANT
Let me search for tool call status updates in the protocol docs.

```bash
rg -n --heading -C 4 'tool_call_update\|ToolCallStatus' agent-client-protocol/docs/protocol/tool-calls.mdx
```

---USER
agent-client-protocol/docs/protocol/tool-calls.mdx
41-## Updating
42-
43-As tools execute, Agents send updates to report progress and results.
44-
45:Updates use the `session/update` notification with `tool_call_update`:
46-
47-```json
48-{
49-  "jsonrpc": "2.0",
50-  "method": "session/update",
51-  "params": {
52-    "sessionId": "sess_abc123def456",
53-    "update": {
54:      "sessionUpdate": "tool_call_update",
55-      "toolCallId": "call_001",
56-      "status": "in_progress",
--
141-## Status
142-
143:Tool calls progress through different statuses during their lifecycle:
144-
145-<ResponseField name="pending">
146-  The tool call hasn't started running yet because the input is either streaming
147-  or awaiting approval
148-</ResponseField>
149-
150-<ResponseField name="in_progress">
151-  The tool call is currently running
152-</ResponseField>
153-
154-<ResponseField name="completed">
155-  The tool call completed successfully
156-</ResponseField>
157-
158-<ResponseField name="failed">The tool call failed with an error</ResponseField>

---ASSISTANT
From `agent-client-protocol/docs/protocol/tool-calls.mdx:41-158`:

**Tool Call Status Options:**
| Status | Meaning |
|--------|---------|
| `pending` | Not started yet (streaming input or awaiting approval) |
| `in_progress` | Currently executing |
| `completed` | Finished successfully |
| `failed` | Finished with an error |

**How to update status:**

Send a `session/update` notification with `sessionUpdate: "tool_call_update"`:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123def456",
    "update": {
      "sessionUpdate": "tool_call_update",
      "toolCallId": "call_001",
      "status": "in_progress"
    }
  }
}
```

Let me find the SDK types:

```bash
rg -n --heading -A 20 'class ToolCallUpdate' python-sdk/src/acp/schema.py
```

---USER
python-sdk/src/acp/schema.py
2277:class ToolCallUpdate(BaseModel):
2278-    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
2279-    content: Annotated[Optional[List[Union[ContentToolCallContent, FileEditToolCallContent, TerminalToolCallContent]]], Field(description="Replace the content collection.")] = None
2280-    kind: Annotated[Optional[ToolKind], Field(description="Update the tool kind.")] = None
2281-    locations: Annotated[Optional[List[ToolCallLocation]], Field(description="Replace the locations collection.")] = None
2282-    raw_input: Annotated[Optional[Any], Field(alias="rawInput", description="Update the raw input.")] = None
2283-    raw_output: Annotated[Optional[Any], Field(alias="rawOutput", description="Update the raw output.")] = None
2284-    status: Annotated[Optional[ToolCallStatus], Field(description="Update the execution status.")] = None

---ASSISTANT
From `python-sdk/src/acp/schema.py:2277-2284`:

```python
class ToolCallUpdate(BaseModel):
    content: Optional[List[...]]  # Replace content (text, diffs, terminals)
    kind: Optional[ToolKind]      # Update tool category
    locations: Optional[List[ToolCallLocation]]  # File locations being worked on
    raw_input: Optional[Any]      # Tool input parameters
    raw_output: Optional[Any]     # Tool output data
    status: Optional[ToolCallStatus]  # "pending"|"in_progress"|"completed"|"failed"
```

**Typical status flow:**
1. Create tool call with `sessionUpdate: "tool_call"` (status defaults to `pending`)
2. Update to `in_progress` when execution starts
3. Update to `completed` or `failed` with final `content`

**You can also update:**
- `content` — add progress text, diffs, or terminal output
- `locations` — report which files are being accessed (enables "follow-along" UX)
- `kind` — hint for UI icons (`read`, `edit`, `execute`, `search`, etc.)

---USER
What is the session/prompt method used for?

---ASSISTANT
Let me search for the session/prompt documentation.

```bash
rg -n --heading -C 4 'session/prompt' agent-client-protocol/docs/protocol/prompt-turn.mdx | head -50
```

---USER
agent-client-protocol/docs/protocol/prompt-turn.mdx
5-
6-A prompt turn represents a complete interaction cycle between the [Client](./overview#client) and [Agent](./overview#agent), starting with a user message and continuing until the Agent completes its response.
--
58-### 1. User Message
59-
60:The turn begins when the Client sends a `session/prompt`:
61-
62-```json
63-{
64-  "jsonrpc": "2.0",
65-  "id": 2,
66:  "method": "session/prompt",
67-  "params": {
68-    "sessionId": "sess_abc123def456",
69-    "prompt": [
70-      {
71-        "type": "text",
72-        "text": "Can you analyze this code for potential issues?"
73-      }
74-    ]
75-  }
76-}

---ASSISTANT
From `agent-client-protocol/docs/protocol/prompt-turn.mdx:58-76`:

The `session/prompt` method sends a **user message** to start a prompt turn. The Client sends it to the Agent with:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "session/prompt",
  "params": {
    "sessionId": "sess_abc123def456",
    "prompt": [
      {"type": "text", "text": "Can you analyze this code?"},
      {"type": "resource", "resource": {"uri": "file:///path/to/file.py", "text": "..."}}
    ]
  }
}
```

Let me find what content types are supported:

```bash
rg -n 'TextContentBlock\|ImageContentBlock\|ResourceContentBlock' python-sdk/src/acp/schema.py | head -10
```

---USER
python-sdk/src/acp/schema.py
85:class TextContentBlock(BaseModel):
145:class ImageContentBlock(BaseModel):
200:class ResourceContentBlock(BaseModel):

---ASSISTANT
From `python-sdk/src/acp/schema.py`:

**Prompt content types:**
- `TextContentBlock` (line 85) — plain text messages
- `ImageContentBlock` (line 145) — images with base64 data
- `ResourceContentBlock` (line 200) — file references with URI and content

**Prompt turn lifecycle:**
1. Client sends `session/prompt` with user content
2. Agent processes with LLM
3. Agent sends `session/update` notifications (plans, tool calls, message chunks)
4. Agent responds with `stopReason` when complete (`end_turn`, `max_tokens`, `cancelled`, etc.)
