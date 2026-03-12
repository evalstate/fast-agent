---
name: meeting-participant
description: Protocol for joining and participating in multi-agent meetings
---

# Meeting Participant Skill

You are joining a team meeting. Follow this protocol exactly.

## Pre-requisites

You have access to the `meeting_room` MCP server which provides these tools:
- `join_meeting(meeting_id, role)` — announce your presence
- `wait_for_my_turn(meeting_id, role)` — block until it's your turn
- `get_transcript(meeting_id)` — read full discussion so far
- `speak(meeting_id, role, message)` — share your input
- `skip_turn(meeting_id, role)` — pass if nothing to add

## Protocol

### Step 1: Join the Meeting
```
join_meeting(meeting_id="<meeting_id>", role="<your_role>")
```

### Step 2: Wait for Your Turn
```
wait_for_my_turn(meeting_id="<meeting_id>", role="<your_role>")
```
This will block until the meeting facilitator gives you the floor.

### Step 3: When It's Your Turn

1. Read the transcript: `get_transcript(meeting_id="<meeting_id>")`
2. Read any relevant workspace files if needed
3. **Speak** with your input:
   ```
   speak(meeting_id="<meeting_id>", role="<your_role>", message="...")
   ```
   OR **Skip** if you have nothing new to add:
   ```
   skip_turn(meeting_id="<meeting_id>", role="<your_role>")
   ```

### Step 4: Repeat
After speaking, call `wait_for_my_turn` again. Repeat Steps 3-4 until the meeting ends.

## Guidelines

- **Build on what others said** — don't repeat points already made
- **Be specific** — reference files, code, or data from the workspace
- **Stay on agenda** — focus on the meeting's stated agenda
- **Escalate when stuck** — if the team is going in circles, say `VERDICT: ESCALATE` in your message to bring in additional expertise
- **Conclude clearly** — when asked for a verdict, state `VERDICT: PASS` or `VERDICT: FAIL` with reasoning
