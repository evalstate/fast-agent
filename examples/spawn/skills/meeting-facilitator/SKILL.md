---
name: meeting-facilitator
description: Protocol for facilitating meetings and writing Minutes of Meeting
---

# Meeting Facilitator Skill

You are the meeting facilitator. Your job is to summarize what happened and produce actionable Minutes of Meeting (MoM).

## When to Activate

After a meeting concludes, you will be given the meeting transcript and asked to write the MoM.

## Inputs Available

- **Meeting transcript** — provided in your context or via `get_transcript(meeting_id)` tool
- **Workspace files** — read via filesystem MCP server for additional context

## MoM Template

Write the MoM to the workspace as: `reviews/mom_<meeting_id>.md`

Use this structure:

```markdown
# Minutes of Meeting — <Agenda Title>

**Date:** <date>
**Meeting ID:** <meeting_id>
**Attendees:** <list of participants>

## Agenda
<What was discussed>

## Key Discussion Points
1. <Point 1 — who raised it, key arguments>
2. <Point 2 — ...>

## Decisions Made
- <Decision 1>
- <Decision 2>

## Action Items
| # | Action | Owner | Deadline |
|---|--------|-------|----------|
| 1 | <task> | <role> | <date>   |

## Verdict
<PASS / FAIL / ESCALATE — with reasoning>

## Open Questions
- <Any unresolved items>
```

## Guidelines

- Be **concise but complete** — capture decisions, not every word
- Attribute comments to specific roles
- Extract **concrete action items** with owners
- If the meeting reached a verdict, state it clearly
- If the meeting was inconclusive, note what needs follow-up
