---
name: team-control
description: Protocol for orchestrator to spawn, monitor, and communicate with a team of agents
---

# Team Control Skill

You can spawn and manage a team of AI agents that work together autonomously.

## Key Concepts

- **Team agents are always reachable** — after finishing their primary task, they go `idle` (not offline). Send them a message and they wake up automatically.
- **PM controls completion** — the team runs as a closed loop. PM decides when the sprint is done.
- **You can intervene at any time** — audit, send messages, wait for results.

## Tools

### Spawn a Team
```
spawn_team_tool(
    template="agile-team",      # team template name
    project_brief="...",         # what the team should build
    mode="blocking"              # "blocking" waits for completion, "async" returns immediately
)
```

### Check Team Status
```
get_team_status(session_id="0039abf1")
→ { sprint_status: "running", agents: { "Linh - PM": "idle", "Minh - Dev": "running", ... } }
```

### Send Message to Any Agent
```
send_message_to_agent(to="Minh - Dev", message="Add feature X")
```
> If the agent is idle, they will be **automatically woken up** to process your message.

### Wait for an Agent
```
wait_for_agent(agent_name="Linh - PM", timeout_seconds=120)
```

### Get Team Results
```
get_team_result(session_id="0039abf1")
→ { workspace_contents: { src: [...], specs: [...] }, agents: { ... } }
```

## Workflow

1. **Spawn** the team with `spawn_team_tool(mode="blocking")`
2. Team works autonomously — PM coordinates BA, Dev, QE
3. When blocking returns, **check results** with `get_team_result`
4. If you need changes: `send_message_to_agent` to the relevant agent → they wake up and process it
5. Use `wait_for_agent` or `get_team_status` to track progress

## Tips

- Use `mode="blocking"` for most cases — it waits until the team finishes
- After team returns, you can still send follow-up messages to idle agents
- The PM agent coordinates the team internally — you only need to intervene for external requirements
