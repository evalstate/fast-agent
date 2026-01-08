---
type: agent
name: PMO-orchestrator
default: true
agents:
- NY-Project-Manager
- London-Project-Manager
max_parallel: 128
child_timeout_sec: 120
max_display_instances: 20
---
Get project updates from the New York and London project managers. Ask NY-Project-Manager three times about different projects: Anthropic, evalstate/fast-agent, and OpenAI, and London-Project-Manager for economics review. Return a brief, concise combined summary with clear city/time/topic labels.
