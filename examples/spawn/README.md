# Dynamic Agent Spawn Examples

Example configurations demonstrating `fast_agent.spawn` capabilities.

## Single Agent Spawn (`single_spawn/`)

Spawn an isolated sub-agent to handle a task in a subprocess.

## Team Spawn (`team_spawn/`)

Orchestrate a team of agents using a YAML template with dependency DAG,
review meetings, and shared workspace.

## Quick Start

```bash
cd examples/spawn/single_spawn
# Edit fastagent.secrets.yaml with your API keys
uv run python single_spawn.py
```
