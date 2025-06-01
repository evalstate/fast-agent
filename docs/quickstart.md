## Introduction

**Bee Agent** (`bee_agent`) is a Python framework for building AI “Queen” and “Worker” sub-agents that communicate via Redis Pub/Sub and use Model Context Protocol (MCP) servers (e.g., Brave Search, fetch). You can dynamically register multiple sub-agents, each with its own prompt instructions and tooling access.

---

## Quickstart Guide

1. **Start Redis**

   ```bash
   redis-server
   ```
2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install `bee_agent`**

   ```bash
   pip install bee_agent
   ```
4. **Set environment variables**

   ```text
   # In .env file at project root
   ANTHROPIC_API_KEY=<your_anthropic_api_key>
   PRIVATE_AGENT_KEY=<your_private_agent_key>
   ```
5. **Run the Queen Agent**

   ```bash
   python agent.py
   ```
6. **(Optional) Run the listener**

   ```bash
   python listener.py
   ```
7. **Test via Redis CLI**

   ```bash
   redis-cli PUBLISH agent:queen \
     '{"type":"user","content":"tell me price of polygon","channel_id":"agent:queen","metadata":{"model":"claude-3-5-haiku-latest"}}'
   ```

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install bee_agent
```

Ensure **Redis** is running on `localhost:6379`.

---

## Environment Variables

Create a file named `.env` in the project root:

```text
ANTHROPIC_API_KEY=<your_anthropic_api_key>
PRIVATE_AGENT_KEY=<your_private_agent_key>
```

These are required for any Anthropic-based models or internal agent authentication.

---
