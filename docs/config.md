## Configuration

### Sub-Agents

Define a list of sub-agent dicts in `agent.py`. Each entry needs:

* **name**: Unique identifier (e.g., `"finder"`).
* **instruction**: Prompt for that agent’s behavior.
* **servers**: MCP server names this agent can call (e.g., `["fetch","brave"]`).
* **model**: LLM model identifier (e.g., `"haiku"`).

Example:

```python
subagents_config = [
    {
        "name": "finder",
        "instruction": "Search latest crypto prices and report back.",
        "servers": ["fetch", "brave"],
        "model": "haiku"
    },
    {
        "name": "reporter",
        "instruction": "Summarize raw pricing data with key insights.",
        "servers": [],
        "model": "haiku"
    }
]
```

### MCP Settings

In `agent.py`, include a JSON config dict (here called `sample_json_config`) to specify:

* **mcp.servers**: Each server’s `name`, `transport`, `command`, `args`, and any `env` vars.
* **default\_model**: Fallback LLM (e.g., `"haiku"`).
* **logger**: `"level"` (e.g., `"info"`) and `"type"` (`"console"`).
* **pubsub\_enabled**: `True` to use Redis.
* **pubsub\_config.redis**:

  * `host`: e.g., `"localhost"`
  * `port`: `6379`
  * `db`: `0`
  * `channel_prefix`: `"agent:"`
* **anthropic.api\_key**: Loaded from `os.environ.get("CLAUDE_API_KEY", "")`.

A minimal example:

```python
sample_json_config = {
    "mcp": {
        "servers": {
            "fetch": {
                "name": "fetch",
                "transport": "stdio",
                "command": "uvx",
                "args": ["mcp-server-fetch"],
                "tool_calls": [{"name":"fetch","seek_confirm":True,"time_to_confirm":120000,"default":"reject"}]
            },
            "brave": {
                "name": "brave",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y","@modelcontextprotocol/server-brave-search"],
                "env": {"BRAVE_API_KEY":"<your_brave_key>"}
            }
        }
    },
    "default_model": "haiku",
    "logger": {"level":"info","type":"console"},
    "pubsub_enabled": True,
    "pubsub_config": {
        "use_redis": True,
        "channel_name": "queen",
        "redis": {"host":"localhost","port":6379,"db":0,"channel_prefix":"agent:"}
    },
    "anthropic": {"api_key": os.environ.get("CLAUDE_API_KEY","")}
}
```

---

## agent.py Overview

`agent.py` sets up a **Queen Agent** named `"queen"` and dynamically registers sub-agents defined in `subagents_config`. The main steps are:

1. **Load `.env`** (via `dotenv`).
2. **Instantiate `FastAgent(name="queen", json_config=sample_json_config)`**.
3. **Dynamically decorate each sub-agent** (looping over `subagents_config`) with `@fast.agent(name=..., instruction=..., servers=..., model=...)`.
4. **Define an orchestrator** with `@fast.orchestrator(name="orchestrate", agents=[...], plan_type="full", model="haiku")`.
5. **In `main()`**, create an async Redis client, subscribe to `"agent:queen"`, send an initial task to `agent.orchestrate(...)`, and then loop reading Redis messages. For each message:

   * Decode JSON; if it has `type == "user"`, forward `content` to `agent.orchestrate(user_input)`.
   * If JSON decoding fails, pass raw text to `agent.orchestrate(text)`.
6. **Run** with:

   ```bash
   python agent.py
   ```

**Key Points**

* The channel used is `"agent:queen"`.
* Sub-agent names in `@fast.orchestrator(agents=[...])` must match those registered via `@fast.agent(...)`.
* Redis Pub/Sub handles message delivery.

---
