# AgentCard RFC Samples (MD)

Note: samples mirror README.md code blocks in order. Each block is a full markdown file. Some samples group multiple cards for brevity; in practice, each card lives in its own file (one card per file by default). Non-card snippets (CLI/config/usage) are included as plain markdown file content.

## Sample 1: Quickstart commands
```md
uv pip install fast-agent-mcp          # install fast-agent!
fast-agent go                          # start an interactive session
fast-agent go --url https://hf.co/mcp  # with a remote MCP
fast-agent go --model=generic.qwen2.5  # use ollama qwen 2.5
fast-agent setup                       # create an example agent and config files
uv run agent.py                        # run your first agent
uv run agent.py --model=o3-mini.low    # specify a model
uv run agent.py --transport http --port 8001  # expose as MCP server (server mode implied)
fast-agent quickstart workflow  # create "building effective agents" examples
```

## Sample 2: Basic agent definition
```md
---
type: agent
name: sizer
instruction: "Given an object, respond only with an estimate of its size."
---
```

## Sample 3: Send a message to the agent
```md
---
type: agent
name: sizer
instruction: "Given an object, respond only with an estimate of its size."
messages: ./history.md
---
```

## Sample 4: Interactive chat (no preloaded messages)
```md
---
type: agent
name: sizer
instruction: "Given an object, respond only with an estimate of its size."
---
```

## Sample 5: Full sizer app (single agent file)
```md
---
type: agent
name: sizer
default: true
instruction: "Given an object, respond only with an estimate of its size."
---
```

## Sample 6: Function tools and hooks
```md
---
type: agent
name: assistant
function_tools:
  - tools.py:add_one
tool_hooks:
  - tools.py:audit_hook
instruction: "Use local tools when needed and return concise results."
---
```

## Sample 7: Combining agents and a chain
Card: url_fetcher
```md
---
type: agent
name: url_fetcher
servers:
  - fetch
instruction: "Given a URL, provide a complete and comprehensive summary."
---
```
Card: social_media
```md
---
type: agent
name: social_media
---
Write a 280 character social media post for any given text.
Respond only with the post, never use hashtags.
```
Card: post_writer
```md
---
type: chain
name: post_writer
sequence:
  - url_fetcher
  - social_media
instruction: "Generate a short social media post from a URL summary."
---
```

## Sample 8: Run a chain from the CLI
```md
uv run workflow/chaining.py --agent post_writer --message "<url>"
```

## Sample 9: MAKER workflow
Card: classifier
```md
---
type: agent
name: classifier
instruction: "Reply with only: A, B, or C."
---
```
Card: reliable_classifier
```md
---
type: MAKER
name: reliable_classifier
worker: classifier
k: 3
max_samples: 25
match_strategy: normalized
red_flag_max_length: 16
instruction: "Repeat the worker and return the k-vote winner."
---
```

## Sample 10: Agents as tools (orchestrator-workers)
Card: NY-Project-Manager
```md
---
type: agent
name: NY-Project-Manager
servers:
  - time
tools:
  time: [get_time]
instruction: "Return NY time + timezone, plus a one-line project status."
---
```
Card: London-Project-Manager
```md
---
type: agent
name: London-Project-Manager
servers:
  - time
tools:
  time: [get_time]
instruction: "Return London time + timezone, plus a one-line news update."
---
```
Card: PMO-orchestrator
```md
---
type: agent
name: PMO-orchestrator
default: true
agents:
  - NY-Project-Manager
  - London-Project-Manager
instruction: "Get reports. Always use one tool call per project/news. Responsibilities: NY projects: [OpenAI, Fast-Agent, Anthropic]. London news: [Economics, Art, Culture]. Aggregate results and add a one-line PMO summary."
---
```

## Sample 11: MCP OAuth minimal config
```md
mcp:
  servers:
    myserver:
      transport: http # or sse
      url: http://localhost:8001/mcp # or /sse for SSE servers
      auth:
        oauth: true # default: true
        redirect_port: 3030 # default: 3030
        redirect_path: /callback # default: /callback
        # scope: "user"       # optional; if omitted, server defaults are used
```

## Sample 12: MCP OAuth in-memory tokens
```md
mcp:
  servers:
    myserver:
      transport: http
      url: http://localhost:8001/mcp
      auth:
        oauth: true
        persist: memory
```

## Sample 13: Chain workflow (minimal)
```md
---
type: chain
name: post_writer
sequence:
  - url_fetcher
  - social_media
instruction: "Generate a short social post from a URL summary."
---
```

## Sample 14: Human input agent
```md
---
type: agent
name: assistant
human_input: true
instruction: "An AI agent that assists with basic tasks. Request Human Input when needed."
---
```

## Sample 15: Parallel workflow + chain
Card: translate_fr
```md
---
type: agent
name: translate_fr
instruction: "Translate the text to French."
---
```
Card: translate_de
```md
---
type: agent
name: translate_de
instruction: "Translate the text to German."
---
```
Card: translate_es
```md
---
type: agent
name: translate_es
instruction: "Translate the text to Spanish."
---
```
Card: translate (parallel)
```md
---
type: parallel
name: translate
fan_out:
  - translate_fr
  - translate_de
  - translate_es
instruction: "Translate input text to multiple languages and return the combined results."
---
```
Card: post_writer (chain)
```md
---
type: chain
name: post_writer
sequence:
  - url_fetcher
  - social_media
  - translate
instruction: "Generate a post and return translated variants."
---
```

## Sample 16: Evaluator-optimizer workflow
```md
---
type: evaluator_optimizer
name: researcher
generator: web_searcher
evaluator: quality_assurance
min_rating: EXCELLENT
max_refinements: 3
instruction: "Iterate until the evaluator approves the research output."
---
```

## Sample 17: Router
```md
---
type: router
name: route
agents:
  - agent1
  - agent2
  - agent3
instruction: "Route requests to the most appropriate agent."
---
```

## Sample 18: Orchestrator
```md
---
type: orchestrator
name: orchestrate
agents:
  - task1
  - task2
  - task3
instruction: "Plan work across agents and aggregate the results."
---
```

## Sample 19: Calling agents (usage patterns)
Card: default
```md
---
type: agent
name: default
default: true
instruction: "You are a helpful agent."
---
```
Card: greeter
```md
---
type: agent
name: greeter
instruction: "Respond cheerfully!"
---
```
Usage:
```python
moon_size = await agent("the moon")
result = await agent.greeter("Good morning!")
result = await agent.greeter.send("Hello!")
await agent.greeter()
await agent.greeter.prompt()
await agent.greeter.prompt(default_prompt="OK")
agent["greeter"].send("Good Evening!")
```

## Sample 20: Basic agent definition (full params)
```md
---
type: agent
name: agent
servers:
  - filesystem
model: o3-mini.high
use_history: true
request_params:
  temperature: 0.7
human_input: true
instruction: "You are a helpful Agent."
---
```

## Sample 21: Chain definition (full params)
```md
---
type: chain
name: chain
sequence:
  - agent1
  - agent2
cumulative: false
continue_with_final: true
instruction: "instruction"
---
```

## Sample 22: Parallel definition (full params)
```md
---
type: parallel
name: parallel
fan_out:
  - agent1
  - agent2
fan_in: aggregator
include_request: true
instruction: "instruction"
---
```

## Sample 23: Evaluator-optimizer definition (full params)
```md
---
type: evaluator_optimizer
name: researcher
generator: web_searcher
evaluator: quality_assurance
min_rating: GOOD
max_refinements: 3
instruction: "Refine outputs until quality meets the threshold."
---
```

## Sample 24: Router definition (full params)
```md
---
type: router
name: route
agents:
  - agent1
  - agent2
  - agent3
model: o3-mini.high
use_history: false
human_input: false
instruction: "Route requests based on agent capabilities."
---
```

## Sample 25: Orchestrator definition (full params)
```md
---
type: orchestrator
name: orchestrator
agents:
  - agent1
  - agent2
model: o3-mini.high
use_history: false
human_input: false
plan_type: full
plan_iterations: 5
instruction: "instruction"
---
```

## Sample 26: MAKER definition (full params)
```md
---
type: MAKER
name: maker
worker: worker_agent
k: 3
max_samples: 50
match_strategy: exact
red_flag_max_length: 256
instruction: "instruction"
---
```

## Sample 27: Agents as tools (full params)
```md
---
type: agent
name: orchestrator
agents:
  - agent1
  - agent2
max_parallel: 128
child_timeout_sec: 600
max_display_instances: 20
instruction: "instruction"
---
```

## Sample 28: with_resource usage
```python
summary: str = await agent.with_resource(
    "Summarise this PDF please",
    "mcp_server",
    "resource://fast-agent/sample.pdf",
)
```

## Sample 29: Sampling config
```md
mcp:
  servers:
    sampling_resource:
      command: "uv"
      args: ["run", "sampling_resource_server.py"]
      sampling:
        model: "haiku"
```

## Sample 30: Name defaults to filename (single card)
```md
---
type: agent
instruction: "Respond cheerfully!"
---
```
