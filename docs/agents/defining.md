# Defining Agents

## Basic Agents

Defining an agent is as simple as:

```python
@fast.agent(
  instruction="Given an object, respond only with an estimate of its size."
)
```

We can then send messages to the Agent:

```python
async with fast.run() as agent:
  moon_size = await agent("the moon")
  print(moon_size)
```

Or start an interactive chat with the Agent:

```python
async with fast.run() as agent:
  await agent.interactive()
```

Here is the complete `sizer.py` Agent application, with boilerplate code:

```python title="sizer.py"
import asyncio
from fast_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Agent Example")

@fast.agent(
  instruction="Given an object, respond only with an estimate of its size."
)
async def main():
  async with fast.run() as agent:
    await agent()

if __name__ == "__main__":
    asyncio.run(main())
```

The Agent can then be run with `uv run sizer.py`.

Specify a model with the `--model` switch - for example `uv run sizer.py --model sonnet`.

You can also pass a `Path` for the instruction - e.g. 

```python
from pathlib import Path

@fast.agent(
  instruction=Path("./sizing_prompt.md")
)

```

See [Workflows](workflows.md) for chaining, routing, parallelism, orchestrators, and MAKER.

## Human Input

Agents can request Human Input to assist with a task or get additional context:

```python
@fast.agent(
    instruction="An AI agent that assists with basic tasks. Request Human Input when needed.",
    human_input=True,
)

await agent("print the next number in the sequence")
```

In the example `human_input.py`, the agent will prompt the user for additional information to complete the task.

## Agent and Workflow Reference

### Calling Agents

All definitions allow omitting the name and instructions arguments for brevity:

```python
@fast.agent("You are a helpful agent")          # Create an agent with a default name.
@fast.agent("greeter","Respond cheerfully!")    # Create an agent with the name "greeter"

moon_size = await agent("the moon")             # Call the default (first defined agent) with a message

result = await agent.greeter("Good morning!")   # Send a message to an agent by name using dot notation
result = await agent.greeter.send("Hello!")     # You can call 'send' explicitly

agent["greeter"].send("Good Evening!")          # Dictionary access to agents is also supported
```

Read more about prompting agents [here](prompting.md)

## Configuring Agent Request Parameters

You can customize how an agent interacts with the LLM by passing `request_params=RequestParams(...)` when defining it.

### Example

```python
from fast_agent.types import RequestParams

@fast.agent(
  name="CustomAgent",                              # name of the agent
  instruction="You have my custom configurations", # base instruction for the agent
  request_params=RequestParams(
    maxTokens=8192,
    use_history=False,
    max_iterations=20
  )
)
```

### Available RequestParams Fields

| Field                 | Type     | Default | Description                                                                |
| --------------------- | -------- | ------- | -------------------------------------------------------------------------- |
| `maxTokens`           | `int`    | `2048`  | The maximum number of tokens to sample, as requested by the server         |
| `model`               | `string` | `None`  | The model to use for the LLM generation. Can only be set at Agent creation time                                    |
| `use_history`         | `bool`   | `True`  | Agent/LLM maintains conversation history. Does not include applied Prompts                        |
| `max_iterations`      | `int`    | `99`    | The maximum number of tool calls allowed in a conversation turn                        |
| `parallel_tool_calls` | `bool`   | `True`  | Whether to allow simultaneous tool calls   |
| `response_format`     | `Any`    | `None`  | Response format for structured calls (advanced use). Prefer to use `structured` with a Pydantic model instead                |
| `template_vars` | `Dict[str,Any]` | `{}` | Dictionary of template values for dynamic templates. Currently only supported for TensorZero provider |
| `mcp_metadata` | `Optional[Dict[str,Any]]` | `None` | Metadata to pass through to MCP tool calls via the _meta field |
| `temperature` | `float` | `None` | Temperature to use for the completion request |



### Defining Agents

#### Basic Agent

```python
@fast.agent(
  name="agent",                          # name of the agent
  instruction="You are a helpful Agent", # base instruction for the agent
  servers=["filesystem"],                # list of MCP Servers for the agent
  #tools={"filesystem": ["tool_1", "tool_2"]  # Filter the tools available to the agent. Defaults to all
  #resources={"filesystem: ["resource_1", "resource_2"]} # Filter the resources available to the agent. Defaults to all
  #prompts={"filesystem": ["prompt_1", "prompt_2"]}  # Filter the prompts available to the agent. Defaults to all.
  model="o3-mini.high",                  # specify a model for the agent
  use_history=True,                      # agent maintains chat history
  request_params=RequestParams(temperature= 0.7), # additional parameters for the LLM (or RequestParams())
  human_input=True,                      # agent can request human input
  elicitation_handler=ElicitationFnT,    # custom elicitation handler (from mcp.client.session)
  api_key="programmatic-api-key",        # specify the API KEY programmatically, it will override which provided in config file or env var
)
```

Workflow definitions (chain/parallel/router/orchestrator/maker) are documented on the [Workflows](workflows.md) page.

#### Custom

```python
@fast.custom(
  cls=Custom,                            # agent class
  name="custom",                         # name of the custom agent
  instruction="instruction",             # base instruction for the orchestrator
  servers=["filesystem"],                # list of MCP Servers for the agent
  #tools={"filesystem": ["tool_1", "tool_2"]  # Filter the tools available to the agent. Defaults to all
  #resources={"filesystem: ["resource_1", "resource_2"]} # Filter the resources available to the agent. Defaults to all
  #prompts={"filesystem": ["prompt_1", "prompt_2"]}  # Filter the prompts available to the agent. Defaults to all
  model="o3-mini.high",                  # specify a model for the agent
  use_history=True,                      # agent maintains chat history
  request_params=RequestParams(temperature= 0.7), # additional parameters for the LLM (or RequestParams())
  human_input=True,                      # agent can request human input
  elicitation_handler=ElicitationFnT,    # custom elicitation handler (from mcp.client.session)
  api_key="programmatic-api-key",        # specify the API KEY programmatically, it will override which provided in config file or env var
)
```
