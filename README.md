<p align="center">
<a href="https://pypi.org/project/fast-agent-mcp/"><img src="https://img.shields.io/pypi/v/fast-agent-mcp?color=%2334D058&label=pypi" /></a>
<a href="#"><img src="https://github.com/evalstate/fast-agent/actions/workflows/main-checks.yml/badge.svg" /></a>
<a href="https://github.com/evalstate/fast-agent/issues"><img src="https://img.shields.io/github/issues-raw/evalstate/fast-agent" /></a>
<a href="https://discord.gg/xg5cJ7ndN6"><img src="https://img.shields.io/discord/1358470293990936787" alt="discord" /></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/fast-agent-mcp?label=pypi%20%7C%20downloads"/>
<a href="https://github.com/evalstate/fast-agent-mcp/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/fast-agent-mcp" /></a>
</p>

## Overview

> [!TIP]
> Documentation site is in production here : https://fast-agent.ai. Feel free to feed back what's helpful and what's not. There is also an LLMs.txt [here](https://fast-agent.ai/llms.txt)

**`fast-agent`** enables you to create and interact with sophisticated Agents and Workflows in minutes. It is the first framework with complete, end-to-end tested MCP Feature support including Sampling. Both Anthropic (Haiku, Sonnet, Opus) and OpenAI models (gpt-4o/gpt-4.1 family, o1/o3 family) are supported.

The simple declarative syntax lets you concentrate on composing your Prompts and MCP Servers to [build effective agents](https://www.anthropic.com/research/building-effective-agents).

`fast-agent` is multi-modal, supporting Images and PDFs for both Anthropic and OpenAI endpoints via Prompts, Resources and MCP Tool Call results. The inclusion of passthrough and playback LLMs enable rapid development and test of Python glue-code for your applications.

> [!IMPORTANT]
>
> `fast-agent` The fast-agent documentation repo is here: https://github.com/evalstate/fast-agent-docs. Please feel free to submit PRs for documentation, experience reports or other content you think others may find helpful. All help and feedback warmly received.

### Agent Application Development

Prompts and configurations that define your Agent Applications are stored in simple files, with minimal boilerplate, enabling simple management and version control.

Chat with individual Agents and Components before, during and after workflow execution to tune and diagnose your application. Agents can request human input to get additional context for task completion.

Simple model selection makes testing Model <-> MCP Server interaction painless. You can read more about the motivation behind this project [here](https://llmindset.co.uk/resources/fast-agent/)

![2025-03-23-fast-agent](https://github.com/user-attachments/assets/8f6dbb69-43e3-4633-8e12-5572e9614728)

## Get started:

Start by installing the [uv package manager](https://docs.astral.sh/uv/) for Python. Then:

```bash
uv pip install fast-agent-mcp          # install fast-agent!
fast-agent go                          # start an interactive session
fast-agent go https://hf.co/mcp        # with a remote MCP
fast-agent go --model=generic.qwen2.5  # use ollama qwen 2.5
fast-agent setup                       # create an example agent and config files
uv run agent.py                        # run your first agent
uv run agent.py --model=o3-mini.low    # specify a model
fast-agent quickstart workflow  # create "building effective agents" examples
```

Other quickstart examples include a Researcher Agent (with Evaluator-Optimizer workflow) and Data Analysis Agent (similar to the ChatGPT experience), demonstrating MCP Roots support.

> [!TIP]
> Windows Users - there are a couple of configuration changes needed for the Filesystem and Docker MCP Servers - necessary changes are detailed within the configuration files.

### Basic Agents

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

```python
import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Agent Example")

@fast.agent(
  instruction="Given an object, respond only with an estimate of its size."
)
async def main():
  async with fast.run() as agent:
    await agent.interactive()

if __name__ == "__main__":
    asyncio.run(main())
```

The Agent can then be run with `uv run sizer.py`.

Specify a model with the `--model` switch - for example `uv run sizer.py --model sonnet`.

### Combining Agents and using MCP Servers

_To generate examples use `fast-agent quickstart workflow`. This example can be run with `uv run workflow/chaining.py`. fast-agent looks for configuration files in the current directory before checking parent directories recursively._

Agents can be chained to build a workflow, using MCP Servers defined in the `fastagent.config.yaml` file:

```python
@fast.agent(
    "url_fetcher",
    "Given a URL, provide a complete and comprehensive summary",
    servers=["fetch"], # Name of an MCP Server defined in fastagent.config.yaml
)
@fast.agent(
    "social_media",
    """
    Write a 280 character social media post for any given text.
    Respond only with the post, never use hashtags.
    """,
)
@fast.chain(
    name="post_writer",
    sequence=["url_fetcher", "social_media"],
)
async def main():
    async with fast.run() as agent:
        # using chain workflow
        await agent.post_writer("http://llmindset.co.uk")
```

All Agents and Workflows respond to `.send("message")` or `.prompt()` to begin a chat session.

Saved as `social.py` we can now run this workflow from the command line with:

```bash
uv run workflow/chaining.py --agent post_writer --message "<url>"
```

Add the `--quiet` switch to disable progress and message display and return only the final response - useful for simple automations.

## Workflows

### Chain

The `chain` workflow offers a more declarative approach to calling Agents in sequence:

```python

@fast.chain(
  "post_writer",
   sequence=["url_fetcher","social_media"]
)

# we can them prompt it directly:
async with fast.run() as agent:
  await agent.post_writer()

```

This starts an interactive session, which produces a short social media post for a given URL. If a _chain_ is prompted it returns to a chat with last Agent in the chain. You can switch the agent to prompt by typing `@agent-name`.

Chains can be incorporated in other workflows, or contain other workflow elements (including other Chains). You can set an `instruction` to precisely describe it's capabilities to other workflow steps if needed.

### Human Input

Agents can request Human Input to assist with a task or get additional context:

```python
@fast.agent(
    instruction="An AI agent that assists with basic tasks. Request Human Input when needed.",
    human_input=True,
)

await agent("print the next number in the sequence")
```

In the example `human_input.py`, the Agent will prompt the User for additional information to complete the task.

### Parallel

The Parallel Workflow sends the same message to multiple Agents simultaneously (`fan-out`), then uses the `fan-in` Agent to process the combined content.

```python
@fast.agent("translate_fr", "Translate the text to French")
@fast.agent("translate_de", "Translate the text to German")
@fast.agent("translate_es", "Translate the text to Spanish")

@fast.parallel(
  name="translate",
  fan_out=["translate_fr","translate_de","translate_es"]
)

@fast.chain(
  "post_writer",
   sequence=["url_fetcher","social_media","translate"]
)
```

If you don't specify a `fan-in` agent, the `parallel` returns the combined Agent results verbatim.

`parallel` is also useful to ensemble ideas from different LLMs.

When using `parallel` in other workflows, specify an `instruction` to describe its operation.

### Evaluator-Optimizer

Evaluator-Optimizers combine 2 agents: one to generate content (the `generator`), and the other to judge that content and provide actionable feedback (the `evaluator`). Messages are sent to the generator first, then the pair run in a loop until either the evaluator is satisfied with the quality, or the maximum number of refinements is reached. The final result from the Generator is returned.

If the Generator has `use_history` off, the previous iteration is returned when asking for improvements - otherwise conversational context is used.

```python
@fast.evaluator_optimizer(
  name="researcher",
  generator="web_searcher",
  evaluator="quality_assurance",
  min_rating="EXCELLENT",
  max_refinements=3
)

async with fast.run() as agent:
  await agent.researcher.send("produce a report on how to make the perfect espresso")
```

When used in a workflow, it returns the last `generator` message as the result.

See the `evaluator.py` workflow example, or `fast-agent quickstart researcher` for a more complete example.

### Router

Routers use an LLM to assess a message, and route it to the most appropriate Agent. The routing prompt is automatically generated based on the Agent instructions and available Servers.

```python
@fast.router(
  name="route",
  agents=["agent1","agent2","agent3"]
)
```

Look at the `router.py` workflow for an example.

### Orchestrator

Given a complex task, the Orchestrator uses an LLM to generate a plan to divide the task amongst the available Agents. The planning and aggregation prompts are generated by the Orchestrator, which benefits from using more capable models. Plans can either be built once at the beginning (`plantype="full"`) or iteratively (`plantype="iterative"`).

```python
@fast.orchestrator(
  name="orchestrate",
  agents=["task1","task2","task3"]
)
```

See the `orchestrator.py` or `agent_build.py` workflow example.

## Agent Features

### Calling Agents

All definitions allow omitting the name and instructions arguments for brevity:

```python
@fast.agent("You are a helpful agent")          # Create an agent with a default name.
@fast.agent("greeter","Respond cheerfully!")    # Create an agent with the name "greeter"

moon_size = await agent("the moon")             # Call the default (first defined agent) with a message

result = await agent.greeter("Good morning!")   # Send a message to an agent by name using dot notation
result = await agent.greeter.send("Hello!")     # You can call 'send' explicitly

await agent.greeter()                           # If no message is specified, a chat session will open
await agent.greeter.prompt()                    # that can be made more explicit
await agent.greeter.prompt(default_prompt="OK") # and supports setting a default prompt

agent["greeter"].send("Good Evening!")          # Dictionary access is supported if preferred
```

### Defining Agents

#### Basic Agent

```python
@fast.agent(
  name="agent",                          # name of the agent
  instruction="You are a helpful Agent", # base instruction for the agent
  servers=["filesystem"],                # list of MCP Servers for the agent
  model="o3-mini.high",                  # specify a model for the agent
  use_history=True,                      # agent maintains chat history
  request_params=RequestParams(temperature= 0.7), # additional parameters for the LLM (or RequestParams())
  human_input=True,                      # agent can request human input
)
```

#### Chain

```python
@fast.chain(
  name="chain",                          # name of the chain
  sequence=["agent1", "agent2", ...],    # list of agents in execution order
  instruction="instruction",             # instruction to describe the chain for other workflows
  cumulative=False,                      # whether to accumulate messages through the chain
  continue_with_final=True,              # open chat with agent at end of chain after prompting
)
```

#### Parallel

```python
@fast.parallel(
  name="parallel",                       # name of the parallel workflow
  fan_out=["agent1", "agent2"],          # list of agents to run in parallel
  fan_in="aggregator",                   # name of agent that combines results (optional)
  instruction="instruction",             # instruction to describe the parallel for other workflows
  include_request=True,                  # include original request in fan-in message
)
```

#### Evaluator-Optimizer

```python
@fast.evaluator_optimizer(
  name="researcher",                     # name of the workflow
  generator="web_searcher",              # name of the content generator agent
  evaluator="quality_assurance",         # name of the evaluator agent
  min_rating="GOOD",                     # minimum acceptable quality (EXCELLENT, GOOD, FAIR, POOR)
  max_refinements=3,                     # maximum number of refinement iterations
)
```

#### Router

```python
@fast.router(
  name="route",                          # name of the router
  agents=["agent1", "agent2", "agent3"], # list of agent names router can delegate to
  model="o3-mini.high",                  # specify routing model
  use_history=False,                     # router maintains conversation history
  human_input=False,                     # whether router can request human input
)
```

#### Orchestrator

```python
@fast.orchestrator(
  name="orchestrator",                   # name of the orchestrator
  instruction="instruction",             # base instruction for the orchestrator
  agents=["agent1", "agent2"],           # list of agent names this orchestrator can use
  model="o3-mini.high",                  # specify orchestrator planning model
  use_history=False,                     # orchestrator doesn't maintain chat history (no effect).
  human_input=False,                     # whether orchestrator can request human input
  plan_type="full",                      # planning approach: "full" or "iterative"
  plan_iterations=5,                     # maximum number of full plan attempts, or iterations
)
```

### Multimodal Support

Add Resources to prompts using either the inbuilt `prompt-server` or MCP Types directly. Convenience class are made available to do so simply, for example:

```python
  summary: str =  await agent.with_resource(
      "Summarise this PDF please",
      "mcp_server",
      "resource://fast-agent/sample.pdf",
  )
```

#### MCP Tool Result Conversion

LLM APIs have restrictions on the content types that can be returned as Tool Calls/Function results via their Chat Completions API's:

- OpenAI supports Text
- Anthropic supports Text and Image

For MCP Tool Results, `ImageResources` and `EmbeddedResources` are converted to User Messages and added to the conversation.

### Prompts

MCP Prompts are supported with `apply_prompt(name,arguments)`, which always returns an Assistant Message. If the last message from the MCP Server is a 'User' message, it is sent to the LLM for processing. Prompts applied to the Agent's Context are retained - meaning that with `use_history=False`, Agents can act as finely tuned responders.

Prompts can also be applied interactively through the interactive interface by using the `/prompt` command.

### Sampling

Sampling LLMs are configured per Client/Server pair. Specify the model name in fastagent.config.yaml as follows:

```yaml
mcp:
  servers:
    sampling_resource:
      command: "uv"
      args: ["run", "sampling_resource_server.py"]
      sampling:
        model: "haiku"
```

### Secrets File

> [!TIP]
> fast-agent will look recursively for a fastagent.secrets.yaml file, so you only need to manage this at the root folder of your agent definitions.

### Interactive Shell

![fast-agent](https://github.com/user-attachments/assets/3e692103-bf97-489a-b519-2d0fee036369)

## Project Notes

`fast-agent` builds on the [`mcp-agent`](https://github.com/lastmile-ai/mcp-agent) project by Sarmad Qadri.

### Contributing

Contributions and PRs are welcome - feel free to raise issues to discuss. Full guidelines for contributing and roadmap coming very soon. Get in touch!
