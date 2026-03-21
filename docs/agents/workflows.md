# Workflows

Workflows let you compose multiple agents into a single higher-level capability (e.g. chaining steps, routing, or adding reliability via voting). They can be used alongside MCP servers defined in `fastagent.config.yaml`.

## Workflows and MCP Servers

To generate examples use `fast-agent quickstart workflow`.

Agents can use MCP Servers defined in `fastagent.config.yaml`:

```yaml title="fastagent.config.yaml"
# Example of a STDIO sever named "fetch"
mcp:
  servers:
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]
```

```python title="social.py"
@fast.agent(
    "url_fetcher",
    "Given a URL, provide a complete and comprehensive summary",
    servers=["fetch"],  # Name of an MCP Server defined in fastagent.config.yaml
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
        await agent.post_writer.send("http://fast-agent.ai")
```

Saved as `social.py` you can run the workflow from the command line with:

```bash
uv run social.py --agent post_writer --message "<url>"
```

Add the `--quiet` switch to disable progress and message display and return only the final response.

Read more about running **fast-agent** agents [here](running.md)

## Workflow Types

**fast-agent** has built-in support for common agentic workflow patterns (including those referenced in Anthropic's [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)).

### Chain

The `chain` workflow offers a declarative approach to calling Agents in sequence.

```python
@fast.chain(
  name="post_writer",
  sequence=["url_fetcher", "social_media"],
)

async with fast.run() as agent:
  await agent.interactive(agent="post_writer")
```

Chains can be incorporated in other workflows, or contain other workflow elements (including other Chains). You can set an `instruction` to describe its capabilities to other workflow steps if needed.

### Parallel

The `parallel` workflow sends the same message to multiple agents simultaneously (`fan_out`), then optionally uses a `fan_in` agent to process the combined content.

```python
@fast.agent("translate_fr", "Translate the text to French")
@fast.agent("translate_de", "Translate the text to German")
@fast.agent("translate_es", "Translate the text to Spanish")

@fast.parallel(
  name="translate",
  fan_out=["translate_fr", "translate_de", "translate_es"],
)
```

If you don't specify a `fan_in` agent, `parallel` returns the combined agent results verbatim.

### Evaluator-Optimizer

Evaluator-Optimizers combine 2 agents: one to generate content (the `generator`), and the other to judge that content and provide actionable feedback (the `evaluator`). Messages are sent to the generator first, then the pair run in a loop until either the evaluator is satisfied with the quality, or the maximum number of refinements is reached. The final result from the generator is returned.

```python
@fast.evaluator_optimizer(
  name="researcher",
  generator="web_searcher",
  evaluator="quality_assurance",
  min_rating="EXCELLENT",
  max_refinements=3,
)

async with fast.run() as agent:
  await agent.researcher.send("produce a report on how to make the perfect espresso")
```

### Router

Routers use an LLM to assess a message and route it to the most appropriate agent. The routing prompt is automatically generated based on the agent instructions and available servers.

```python
@fast.router(
  name="route",
  agents=["agent1", "agent2", "agent3"],
)
```

NB - If only one agent is supplied to the router, it forwards directly.

### Orchestrator

Given a complex task, the Orchestrator uses an LLM to generate a plan to divide the task amongst the available Agents. Plans can either be built once at the beginning (`plan_type="full"`) or iteratively (`plan_type="iterative"`).

```python
@fast.orchestrator(
  name="orchestrate",
  agents=["task1", "task2", "task3"],
)
```

### Iterative Planner

The `iterative_planner` workflow is a specialized orchestrator for long-running plans that are refined over multiple iterations.

```python
@fast.iterative_planner(
  name="planner",
  agents=["task1", "task2", "task3"],
)
```

### MAKER

MAKER (“Massively decomposed Agentic processes with K-voting Error Reduction”) wraps a worker agent and samples it repeatedly until a response achieves a k-vote margin over all alternatives (“first-to-ahead-by-k” voting). This is useful for long chains of simple steps where rare errors would otherwise compound.

- Reference: [Solving a Million-Step LLM Task with Zero Errors](https://arxiv.org/abs/2511.09030)
- Credit: Lucid Programmer (PR author)

```python
@fast.agent(
  name="classifier",
  instruction="Reply with only: A, B, or C.",
)
@fast.maker(
  name="reliable_classifier",
  worker="classifier",
  k=3,
  max_samples=25,
  match_strategy="normalized",
  red_flag_max_length=16,
)
async def main():
  async with fast.run() as agent:
    await agent.reliable_classifier.send("Classify: ...")
```

### Agents As Tools

The Agents As Tools workflow takes a complex task, breaks it into subtasks, and calls other agents as tools based on the main agent instruction.

This pattern is inspired by the OpenAI Agents SDK [Agents as tools](https://openai.github.io/openai-agents-python/tools/#agents-as-tools) feature.

With child agents exposed as tools, you can implement routing, parallelization, and orchestrator-workers [decomposition](https://www.anthropic.com/engineering/building-effective-agents) directly in the instruction (and combine them). Multiple tool calls per turn are supported and executed in parallel.

Common usage patterns may combine:

- Routing: choose the right specialist tool(s) based on the user prompt.
- Parallelization: fan out over independent items/projects, then aggregate.
- Orchestrator-workers: break a task into scoped subtasks (often via a simple JSON plan), then coordinate execution.

```python
@fast.agent(
    name="NY-Project-Manager",
    instruction="Return NY time + timezone, plus a one-line project status.",
    servers=["time"],
)
@fast.agent(
    name="London-Project-Manager",
    instruction="Return London time + timezone, plus a one-line news update.",
    servers=["time"],
)
@fast.agent(
    name="PMO-orchestrator",
    instruction=(
        "Get reports. Always use one tool call per project/news. "
        "Responsibilities: NY projects: [OpenAI, Fast-Agent, Anthropic]. London news: [Economics, Art, Culture]. "
        "Aggregate results and add a one-line PMO summary."
    ),
    default=True,
    agents=["NY-Project-Manager", "London-Project-Manager"],
)
async def main() -> None:
    async with fast.run() as agent:
        await agent("Get PMO report. Projects: all. News: Art, Culture")
```

## Workflow Reference

### Chain

```python
@fast.chain(
  name="chain",
  sequence=["agent1", "agent2", ...],
  instruction="instruction",
  cumulative=False,
)
```

### Parallel

```python
@fast.parallel(
  name="parallel",
  fan_out=["agent1", "agent2"],
  fan_in="aggregator",
  instruction="instruction",
  include_request=True,
)
```

### Evaluator-Optimizer

```python
@fast.evaluator_optimizer(
  name="researcher",
  generator="web_searcher",
  evaluator="quality_assurance",
  instruction="instruction",
  min_rating="GOOD",
  max_refinements=3,
  refinement_instruction="optional guidance",
)
```

### Router

```python
@fast.router(
  name="route",
  agents=["agent1", "agent2", "agent3"],
  instruction="routing instruction",
  servers=["filesystem"],
  model="o3-mini.high",
  use_history=False,
  human_input=False,
  api_key="programmatic-api-key",
)
```

### Orchestrator

```python
@fast.orchestrator(
  name="orchestrator",
  instruction="instruction",
  agents=["agent1", "agent2"],
  model="o3-mini.high",
  use_history=False,
  human_input=False,
  plan_type="full",
  plan_iterations=5,
  api_key="programmatic-api-key",
)
```

### Iterative Planner

```python
@fast.iterative_planner(
  name="planner",
  agents=["agent1", "agent2"],
  model="o3-mini.high",
  plan_iterations=-1,
  api_key="programmatic-api-key",
)
```

### MAKER

```python
@fast.maker(
  name="maker",
  worker="worker_agent",
  k=3,
  max_samples=50,
  match_strategy="exact",  # exact|normalized|structured
  red_flag_max_length=256,
  instruction="instruction",
)
```

### Agents As Tools

```python
@fast.agent(
  name="orchestrator",
  instruction="instruction",
  agents=["agent1", "agent2"],  # exposed as tools: agent__agent1, agent__agent2
  history_mode="fork",          # scratch|fork|fork_and_merge
  max_parallel=128, # OpenAI limitation
  child_timeout_sec=600,
  max_display_instances=20,
)
```
