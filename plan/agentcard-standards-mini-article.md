# AgentCard at the Summit: The Multi-Agent Standardization Revolution 
github.com/evalstate, github.com/iqdoctor 
Draft

LLM platforms have been climbing a clear ladder: plain text completions, chat completions, tool invocations, and then the MCP revolution. Each step widened what models could do, but also revealed a new bottleneck. Tools brought power, and MCP brought connectivity, yet we still struggle with large, brittle wrappers, heavy context costs, and inconsistent packaging. The next plateau is not just more tools. It is standardization: skills that can move, compose, and scale. AgentCard is a credible summit for that climb.

## The hidden cost of "just add tools"

MCP made it easy to plug in any tool server, but the price shows up immediately in context. Large tool schemas can consume tens of thousands of tokens before the agent does any real work. This is not just waste. It weakens continuity, degrades selection accuracy, and forces frequent context resets. The fast-agent "advanced tool use" proposal responds to the same pain Anthropic highlighted: stop loading everything up front and move to on-demand discovery and selective schema hydration.

That perspective reframes the problem: a tool surface should be thin and dynamic. A proxy or runtime can expose only a minimal meta-tool interface (discover, learn, execute), and load full schemas only when required. This reduces context bloat, improves correctness, and aligns with least-privilege configuration using AgentCard-style allowlists.

# AgentCard RFC 
RFC reference: <https://github.com/evalstate/fast-agent/blob/main/plan/agent-card-rfc.md>

This direction tracks real-world practice: many platforms and developers already store prompts in Markdown files, and many of those pair the prompt body with a metadata frontmatter to instantiate an agent. The RFC is based on the well-tested fast-agent.ai workflow system, and because of that it aligns with the full set of workflow types from Anthropic’s foundational piece, “Building Effective Agents” (<https://www.anthropic.com/engineering/building-effective-agents>), and incorporates the OpenAI “Agents as Tools” paradigm (<https://openai.github.io/openai-agents-python/tools/#agents-as-tools>), allowing user defined agents to call each other. That combination is why the spec aims not just for simplicity, but also for completeness. The spec also requires full support for MCP and python function tools. Tool filtering methods for saving context are also described.

## What the RFC nails down (concrete, not vibes)

AgentCard is a text-first format (`.md` or `.yaml`) that compiles into a single canonical IR: `AgentConfig`. The RFC defines a strict surface:

- One card per file by default; multi-card files are optional/experimental.
- Strict validation by `type`; unknown fields are rejected.
- Minimal attributes: `type`, `name`, `instruction`.
- `description` is optional and becomes the tool description when agents are exposed as tools.
- `instruction` is either the body or the `instruction` field (never both).
- `schema_version` is optional (int); defaults to 1.
- Runtime wiring fields are explicit: 
- `servers` select MCP endpoints by name, 
 - `tools` allowlist tools per server, 
 - `agents` declares child agents for routing/orchestration, and 
 - `messages` points to external history files.
- History preload formats are defined (JSON PromptMessageExtended, or delimited text/Markdown with role markers).
- Supported types: `agent`, `chain`, `parallel`, `evaluator_optimizer`, `router`, `orchestrator`, `iterative_planner`, `MAKER`, which define the base agent or specialized workflows.

AgentCard is not the same thing as a skill. The RFC draws the line clearly: a **Skill** is a reusable prompt fragment or capability description, while an **AgentCard** is a full runtime configuration (model, servers, tools, history source, instruction). That makes AgentCard the manifest layer above skills.

## Skills as portable expertise

The video "Don't Build Agents, Build Skills Instead" (https://www.youtube.com/watch?v=CEvIs9y1uog) frames the deeper issue: agents are smart but not expert. Intelligence without procedural knowledge leads to fragile outcomes. The proposed solution is simple and powerful: skills are just folders. A skill is a redistributable package of code, prompts, and documentation, with a progressive disclosure model where the agent reads only the small metadata until it needs deeper instructions.

This turns expertise into something tangible: versioned in Git, shared in a ZIP, and improved over time. It also flips the cost of context. Instead of flooding the model with everything, we keep a tiny surface in memory and pull in the rest only when a task truly needs it.

## Spec snapshots (from the RFC)

Basic Markdown card:

```md
---
type: agent
name: sizer
---
Given an object, respond only with an estimate of its size.
```

Agent with servers, tools, and child agents:

```md
---
type: agent
name: PMO-orchestrator
servers:
  - time
  - github
agents:
  - NY-Project-Manager
  - London-Project-Manager
tools:
  time: [get_time]
  github: [search_*]
---
Get reports. Always use one tool call per project/news.
Responsibilities: NY projects: [OpenAI, Fast-Agent, Anthropic].
London news: [Economics, Art, Culture].
Aggregate results and add a one-line PMO summary.
```

Agent with external history preload:

```md
---
type: agent
name: analyst
messages: ./history.md
---
You are a concise analyst.
```

## From SKILL.md to AgentCard

If SKILL.md is the minimal contract for a redistributable skill set (meta + prompt + code), then AgentCard is the next abstraction: meta + agent prompt + skills + workflows. It does not replace skills; it composes them. In that sense, AgentCard is to agent systems what a package manifest is to a codebase: a stable description of dependencies, behaviors, and integration points.

Think of the historical ladder:

- LLM completions -> chat completions -> tool invocations
- MCP as a connectivity standard
- SKILL.md as a portable skill container
- AgentCard as a higher-order manifest: prompts, skills, workflows, policies

Each step reduces friction in a different layer. SKILL.md makes expertise portable. AgentCard makes multi-agent systems interoperable.

## Why AgentCard looks like the summit

Standardization only works if the surface is small and composable. AgentCard aims for that. It can point to skills, define workflows, and express policies without re-encoding the world in a giant wrapper. That keeps implementations lean and lowers the cost of maintenance. It also aligns with the fast-agent view: keep tool surfaces small, discover capabilities on demand, and avoid pushing every schema into the context window.

Once you adopt a minimal AgentCard surface, a new distribution model emerges. Imagine an Agent Archive (AAR) that bundles:

- AgentCard metadata
- SKILL.md folders
- workflows and examples
- optional tests and evaluation scripts

This is the agent-era analog of a Java JAR: a single file that can be shared, versioned, and executed by any compatible runtime.

## A practical engine for the next phase

The fast-agent runtime illustrates how the engine can do the heavy lifting. It can filter tools by policy, defer schema loading, and proxy multiple MCP servers while exposing a tiny model-facing surface. That matters because it keeps the agent lightweight and the system scalable.

The net effect is the same pattern we already see in engineering: smaller interfaces, more reusable components, and better performance. In practice, function tools often cut both development time and context usage compared to a traditional MCP-only loop, especially when combined with skills and on-demand discovery.

## Conclusion

We are watching a standardization revolution unfold. LLMs brought general intelligence; MCP connected them to the world; skills turned experience into reusable code. AgentCard can be the summit that ties it all together: a minimal, portable interface that composes skills, workflows, and policies into a system others can run and extend.

If SKILL.md makes expertise portable, AgentCard makes multi-agent systems interoperable. That is not just a nice idea. It is the prerequisite for redistribution, reuse, and scale.

## References

- AgentCard RFC: <https://github.com/evalstate/fast-agent/blob/main/plan/agent-card-rfc.md>
- Anthropic: Building Effective Agents: <https://www.anthropic.com/engineering/building-effective-agents>
- OpenAI Agents SDK: Agents as Tools: <https://openai.github.io/openai-agents-python/tools/#agents-as-tools>
- Video: "Don't Build Agents, Build Skills Instead" <https://www.youtube.com/watch?v=CEvIs9y1uog>
