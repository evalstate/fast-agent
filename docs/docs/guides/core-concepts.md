---
title: Core Concepts
description: Understand fast-agent homes, AgentCards, packs, model references, MCP servers, skills, and the TUI workflow.
social:
  title: Core Concepts Guide
  tagline: Homes, execution environments, agents, packs, models, MCP, and skills in one place.
  description: Understand fast-agent homes, AgentCards, packs, model references, MCP servers, skills, and the TUI workflow.
  alt: fast-agent social card — Core Concepts Guide
---

# Core Concepts

**`fast-agent`** scales from a single markdown file, to a fully managed multi-agent environment. There's a comprehensive CLI and Harness API for automation, and a TUI for interactive usage.

You can use **`fast-agent`** with zero configuration: simply use `fast-agent go` to start an interactive session, use local or remote models, add skills or connect to MCP Servers.

Most **`fast-agent`** usage is centred on a "home":  a configuration folder that declares agents, skills, MCP Servers, hooks, plugins and more.

The table below introduces the main configuration concepts:

| Concept | What it means | Default / example |
| --- | --- | --- |
| **Home** | fast-agent's local storage and configuration: config files, AgentCards, skills, sessions, plugins, model overlays, UI assets, and permission history. | `<workspace>/.fast-agent`, or `--home ~/agent-homes/client-a` |
| **Workspace** | The project or request file tree the run is working from. Relative paths, prompt file references, and local shell cwd behavior are anchored here. | Current directory, or `--workspace ~/client-a` |
| **Execution environment** | The runtime used by tools or agents to execute commands and access files. Environments can be defined in your home config and resolved by name. | `local`, Docker or Hugging Face sandbox environment |


<div class="fa-hero__actions">
  <a class="fa-btn fa-btn--primary" href="#make-your-first-home">Build one</a>
  <a class="fa-btn" href="#share-it-as-a-pack">Share as a pack</a>
  <a class="fa-btn" href="../agents/defining/agent_cards/">AgentCard reference</a>
  <a class="fa-btn" href="../agents/environments.md">Define environments</a>
</div>

## What is a home?

The active fast-agent home stores local config and runtime state. By default it
is `./.fast-agent` in your current workspace.
`--home` names this directory itself, not its parent. Relative `--home` paths
resolve under the selected workspace; use an absolute path for a separate home.

```text
.fast-agent/
├── fast-agent.yaml          # providers, model defaults, MCP servers, registries
├── fast-agent.secrets.yaml  # optional local secrets, same shape as config
├── agent-cards/             # runnable agents loaded by default
├── tool-cards/              # agents attached as tools to another agent
├── card-packs/              # installed packs and provenance metadata
├── plugins/                 # command plugins and hook support files
├── skills/                  # Agent Skills available through {{agentSkills}}
├── sessions/                # persisted chat/session history
├── model-overlays/          # optional local model definitions
├── ui/                      # generated MCP UI assets
└── auths.md                 # home-scoped permission/auth history
```

The defaults are deliberately useful:

- `fast-agent go` selects `./.fast-agent` unless you choose another home (`--home`).
- The configuration file (`fast-agent.yaml`) is loaded from the active fast-agent home first, then the current
  directory if no home config is present.
- AgentCards in `<home>/agent-cards/` are loaded as runnable agents.
- ToolCards in `<home>/tool-cards/` are loaded and attached as tools.
- Sessions are saved in `<home>/sessions/` so you can resume work later.
- Skills are discovered from the active fast-agent home's `skills/` directory
  (normally `.fast-agent/skills`), plus `.agents/skills` and `.claude/skills`.

<br />

<section class="fa-grid fa-grid--3" markdown="1">
<article class="fa-card" markdown="1">
<h3>Agents</h3>

Optionally defined in markdown AgentCard files. The first card marked `default: true` is used as the default for messages. If none are present, a simple default is provided.
</article>

<article class="fa-card" markdown="1">
<h3>Config</h3>

`fast-agent.yaml` holds provider settings, Environment definitions, MCP server config, model
references, registry URLs, logging and session settings.
</article>

<article class="fa-card" markdown="1">
<h3>Skills</h3>

Reusable capabilities installed under `skills/` and managed interactively with
`/skills`.
</article>
</section>

<br/>

If no AgentCards are present, `fast-agent go` still starts a simple default
agent from the command-line options you provide.

The built-in default prompt is already practical: it includes `AGENTS.md` from
the current project when that file exists, so project conventions are picked up
without making an AgentCard first. If you want a more capable generated default,
run:

```bash
fast-agent go --smart
```

`--smart` asks **fast-agent** to use a _smart_ default agent. A smart agent has extra guidance for working with fast-agent concepts, including creating and delegating to sub-agents.

Home directories can be distributed as "packs" - simplifying sharing, installation and version management.


## Make your first home

Create a project home with one default coding agent and two supporting
agents:

```bash
mkdir -p .fast-agent/agent-cards
```

```md title=".fast-agent/agent-cards/dev.md"
---
name: dev
type: smart
default: true
model: $system.default
shell: true
---

You help with software development. Be direct, make small changes, and explain
trade-offs clearly.

{{agentSkills}}
{{file_silent:AGENTS.md}}
{{env}}
```

```md title=".fast-agent/agent-cards/planner.md"
---
name: planner
model: $system.default
---

You turn ambiguous software work into a short, testable plan. Prefer milestones,
risks, and explicit assumptions over long prose.
```

```md title=".fast-agent/agent-cards/reviewer.md"
---
name: reviewer
model: $system.default
---

You review plans and code for correctness, maintainability, missing tests, and
unnecessary complexity. Be concise and specific.
```

Run it:

```bash
fast-agent go --model sonnet
```

Because each card uses `model: $system.default`, the selected model comes from
`--model`, then the home config, then normal provider defaults. If no default is found an interactive model picker is displayed.

## Work with multiple agents in the TUI

Inside the interactive prompt, agents are lightweight to move between:

<div class="fa-term" aria-label="agent switching and targeted messages">
  <div class="fa-term__bar">
    <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    <strong>interactive</strong>
  </div>
  <pre><code><span class="fa-muted">dev&gt;</span> @planner
<span class="fa-good">switched</span> planner

<span class="fa-muted">planner&gt;</span> sketch a migration plan for the auth module

<span class="fa-muted">planner&gt;</span> #reviewer find risks in this plan
<span class="fa-good">response</span> copied into your input buffer for editing

<span class="fa-muted">planner&gt;</span> ##reviewer check quietly and draft comments
<span class="fa-good">quiet</span> response loaded without streaming display</code></pre>
</div>

- `@agent_name` switches the active conversation.
- `#agent_name message` asks another agent and loads its response into your
  input buffer. There must be no space after `#`; `# Heading` remains normal
  Markdown text.
- `##agent_name message` is the quiet form. It suppresses interactive display and
  still loads the response into your buffer.
- `/agent`, `/card`, `/reload`, `/history`, `/session`, `/connect`, and
  `/skills` are available while you work.

<div class="fa-term" aria-label="fast-agent TUI example">
  <div class="fa-term__bar">
    <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    <strong>.fast-agent</strong>
  </div>
  <pre><code><span class="fa-muted">$</span> fast-agent go --pack codex
<span class="fa-good">loaded</span> dev, planner, reviewer

<span class="fa-muted">fast-agent&gt;</span> @planner
<span class="fa-good">switched</span> planner is now the active agent

<span class="fa-muted">planner&gt;</span> #reviewer check this plan for risky assumptions
<span class="fa-good">loaded</span> reviewer response into your input buffer</code></pre>
</div>


## Add MCP servers to an agent

MCP Servers can be connected at any time with the `/connect` command. This supports both remote servers with a URL as well as npx, uvx or other STDIO servers.

MCP servers can also be configured once in `fast-agent.yaml`, then referenced by
agents.

```yaml title=".fast-agent/fast-agent.yaml"
default_model: gpt-5.4-mini?reasoning=low

mcp:
  servers:
    fetch:
      command: uvx
      args: [mcp-server-fetch]
    filesystem:
      command: npx
      args:
        - -y
        - "@modelcontextprotocol/server-filesystem"
        - .
```

```md title=".fast-agent/agent-cards/researcher.md"
---
name: researcher
servers: [fetch, filesystem]
model: $system.default
---

You research technical topics and cite sources. Use the filesystem for local
notes and fetch for public pages.
```

You can also define servers directly on an AgentCard:

```md
---
name: papers
mcp_connect:
  - target: https://papers.example.com/mcp
    name: papers
    headers:
      Authorization: Bearer ${PAPERS_TOKEN}
---

You find and summarize relevant papers.
```

Use configured servers for shared infrastructure. Use `mcp_connect` when the
server is part of the agent definition itself.

## Add small tools and sub-agents

When an AgentCard is attached as a tool, fast-agent uses the card's
`description` as the tool description. This is the short hint the parent agent
sees when deciding whether to call that sub-agent.

```md
---
name: reviewer
description: Review a proposed plan or patch for risks, missed tests, and unnecessary complexity.
tool_only: true
model: $system.default
---

You are a concise software reviewer. Focus on correctness, maintainability and
test coverage.
```

For a plain Python function, add it to an agent with `function_tools`:

```python title="tools.py"
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

```md
---
name: calculator
function_tools:
  - tools.py:add
---

You can do small calculations.
```

Hooks and command plugins are the next step when you want to change lifecycle
behaviour or add reusable commands. Use the fast-agent skill for guided setup,
or see [Command Plugins](../agents/plugins/).

## Model references and `--model`

A **model string** can be a provider model, a shorthand alias, a model overlay,
or a model reference.

```bash
fast-agent go --model sonnet
fast-agent go --model 'responses.gpt-5?web_search=on'
fast-agent go --model 'xai.grok-4.3?x_search=on'
```

Model references are exact tokens like `$system.fast`. Define them in the
home config and reuse them in cards:

```yaml
default_model: $system.fast

model_references:
  system:
    fast: gpt-5-mini?reasoning=low
    plan: claude-sonnet-4-6
    research: responses.gpt-5?web_search=on
```

```md
---
name: planner
model: $system.plan
---

You create concise plans.
```

`$system.default` is a special reference for reusable cards. It means "use the
current run's default model". That makes packs easy to share: the card author
sets `model: $system.default`, and the user chooses the model with `--model` or
`default_model` in their home.

Explicit card models usually win over `--model`. Use `$system.default` when you
want `--model` to remain in control.

Use `fast-agent model setup` to see and set configured references.

## Multiple Homes

Use more than one home when you want different bundles for different
workflows:

```bash
# Coding home
fast-agent go --home .fast-agent-coding --pack codex 

# Research home
fast-agent go --home .fast-agent-research --agent researcher \
  --model 'responses.gpt-5?web_search=on'

# Ephemeral run: no implicit home cards, sessions, pack installs or permission-store side effects
fast-agent go --no-home --model haiku --message "summarize this directory"
```

Selection order for the home is:

1. `--home <path>`
2. `--workspace <path>/.fast-agent`
3. `FAST_AGENT_HOME`
4. `./.fast-agent`

You can also set `home` in `fast-agent.yaml`, or override skills for
one run:

```bash
fast-agent go --workspace ~/client-a --skills ~/agent-skills/client-a
```

`--no-home` is useful for clean tests, one-off MCP inspection, or automation that
should not read project AgentCards or write session state. It cannot be combined
with `--pack`, `--home`, or `--resume`.

## Skills are home tools for knowledge

Skills are folders containing a `SKILL.md` manifest plus optional scripts,
references and assets. They let agents load specialized procedures only when
needed. By default, fast-agent looks in:

- the active fast-agent home's `skills/` directory — normally `.fast-agent/skills`
- `.agents/skills`
- `.claude/skills`

The easiest way to get started is to install skills from an existing registry
from inside the TUI. fast-agent includes public registries for fast-agent,
Hugging Face and Anthropic skills, and teams can add their own.

Common workflow:

```text
/skills            # list available skills
/skills add        # browse and install from the active registry
/skills remove 1   # remove by number or name
/skills registry   # view or switch registries
```

`/skills add` presents the available skills as a numbered list, so installing
one is usually just:

```text
/skills add
/skills add 1
```

If an agent or sub-agent should **not** see the default skills, make that
explicit in its AgentCard:

```yaml
---
name: reviewer
skills: []
---
```

You can also point a card at specific skill folders instead of the defaults:

```yaml
---
name: release-manager
skills:
  - .fast-agent/skills/release
  - ~/team-skills/compliance
---
```

Environment config can pin skill directories and registries:

```yaml
skills:
  directories:
    - .fast-agent/skills
    - ~/team/agent-skills
  marketplace_urls:
    - https://github.com/fast-agent-ai/skills
    - https://github.com/my-org/private-skills
```

See [Agent Skills](skills/) for the full skill workflow.

## Share it as a pack

A **card pack** is a publishable bundle of AgentCards, ToolCards and supporting
files. Packs are how teams distribute a good home without asking everyone
to copy files by hand.

Install and run one immediately:

```bash
fast-agent go --pack codex --model sonnet
```

Manage packs explicitly:

```bash
fast-agent cards add codex
fast-agent cards update codex
fast-agent cards readme codex
fast-agent cards publish codex --message "Improve reviewer prompt"
```

Packs are installed into the selected home under `card-packs/`, and their
managed files are copied into places like `agent-cards/`, `tool-cards/`,
`plugins/`, or the home root.

A minimal pack looks like this:

```text
packs/coding/
├── card-pack.yaml
├── README.md
└── agent-cards/
    ├── dev.md
    ├── planner.md
    └── reviewer.md
```

```yaml
# packs/coding/card-pack.yaml
schema_version: 2
name: coding
kind: card
install:
  agent_cards:
    - agent-cards/dev.md
    - agent-cards/planner.md
    - agent-cards/reviewer.md
  files:
    - README.md
model_references_recommended:
  - $system.fast
plugins:
  recommended:
    - agent-finder
```

A registry is a marketplace file that points at packs:

```json
{
  "entries": [
    {
      "name": "coding",
      "description": "A dev agent with planning and review helpers",
      "kind": "card",
      "repo": "https://github.com/my-org/agent-packs",
      "ref": "main",
      "path": "packs/coding"
    }
  ]
}
```

Use the default registry, a URL, or a local registry while developing:

```bash
fast-agent cards add coding
fast-agent cards --registry https://github.com/my-org/agent-packs add coding
fast-agent go --pack coding --pack-registry ./marketplace.json
```

GitHub repository URLs are accepted as registries when they contain a
`marketplace.json` file. During development, a local JSON file or local GitHub
checkout is often enough:

```json
{
  "entries": [
    {
      "name": "coding-local",
      "description": "Local pack while iterating",
      "repo_url": ".",
      "repo_path": "packs/coding"
    }
  ]
}
```

```bash
fast-agent go --pack coding-local --pack-registry ./marketplace.json
```

## Two common bundles

<section class="fa-grid fa-grid--2" markdown="1">
<article class="fa-card" markdown="1">
<h3>Coding</h3>

- `dev` is the default smart agent.
- `planner` turns issues into implementation plans.
- `reviewer` checks patches and test strategy.
- MCP servers provide filesystem, shell, docs search, or project-specific tools.
- Skills encode repository conventions, release procedures, and debugging playbooks.
</article>

<article class="fa-card" markdown="1">
<h3>Research</h3>

- `researcher` uses web search and fetch tools.
- `librarian` stores source notes in the filesystem.
- `critic` checks claims and asks for citations.
- Model strings can enable provider tools, for example `responses.gpt-5?web_search=on`
  or `xai.grok-4.3?x_search=on`.
- MCP servers can connect to internal papers, notebooks, or databases.
</article>
</section>


<div class="fa-band" markdown="1">
<div markdown="1">
<h2>Where to go next</h2>

A home is just a folder. Start with one default card, add focused agents
as Markdown files, then publish the bundle when it becomes useful.
</div>
<div markdown="1">
<a class="fa-btn fa-btn--primary" href="../agents/defining/agent_cards/">Read AgentCards</a>
</div>
</div>

- [Agent Cards](../agents/defining/agent_cards/) — advanced card fields,
  ToolCards, runtime MCP targets and loading rules.
- [Environments](../agents/environments.md) — Set up local and remote containers and sandboxes.
- [Configuration Reference](../ref/config_file/) — every `fast-agent.yaml`
  setting, including providers, MCP, sessions, skills and model references.
- [fast-agent go](../ref/go_command/) — all CLI switches for homes,
  packs, models, cards, skills and non-interactive runs.
- [MCP configuration](../mcp/client-servers/) — configure and inspect MCP servers.
- [Model Features](../models/) — model strings, provider web tools and overlays.
- [fast-agent skills](https://github.com/fast-agent-ai/skills) - Skills to work with Agent Cards, Hooks, Plugins, Automations and more.
