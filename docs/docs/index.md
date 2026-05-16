---
title: "fast-agent - MCP native agents and workflows"
hide:
  - navigation
  - toc
---

<section class="fa-hero">
  <div class="fa-hero__copy">
    <p class="fa-kicker">MCP-native agents, workflows, and servers</p>
    <h1><span class="fa-prompt">❯</span> fast-agent</h1>
    <p class="fa-lede">
      Build useful agent applications from the terminal, connect them to MCP servers, test them
      across models, and ship the same work as CLI, MCP, or ACP experiences.
    </p>
    <div class="fa-actions">
      <a class="fa-button fa-button--primary" href="#try-it-now">Try it now</a>
      <a class="fa-button" href="agents/defining/">Build an agent</a>
      <a class="fa-button" href="ref/go_command/">Explore the CLI</a>
    </div>
  </div>
  <div class="fa-terminal" aria-label="fast-agent terminal quickstart">
    <div class="fa-terminal__bar">
      <span></span><span></span><span></span>
      <strong>terminal</strong>
    </div>
    <pre><code><span class="fa-muted">$</span> uvx fast-agent-mcp@latest -x
<span class="fa-good">ready</span>  interactive agent with shell tools

<span class="fa-muted">$</span> fast-agent go --url https://hf.co/mcp --model kimi
<span class="fa-good">connected</span>  Hugging Face MCP server

<span class="fa-muted">$</span> fast-agent go --pack analyst --model haiku
<span class="fa-good">loaded</span>  reusable agent card pack</code></pre>
  </div>
</section>

<section id="try-it-now" class="fa-band">
  <div>
    <p class="fa-kicker">Zero setup path</p>
    <h2>Start from `uvx`, graduate to real applications</h2>
  </div>
  <div class="fa-command-row">
    <code>uvx fast-agent-mcp@latest -x</code>
    <code>uvx fast-agent-mcp@latest --pack hf-dev</code>
    <code>uvx fast-agent-mcp@latest --pack codex</code>
  </div>
</section>

<section class="fa-feature-grid">
  <article>
    <h2>Use MCP without ceremony</h2>
    <p>
      Attach local or remote MCP servers from config or the command line. Use Elicitations,
      Sampling, Resources, Prompts, OAuth, UI resources, and transport diagnostics from one client.
    </p>
    <a href="mcp/">MCP guide</a>
  </article>
  <article>
    <h2>Compose workflow patterns</h2>
    <p>
      Chain agents, run parallel checks, route work, orchestrate subtasks, evaluate outputs, or use
      MAKER to turn classifier examples into reliable structured decisions.
    </p>
    <a href="agents/workflows/">Workflow docs</a>
  </article>
  <article>
    <h2>Test model behavior quickly</h2>
    <p>
      Switch among Anthropic, OpenAI, Responses, Google, Hugging Face, Groq, xAI, Bedrock, and local
      OpenAI-compatible endpoints. Built-in playback and passthrough models make app tests practical.
    </p>
    <a href="models/">Model features</a>
  </article>
  <article>
    <h2>Ship beyond the terminal</h2>
    <p>
      Run the same agent definitions interactively, expose them as MCP servers, or connect them to
      ACP clients with slash commands, file access, and permission-aware tools.
    </p>
    <a href="acp/">ACP docs</a>
  </article>
</section>

<section class="fa-split">
  <div>
    <p class="fa-kicker">Developer loop</p>
    <h2>Keep the agent close while you build</h2>
    <p>
      `fast-agent go` lets you inspect an MCP server, try a model, load a card pack, compare
      prompts, and iterate before writing application code. When the shape is right, scaffold a
      project or move the same configuration into a Python workflow.
    </p>
  </div>
  <div class="fa-steps">
    <a href="ref/go_command/"><span>1</span> Probe with `fast-agent go`</a>
    <a href="agents/skills/"><span>2</span> Add focused Agent Skills</a>
    <a href="agents/workflows/"><span>3</span> Compose workflows</a>
    <a href="mcp/mcp-server/"><span>4</span> Serve as MCP</a>
  </div>
</section>

<section class="fa-proof">
  <h2>Why people try fast-agent first</h2>
  <div class="fa-proof__grid">
    <p><strong>Simple commands.</strong> `uvx`, `fast-agent go`, and card packs make first contact fast.</p>
    <p><strong>MCP depth.</strong> The docs and tests cover advanced MCP behavior, not only tool calls.</p>
    <p><strong>Reusable definitions.</strong> Agent cards, workflows, skills, and config files stay editable.</p>
    <p><strong>Real deployment paths.</strong> Use the same work from CLI, Python, MCP, and ACP surfaces.</p>
  </div>
</section>

