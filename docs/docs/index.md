---
title: "fast-agent - Code, Build and Evaluate Agents"
hide:
  - navigation
  - toc
  - edit
  - view
---

<section class="fa-hero fa-hero--home">
  <div>
  <p class="fa-kicker">Coding Agent and Development Toolkit</p> 
    <span class="fa-hero__brand" aria-label="fast-agent">
      <img class="fa-hero__wordmark fa-hero__wordmark--dark" src="assets/brand/fast-agent-anim-dark.svg" alt="fast-agent">
      <img class="fa-hero__wordmark fa-hero__wordmark--light" src="assets/brand/fast-agent-anim-light.svg" alt="fast-agent">
    </span>
    <p class="fa-lede">
      Simple extendable agents. Excellent provider and local model support. Flexible context management. Terminal native and scriptable.
    </p>
    <div class="fa-hero__actions">
      <a class="fa-btn fa-btn--primary" href="#try-it-now">Try it now</a>
      <a class="fa-btn" href="agents/defining/">Build an agent</a>
      <a class="fa-btn" href="ref/go_command/">Explore the CLI</a>
    </div>
  </div>
  <div class="fa-term" aria-label="fast-agent terminal quickstart">
    <div class="fa-term__bar">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      <strong>~/projects/fast-agent</strong>
    </div>
    <pre><code><span class="fa-muted">$</span> uvx fast-agent-mcp@latest -x
<span class="fa-good">ready</span>  interactive agent with shell tools

<span class="fa-muted">$</span> fast-agent go --url https://hf.co/mcp --model kimi
<span class="fa-good">connected</span>  Hugging Face MCP server

<span class="fa-muted">$</span> fast-agent go --pack analyst --model haiku
<span class="fa-good">loaded</span>  reusable agent card pack</code></pre>
  </div>
</section>

<section id="try-it-now" class="fa-band fa-band--start">
  <div>
    <h2>Get started now</h2>
  </div>
  <pre><code>uvx fast-agent-mcp@latest -x</code></pre>
</section>

<section class="fa-grid fa-grid--4">
  <article class="fa-card">
    <h3>Test model behavior quickly</h3>
    <p>
      Switch among Anthropic, OpenAI, Responses, Google, Hugging Face, Groq, xAI, Bedrock, and local
      OpenAI-compatible endpoints. Built-in playback and passthrough models make app tests practical.
    </p>
    <a href="models/">Model features</a>
  </article>
  <article class="fa-card">
    <h3>Use MCP without ceremony</h3>
    <p>
      Attach local or remote MCP servers from config or the command line. Use Elicitations,
      Sampling, Resources, Prompts, OAuth, UI resources, and transport diagnostics from one client.
    </p>
    <a href="mcp/">MCP guide</a>
  </article>
  <article class="fa-card">
    <h3>Compose workflow patterns</h3>
    <p>
      Chain agents, run parallel checks, route work, orchestrate subtasks, evaluate outputs, or use
      MAKER to turn classifier examples into reliable structured decisions.
    </p>
    <a href="agents/workflows/">Workflow docs</a>
  </article>
  <article class="fa-card">
    <h3>Ship beyond the terminal</h3>
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
    <a class="fa-step" href="ref/go_command/"><h3>Probe with `fast-agent go`</h3></a>
    <a class="fa-step" href="agents/skills/"><h3>Add focused Agent Skills</h3></a>
    <a class="fa-step" href="agents/workflows/"><h3>Compose workflows</h3></a>
    <a class="fa-step" href="mcp/mcp-server/"><h3>Serve as MCP</h3></a>
  </div>
      <div class="fa-band" style="margin-top: var(--sp-7);">
      <div>
        <h2 style="font-size: var(--t-h2);">Next: the theme overlay</h2>
        <p>I'll build the MkDocs <code>overrides/</code> theme overlay next — header replacement, custom admonition mappings, code-block restyling, page templates. Plus a worked demo doc page rendered through it.</p>
      </div>
      <div>
        <a class="fa-btn fa-btn--primary fa-btn--lg" href="fast-agent.ai integration.html">See integration v1 →</a>
      </div>

</section>

<!-- 

<section class="fa-proof">
  <h2>Why people try fast-agent first</h2>
  <div class="fa-grid fa-grid--4">
    <p class="fa-card"><strong>Simple commands.</strong> `uvx`, `fast-agent go`, and card packs make first contact fast.</p>
    <p class="fa-card"><strong>MCP depth.</strong> The docs and tests cover advanced MCP behavior, not only tool calls.</p>
    <p class="fa-card"><strong>Reusable definitions.</strong> Agent cards, workflows, skills, and config files stay editable.</p>
    <p class="fa-card"><strong>Real deployment paths.</strong> Use the same work from CLI, Python, MCP, and ACP surfaces.</p>
  </div>
</section>

-->
