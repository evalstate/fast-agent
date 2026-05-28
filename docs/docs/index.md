---
title: fast-agent - Code, Build and Evaluate Agents
hide:
- navigation
- toc
- edit
- view
social:
  title: fast-agent
  tagline: Simple, extendable agents.
  description: MCP-native agents, workflows, and servers.
  alt: fast-agent — MCP-native agents, workflows, and servers
  variant: hero
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
      <a class="fa-btn fa-btn--primary" href="guides/codex/">Try it now</a>
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
<span class="fa-good">start</span> an interactive session with shell tools

<span class="fa-muted">$</span>uv tool install -U fast-agent-mcp
<span class="fa-good">install</span> the latest version of fast-agent

<span class="fa-muted">$</span> fast-agent --pack codex 
<span class="fa-good">code</span> download configuration to use codex</code></pre>
  </div>
</section>

<!--

<section id="try-it-now" class="fa-band fa-band--start">
  <div>
    <h2>Get started now</h2>
  </div>
  <div class="fa-term" aria-label="fast-agent terminal quickstart">
    <div class="fa-term__bar">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      <strong>~/projects/fast-agent</strong>
    </div>
    <pre><code><span class="fa-muted">$</span> uvx fast-agent-mcp@latest -x
<span class="fa-good">ready</span> Start an interactive agent with shell tools

<span class="fa-muted">$</span> fast-agent --url https://hf.co/mcp --auth $HF_TOKEN
<span class="fa-good">connected</span> Quick connect to an MCP Server

<span class="fa-muted">$</span> fast-agent go --pack analyst --model haiku
<span class="fa-good">install</span>  reusable agent card pack</code></pre>
  </div>  
  <pre><code>uvx fast-agent-mcp@latest -x</code></pre>
</section>

-->

<section class="fa-grid fa-grid--4">
  <article class="fa-card">
    <h3>Extensive Model Support</h3>
    <p>
      Native provider support for Anthropic, Google and OpenAI compatible endpoints. <br />Auto configuration for <b>llama.cpp</b> hosted models. <br />
    </p>
    <a href="models/">Model features</a>
  </article>
  <article class="fa-card">
    <h3>MCP and ACP</h3>
    <p>
      Attach MCP servers from config or command line. Deploy Agents via ACP or MCP. Comprehensive support and transport diagnostics. 
    </p>
    <a href="mcp/">ACP guide</a> <br />
    <a href="mcp/">MCP guide</a> 
  </article>
  <article class="fa-card">
    <h3>Plugin and Extend</h3>
    <p>
       Write plugins and hooks with simple Python, or use directly from the API. Distribute and update configurations with Card Packs. 
    </p>
    <a href="agents/plugins/">Plugin docs</a>
  </article>
  <article class="fa-card">
    <h3>Control your context</h3>
    <p>
      Template based <a href="agents/instructions">System Prompts</a>. Install and update Agent Skills, prompt files or Agent definitions.
    </p>
    <a href="agents/skills/">Agent Skills</a>
  </article>
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
