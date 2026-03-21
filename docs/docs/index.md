---
title: "fast-agent - MCP native Agents and Workflows"
hide:
    - navigation
    - toc
---

<center>

<h1>fast-agent</h1>

</center>

<div class="grid cards" markdown>
-   :material-clock-fast:{ .lg .middle } __Start now__

    ---

    Run `fast-agent` with the [uv](https://astral.sh/uv) package manager:

    ```
    uvx fast-agent-mcp@latest -x
    ```
    

    <!-- [:octicons-arrow-right-24: Get started](#get-started) -->


-   :material-battery-charging:{ .lg .middle } __Agent Skills__

    ---

     Use [Agent Skills](https://agentskills.io) to progressively disclose capabilities for your Agents. With registry update support. Check the documentation [here](./agents/skills.md).

-   :material-connection:{ .lg .middle } __New - Elicitation Quickstart Guide__

    ---

    Get started with MCP Elicitations for User Interaction. 
    [:octicons-arrow-right-24: Try now](./mcp/elicitations.md)

-   :material-beaker-check-outline:{ .lg .middle } __Validated Model Support__

    ---

    Advanced [support](./models/llm_providers.md) for models including OpenAI, Anthropic, Hugging Face, llama.cpp and more. Supports using Codex with OAuth.


-   :material-check-all:{ .lg .middle } __MCP Feature Support__

    ---

    Full MCP feature support including Elicitation and Sampling and advanced transport diagnostics

    [:octicons-arrow-right-24: Reference](mcp/index.md)

-   :material-application-braces-outline:{ .lg .middle } __Agent Developer Friendly__

    ---

    Lightweight deployment - in-built Echo and Playback LLMs allow robust agent application testing

</div>

<center>

<h2> Getting Started </h2>

</center>

<br />
<div class="grid" markdown>
<div align="top" markdown>


### Coding Agent

**fast-agent** makes supervising LLM and Tool streams simple, highlighting important results, actions and exceptions in long tool loops.

Fully extendable agent hook system and bundled skills to customise `fast-agent` with Python code. Fork, rewind and pin sessions.

Seamless shell integration, protecting your scrollback buffer for review, or use with ACP to use Agents with your editor of choice. 

### Agent Developers and Application Builders

**fast-agent** lets you create and interact with sophisticated Agents and Workflows in minutes. 

It's multi-modal - supporting Images and PDFs in Prompts, Resources and MCP Tool Call results.  

Prebuilt agents and examples implementing the patterns in Anthropic's [building effective agents](https://www.anthropic.com/engineering/building-effective-agents) paper get you building valuable applications quickly. Seamlessly use MCP Servers with your agents, or host your agents as MCP Servers.

### Evaluations and Automations

Comprehensive CLI and Model support with embedded telemetry in results files make `fast-agent` an excellent choice for testing and evaluations. 

### Quick Installation Guide

* `uv tool install fast-agent-mcp` - Install fast-agent. If you have multiple Python versions, use `--python 3.13.5` to ensure you get the latest version of fast-agent.
* `fast-agent go` - Start an interactive session...
* `fast-agent go --url https://hf.co/mcp` - ...with a remote MCP.
* `fast-agent go --pack hf-dev` - ...from an installed or marketplace card pack.
* `fast-agent scaffold` - Create Agent and Configuration files for Python development.

</div>
<div markdown>
<!--[Welcome Image](welcome_small.png)-->
<img src="welcome_small.png" style="padding: 0.5em;" />
</div>
</div>

