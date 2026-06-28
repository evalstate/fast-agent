---
title: Defining Agents
social:
  title: Defining Agents
  tagline: Define agents with Python decorators or portable AgentCards.
  description: Define agents with Python decorators or portable AgentCards.
  alt: fast-agent social card — Defining Agents
---

# Defining Agents

Agents represent a bundle of Tools, Hooks, Skills, Instructions and Configuration for interacting with models.

**fast-agent** supports two complementary ways to define agents:

- [Python API](python_api/) - define agents and workflows in Python with decorators such as `@fast.agent` or `@fast.parallel`. 
- [Agent Cards](agent_cards/) - portable markdown files that can be loaded by `fast-agent go`, the TUI, and card tooling.

Use the Python API when you are want to build a self contained Python application or script. Use Agent Cards when you want portable, reusable agent definitions that can be easily loaded, edited or [shared](../../guides/core-concepts/#share-it-as-a-pack).

Agents can be used in a number of ways:

 - From the [Harness API](harness-api.md): a high level session based automation framework.
 - Directly from a Python program using `fast_agent.run()`
 - Interactively, from the **`fast-agent`** [TUI](../../guides/tui.md) or [ACP](../../acp/)
 - Deployed as an MCP Server

