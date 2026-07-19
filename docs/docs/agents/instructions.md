---
social:
  title: System Prompts
  tagline: Shape agent behavior with reusable instructions and prompt files.
  description: Shape agent behavior with reusable instructions and prompt files.
  alt: fast-agent social card — System Prompts
---

# System Prompts

Agents can have their System Instructions set and customised in a number of flexible ways. The default System Prompt caters for Agent Skills, MCP Server Instructions, `AGENTS.md` and Shell access.

## Template Variables

The following variables are available in System Prompt templates:

| Variable | Description |  Notes |
|----------|-------------|-------|
| <nobr>`{{internal:resource_id}}`</nobr> | Loads packaged internal markdown resources | Examples: `{{internal:smart_prompt}}`, `{{internal:smart_agent_cards}}` |
| <nobr>`{{file:path}}`</nobr> | Reads and embeds local file contents (errors if file missing) |  **Must be a relative path** (resolved relative to `workspaceRoot`) |
| <nobr>`{{file_silent:path}}`</nobr> | Reads and embeds local file contents (empty if file missing) |  **Must be a relative path** (resolved relative to `workspaceRoot`) |
| <nobr>`{{url:https://...}}`</nobr> | Fetches and embeds content from an HTTP(S) URL |
| <nobr>`{{url:hf://...}}`</nobr> | Fetches and embeds text content from Hugging Face Hub |
| <nobr>`{{serverInstructions}}`</nobr> | MCP server instructions with available tools |  Warning displayed in `/mcp` if Instructions are present and template variable missing |
| <nobr>`{{agentSkills}}`</nobr> | Agent skill manifests with descriptions |  |
| <nobr>`{{workspaceRoot}}`</nobr> | Current working directory / workspace root | Set by Client in ACP Mode |
| <nobr>`{{agentName}}`</nobr> | Current agent name |  |
| <nobr>`{{agentType}}`</nobr> | Current agent type |  |
| <nobr>`{{agentCardPath}}`</nobr> | Source AgentCard path | `(internal)` when not loaded from a card |
| <nobr>`{{agentCardDir}}`</nobr> | Directory containing the source AgentCard | `(internal)` when not loaded from a card |
| <nobr>`{{modelReferences}}`</nobr> | Model references | Includes the effective `$system.default` Useful for instructing CLI based subagents |
| <nobr>`{{model_specific}}`</nobr> | Model-specific prompt guidance from the resolved model catalog entry or model overlay | Empty when the selected model has no model-specific guidance |
| <nobr>`{{hostPlatform}}`</nobr> | Host platform information |  |
| <nobr>`{{pythonVer}}`</nobr> | Python version |  |
| <nobr>`{{env}}`</nobr> | Formatted environment block with all environment details |  |
| <nobr>`{{currentDate}}`</nobr> | Current date in long format |  |

**Example `{{env}}` output:**
```
Environment:
- Workspace root: /home/user/project
- Client: fast-agent (pid 12345)
- Host platform: Linux-6.6.87.2-microsoft-standard-WSL2
```

**Note on file templates:** File paths in `{{file:...}}` and `{{file_silent:...}}` must be relative paths. They will be resolved relative to the `workspaceRoot` at runtime. Absolute paths are not allowed and will raise an error.

**Viewing the System Prompt** The System Prompt can be inspected with the `/system` command from `fast-agent` or the `/status system` Slash Command in ACP Mode.

The standard default System Prompt used with `fast-agent go` or `fast-agent-acp` (without `--smart`) is:

```markdown title="Default System Prompt"
You are a helpful AI Agent.

{{serverInstructions}}
{{agentSkills}}
{{file_silent:AGENTS.md}}
{{env}}
{{model_specific}}

The current date is {{currentDate}}.
```

When `--smart` is enabled, fast-agent uses the internal smart prompt resource:

```markdown title="Smart System Prompt Selector"
{{internal:smart_prompt}}
```

You can also include only the AgentCard guidance section inside your own instruction template:

```markdown title="Custom Prompt Including AgentCard Guidance"
You are a safety-focused assistant.
{{internal:smart_agent_cards}}
Always confirm before destructive operations.
```


## Using Instructions

When defining an Agent, you can load the instruction as either a `String`, `Path` or `AnyUrl`.

Instructions support embedding the current date, as well as content from other URLs and `hf://` URIs. This is really helpful if you want to refer to files on GitHub, assemble useful prompts/content in Gists, or reuse prompt assets stored in Hugging Face Hub.

```python title="Simple String"
@fast.agent(name="example",
    instruction="""
You are a helpful AI Agent.
""")
```

```python title="With current date"
@fast.agent(name="example",
    instruction="""
You are a helpful AI Agent.
Your reliable knowledge cut-off date is December 2024.
Today's date is {{currentDate}}.
""")
```

Will produce: `You are a helpful AI Agent. Your reliable knowledge cut-off date is January 2025. Today's date is 28 May 2026.`

```python title="With URL"
@fast.agent(name="mcp-expert",
    instruction="""
You have expert knowledge of the
MCP (Model Context Protocol) schema.

{{url:https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/refs/heads/main/schema/2025-11-25/schema.ts}}

Answer any questions about the protocol by referring
to and quoting the schema where necessary.
""")
```

```python title="With Hugging Face Hub content"
@fast.agent(name="hf-prompt",
    instruction="""
Use the following shared guidance:

{{url:hf://buckets/evalstate/home/demo.md}}
""")
```

You can store the prompt in an external file for easy editing - including template variables:

```python title="From file"
from pathlib import Path

@fast.agent(name="mcp-expert",
    instruction=Path("./mcp-expert.md"))
async def main():
    pass
```

```md title="mcp-expert.md"
You have expert knowledge of the MCP (Model Context Protocol) schema.

{{url:https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/refs/heads/main/schema/2025-11-25/schema.ts}}

Answer any questions about the protocol by referring to and quoting the schema where necessary.
Your knowledge cut-off is December 2024; today's date is {{currentDate}}.

```

Or you can load the prompt directly from an HTTP(S) URL or `hf://` URI:

```python title="From URL"
from pydantic import AnyUrl

@fast.agent(name="mcp-expert",
    instruction=AnyUrl("https://gist.githubusercontent.com/evalstate/d432921aaaee2c305cf46ae320840360/raw/eb9c7ff93adc780171bfb0ae2560be2178304f16/gistfile1.txt"))

# --> fast-agent system prompt demo
```

You can start an agent with instructions from a file using the `fast-agent` command:

```bash
fast-agent --instruction mcp-expert.md
```

This can be combined with other options to specify model and available servers:

```bash
fast-agent --instruction mcp-expert.md --model sonnet --url https://huggingface.co/mcp
```

Starts an interactive agent session, with the MCP Schema loaded, attached to Sonnet with the Hugging Face MCP Server.

![Instructions](instructions.png)

You can even specify multiple models to directly compare their outputs:

![Instructions Parallel](instructions_parallel.png)

Read more about the `fast-agent` command [here](../ref/go_command/).
