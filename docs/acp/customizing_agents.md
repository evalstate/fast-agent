# Customizing Agents (ACP)

When you run `fast-agent` via `fast-agent-acp` (or `--transport acp`), each configured agent appears to ACP clients as a **Mode**. “ACP-aware” agents can also expose slash commands and customize how their mode is presented to the client.

## Make an agent ACP-aware

Inherit from `ACPAwareMixin` in your agent class. The mixin gives you:

- `is_acp_mode`: whether the agent is currently running under ACP.
- `acp`: an `ACPContext` (mode switching, client capabilities, etc.) when in ACP mode.

## Add slash commands

ACP-aware agents can declare slash commands by overriding `acp_commands` and returning a dict of command name → `ACPCommand`.

Example outline:

```python
from fast_agent.acp import ACPAwareMixin, ACPCommand

class MyAgent(ACPAwareMixin, ...):
    @property
    def acp_commands(self) -> dict[str, ACPCommand]:
        return {
            "ping": ACPCommand(
                description="Health check",
                handler=self._ping,
                input_hint="[text]",
            )
        }

    async def _ping(self, args: str) -> str:
        return f"pong {args}".strip()
```

## Customize mode name and description

By default, `fast-agent-acp` infers ACP mode display info from the agent name and its instruction. If your agent is ACP-aware, you can supply the display metadata directly by overriding `acp_mode_info()`.

```python
from fast_agent.acp import ACPAwareMixin, ACPModeInfo

class MyAgent(ACPAwareMixin, ...):
    def acp_mode_info(self) -> ACPModeInfo | None:
        return ACPModeInfo(
            name="Docs Helper",
            description="Answers questions about this repo and drafts documentation.",
        )
```

Notes:

- The ACP **mode id** remains the agent’s configured name (the key in your agent map); this hook only affects the **display name/description** shown to clients.
- If you return `None` (the default), inference is used as a fallback.

