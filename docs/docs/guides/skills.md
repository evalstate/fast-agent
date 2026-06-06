---
title: Agent Skills
social:
  title: Agent Skills
  tagline: Package reusable capabilities and domain knowledge for agents to load on
    demand.
  description: Package reusable capabilities and domain knowledge for agents to load
    on demand.
  alt: fast-agent social card — Agent Skills
---

**`fast-agent`** supports Agent Skills, and adds them from the following directories by default:

- `.fast-agent/skills`
- `.agents/skills`
- `.claude/skills`

When valid `SKILL.md` files are found:

- The Agent is given shell access with the working directory set to the workspace root.
- Skill descriptions from the manifest and path are added to the System Prompt using the `{{agentSkills}}` template variable. A warning is displayed if this is not present in the System Prompt.
- The `/skills` command lists the available skills.
- If duplicate skill names exist across directories, later directories override earlier ones, and a warning message is shown. (Use `/status` from an ACP Client to view).

!!! tip "Skill Management"

    Skills installed from any git repo (local or remote) can be automatically updated.

## Installing Skills

Skills can be installed, updated and removed from the CLI or TUI with the `skills` command.

To view available skills from the current marketplace, use `/skills add` to display a list. Supply either a path, repository URL, or skill name/index to install a skill.

Use `/skills update` to check current versions and install updates.


You can also install a skill directly from a GitHub `SKILL.md` URL or a local
skill directory/file:

=== "TUI"

    ```text
    /skills add https://github.com/org/repo/blob/main/skills/example/SKILL.md
    /skills add ./skills/example
    /skills add ./skills/example/SKILL.md
    ```

=== "CLI"

    ```bash
    fast-agent skills add https://github.com/org/repo/blob/main/skills/example/SKILL.md
    fast-agent skills add ./skills/example
    fast-agent skills add ./skills/example/SKILL.md
    ```

For direct installs, fast-agent reads `SKILL.md` first and uses the manifest
`name` as the canonical install directory. Skills names must follow the format specified in the agentskills.io [specification](https://agentskills.io/specification).

### Registries and Marketplaces

fast-agent can install skills from online registries published in the Claude plugin format. By default, three registries are configured:

- [fast-agent](https://github.com/fast-agent-ai/skills)
- [Hugging Face](https://github.com/huggingface/skills)
- [Anthropic](https://github.com/anthropics/skills)

To browse and install skills from the marketplace:

=== "TUI"

    ```text
    /skills add
    ```

=== "CLI"

    ```bash
    fast-agent skills add
    ```

This displays available skills with numbers. Install by name or number:

=== "TUI"

    ```text
    /skills add 1
    /skills add skill-name
    ```

=== "CLI"

    ```bash
    fast-agent skills add 1
    fast-agent skills add skill-name
    ```


<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/tui/skills-slash-commands.cast"
  data-fa-asciinema-cols="96"
  data-fa-asciinema-rows="24"
  data-fa-asciinema-poster="npt:0:05"
  data-fa-asciinema-speed="1"
  data-fa-asciinema-idle-time-limit="1.3"
  data-fa-asciinema-fit="width"
>
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-terminal-theme="light">Light</button>
    <button type="button" data-fa-terminal-theme="dark">Dark</button>
  </div>
  <div data-fa-asciinema-target></div>
</div>

<!--
Cast asset:
- Source: docs/docs/assets/tui/skills-slash-commands.cast
- Regenerate: uv run scripts/docs.py cast-build skills-slash-commands
- Replay locally: asciinema play docs/docs/assets/tui/skills-slash-commands.cast
-->

### Removing Skills

Remove installed skills:

=== "TUI"

    ```text
    /skills remove skill-name
    /skills remove 1
    ```

=== "CLI"

    ```bash
    fast-agent skills remove skill-name
    fast-agent skills remove 1
    ```

### Managing Registries

In the TUI, view the current registry and available registries:

=== "TUI"

    ```text
    /skills registry
    ```

Example output:
```
# skills registry

Registry: https://github.com/huggingface/skills

Available registries:
- [1] https://github.com/huggingface/skills
- [2] https://github.com/anthropics/skills

Usage: `/skills registry [number|URL]`
```

In the TUI, switch registries by number or provide a custom URL:

=== "TUI"

    ```text
    /skills registry 2
    /skills registry https://github.com/my-org/my-skills
    ```

For CLI commands, pass `--registry` to browse or install from a specific registry for that invocation:

```bash
fast-agent skills available --registry https://github.com/my-org/my-skills
fast-agent skills add skill-name --registry https://github.com/my-org/my-skills
```

## Configuration

Configure skill directories and registries in `fast-agent.yaml`:

```yaml
skills:
  directories:
    - ".fast-agent/skills"
  marketplace_urls:
    - "https://github.com/huggingface/skills"
    - "https://github.com/anthropics/skills"
```

See the [Configuration Reference](../ref/config_file/#skills-configuration) for details.

## Command Line Options

If using **fast-agent** interactively from the command line, the `--skills <directory>` switch can be used to specify the directory containing skills. The `--env <path>` flag lets you relocate the entire environment directory (including the default skills folder).

```bash
# Specify a skills folder and a model
fast-agent go --skills ~/skill-development/testing/ --model "gpt-5-mini?reasoning=low"

# Give fast-agent access to the shell
fast-agent go -x
```

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/tui/skills-direct-install.cast"
  data-fa-asciinema-cols="96"
  data-fa-asciinema-rows="20"
  data-fa-asciinema-poster="npt:0:01"
  data-fa-asciinema-speed="1"
  data-fa-asciinema-idle-time-limit="1.3"
  data-fa-asciinema-fit="width"
>
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-terminal-theme="light">Light</button>
    <button type="button" data-fa-terminal-theme="dark">Dark</button>
  </div>
  <div data-fa-asciinema-target></div>
</div>

<!--
Cast asset:
- Source: docs/docs/assets/tui/skills-direct-install.cast
- Regenerate: uv run scripts/docs.py cast-build skills-direct-install
- Replay locally: asciinema play docs/docs/assets/tui/skills-direct-install.cast
-->


## Config File

Use `skills.directories` to set multiple skills directories. When provided (even as an empty list), the default search paths are not used.

```yaml
skills:
  directories:
    - ~/skills/team
    - ./skills/local
```

## Programmatic Usage

Skills directories can be defined on a per-agent basis:

```python
from fast_agent.constants import DEFAULT_SKILLS_PATHS

# Define the agent
@fast.agent(instruction=default_instruction, skills=DEFAULT_SKILLS_PATHS + ["~/source/skills"])
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.interactive()
```

This allows each individual agent to use a different set of skills if needed. To disable skills for an agent, pass `skills=[]`.
