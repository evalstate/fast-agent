# Agent Skills

**`fast-agent`** supports Agent Skills, and by default searches the directories listed in `DEFAULT_SKILLS_PATHS`:

- `.fast-agent/skills`
- `.claude/skills`

When valid SKILL.md files are found:

- The Agent is given access to an `execute` tool for running shell commands, with the working directory set to the workspace root.
- Skill descriptions from the manifest and path are added to the System Prompt using the `{{agentSkills}}` expansion. A warning is displayed if this is not present in the System Prompt.
- The `/skills` command lists the available skills.
- If duplicate skill names exist across directories, later directories override earlier ones. Warnings are surfaced in `/status` (ACP).

## Skill Marketplace

fast-agent can install skills from online registries. By default, two registries are configured:

- [HuggingFace Skills](https://github.com/huggingface/skills)
- [Anthropic Skills](https://github.com/anthropics/skills)

### Installing Skills

Use the `/skills add` command to browse and install skills from the marketplace:

```
/skills add
```

This displays available skills with numbers. Install by name or number:

```
/skills add 1
/skills add skill-name
```

### Removing Skills

Remove installed skills with `/skills remove`:

```
/skills remove skill-name
/skills remove 1
```

### Managing Registries

View the current registry and available registries:

```
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

Switch registries by number or provide a custom URL:

```
/skills registry 2
/skills registry https://github.com/my-org/my-skills
```

## Configuration

Configure skill directories and registries in `fastagent.config.yaml`:

```yaml
skills:
  directories:
    - ".fast-agent/skills"
  marketplace_urls:
    - "https://github.com/huggingface/skills"
    - "https://github.com/anthropics/skills"
```

See the [Configuration Reference](../ref/config_file.md#skills-configuration) for details.

## Command Line Options

If using **fast-agent** interactively from the command line, the `--skills <directory>` switch can be used to specify the directory containing skills. The `--env <path>` flag lets you relocate the entire environment directory (including the default skills folder).

```bash
# Specify a skills folder and a model
fast-agent go --skills ~/skill-development/testing/ --model gpt-5-mini.low

# Give fast-agent access to the shell
fast-agent go -x
```

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
