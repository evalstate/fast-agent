# Getting Started

## Install or upgrade

```bash
uv tool install -U fast-agent-mcp
```

If you have multiple Python versions installed, pin the one required by fast-agent:

```bash
uv tool install -U fast-agent-mcp --python 3.13.5
```

## Run

```bash
fast-agent go
```

## Run a card pack

```bash
fast-agent go --pack analyst --model haiku
fast-agent go --pack analyst --pack-registry ./marketplace.json --agent planner --model haiku
```

`--pack` installs the pack into the selected fast-agent environment if needed,
then launches `go` normally. `--model` is a fallback for cards without an
explicit model setting.

## Instruction file

```bash
fast-agent go -i prompt.md
fast-agent go -i https://gist.github.com/....
```

## Model override

```bash
fast-agent go --model sonnet
```
