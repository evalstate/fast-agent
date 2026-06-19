# Plugin development area

This directory holds source for command plugins under development. Each
subdirectory is a plugin containing a `plugin.yaml` manifest and handler
modules — see `docs/docs/agents/plugins.md` for the manifest format and the
local development loop.

These sources are not loaded from this path at runtime. Install a plugin into
an environment's `plugins/` directory (for example `.fast-agent/plugins/`, or
via `fast-agent plugins add`) and enable it by name in `fast-agent.yaml`, or
publish it to a card-packs marketplace.
