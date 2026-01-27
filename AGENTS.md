## fast-agent contributor notes

- Use `uv run` for repo scripts and examples.
- Always run `uv run scripts/lint.py` and `uv run scripts/typecheck.py` after code changes.
- Keep examples under `examples/` in sync with packaged resources when relevant.
- Prefer small, focused diffs; avoid reformatting unrelated code.
- Use Markdown links for images and other content (example: `![Image](https://link.to.image)`).
- Some unit tests emit warning logs (e.g., invalid auths.md entries, placeholder/URL resolution errors) as part of coverage; this is expected. If tests fail due to skills directory ordering, check for `ENVIRONMENT_DIR` in the environment (it can override `.fast-agent` and skew skill discovery).
- Project layout quick map:
  - `src/fast_agent/` core runtime and agent logic.
  - `src/fast_agent/core/fastagent.py` owns agent registry, reload/watch, card tooling, and instance refresh.
  - `src/fast_agent/core/direct_factory.py` builds agents by type (including Agents-as-Tools wiring).
  - `src/fast_agent/core/agent_card_loader.py` parses/dumps AgentCards and resolves history/tool paths.
  - `src/fast_agent/ui/interactive_prompt.py` is the TUI entry; handles slash commands, agent switching, and tool lists.
  - `src/fast_agent/cli/commands/go.py` is the CLI entry for `fast-agent go` (cards, watch, reload).
  - `src/fast_agent/acp/` ACP server, slash commands, and transport glue.
  - `src/fast_agent/agents/` agent types; `agents/workflow/agents_as_tools_agent.py` is Agents-as-Tools.
  - `tests/unit/` and `tests/integration/` mirror runtime vs ACP/CLI behaviors.
  - `examples/` and `examples/workflows-md/` are kept in sync with packaged resources when they change.
