## fast-agent contributor notes

- Use `uv run` for repo scripts and examples.
- Always run `uv run scripts/lint.py` and `uv run scripts/typecheck.py` after code changes.
- Keep examples under `examples/` in sync with packaged resources when relevant.
- Prefer small, focused diffs; avoid reformatting unrelated code.
- Use Markdown links for images and other content (example: `![Image](https://link.to.image)`).
- Project layout quick map:
- `src/fast_agent/` core runtime and agent logic; `src/fast_agent/core/fastagent.py` owns agent registry, reload/watch, and card tooling.
- `src/fast_agent/core/agent_card_loader.py` parses/dumps AgentCards and resolves history/tool paths.
- `src/fast_agent/ui/` interactive prompt, slash commands, and usage display; `interactive_prompt.py` is the TUI entry.
- `src/fast_agent/acp/` ACP server, slash commands, and transport glue.
- `src/fast_agent/agents/` agent types; `agents/workflow/agents_as_tools_agent.py` is agents-as-tools.
- `tests/unit/` and `tests/integration/` mirror runtime vs ACP/CLI behaviors.
- `examples/` and `examples/workflows-md/` are kept in sync with packaged resources when they change.
