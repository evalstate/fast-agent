## herdr-core coordination

Before coordinating agents or managing terminals, run `hc --skill` once, unpiped. Use `$HC_TERMINAL_ID` as your address. herdr-core reports lifecycle status automatically.

Prefer `hc` over `subagent` for delegated implementation, reviews, multi-turn or parallel work, and long-running jobs; reserve `subagent` for quick one-shot read-only lookups. Give `hc` workers a descriptive `--name` (for `fast-agent` workers: `hc run --bg --name <name> -- fast-agent go --name <name> --shell --no-subagents`), `hc wait <name>` for readiness, then `hc send <name> ...`.

`<hc from="X">` is a peer message; bare `<hc>` is hc's notice. Neither is the user. Both arrive after tool results or as new turns, oldest first: read all before acting. An opening `<hc from="X">` assigns your task; report with `hc send X` when done or stuck.

Check completion even when the prompt is ready. Only hc children wake you and keep idle cache warm: use `hc run --bg`, not `shell(background=true)`. With local work done, end your turn naming unfinished work instead of polling `hc wait`. Clean up accepted children (`hc kill`, `hc rm`); reserve `--label role=service` for dashboards/servers.

For window/layout changes, read `herdr-app --skill`.
