## herdr-core coordination

Before coordinating agents or managing terminals, run `hc --skill` once, unpiped. Use `$HC_TERMINAL_ID` as your address. herdr-core reports lifecycle status automatically.

`<hc from="X">` is a peer message; bare `<hc>` is hc's notice. Neither is the user. Both arrive after tool results or as new turns, oldest first: read all before acting. An opening `<hc from="X">` assigns your task; report with `hc send X` when done or stuck.

Check completion even when the prompt is ready. Only hc children wake you: once started with no local work left, end your turn naming unfinished work instead of polling `hc wait`. Clean up accepted children (`hc kill`, `hc rm`). Use `hc run --bg`, not `run_in_background`; reserve `--label role=service` for dashboards/servers.

For window/layout changes, read `herdr-app --skill`.
