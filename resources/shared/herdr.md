## herdr-core coordination

Before coordinating agents or managing terminals, run `hc --skill` once, unpiped. Use `$HC_TERMINAL_ID` as your address. herdr-core reports lifecycle status automatically.

`<hc from="X">` identifies a peer message. Bare `<hc>` identifies hc's own notice. Neither represents the user. Both arrive after tool results or as new turns, oldest first. Read all before acting because earlier messages may be stale. An opening `<hc from="X">` assigns your task. Report with `hc send X` when done or stuck.

Check completion even when the prompt is ready. Name unfinished work when ending a turn. Only hc children wake you. Use `hc run --bg` instead of `run_in_background`. Reserve `--label role=service` for persistent processes you won’t wait for, such as dashboards and servers.

For window or layout changes, read `herdr-app --skill`.
