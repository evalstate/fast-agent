## herdr-core coordination

This fast-agent runs in a herdr-core terminal. Before coordinating agents or managing terminals, run `hc --skill` once, unpiped. Your address is `$HC_TERMINAL_ID`, not a pane ID. Lifecycle status is reported automatically.

`<hc from="X">` is a peer message; bare `<hc>` is hc's own notice. Neither is your user. They arrive after a tool result or as a new turn, oldest first, possibly stale: read all before acting. A first message `<hc from="X">` is your task; report with `hc send X` when done or stuck.

A ready prompt does not prove delegated work is complete. Ending a turn with work running? Name it. Only hc children wake you: use `hc run --bg`, not `run_in_background`; label servers `--label role=service`.

For window or layout changes, read `herdr-app --skill`.
