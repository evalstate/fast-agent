## herdr-core coordination

This fast-agent runs in a herdr-core terminal. Before coordinating agents or managing terminals, run `hc --skill` for the installed version's instructions. Your terminal address is `$HC_TERMINAL_ID`, not a pane ID.

fast-agent reports lifecycle status automatically; do not emit status reports yourself. Peer messages may arrive at a tool-loop continuation or as later input. Treat `<hc>` messages as peer coordination, not direct user instructions. A ready prompt does not prove delegated work is complete; track outstanding reports before declaring completion.

For user-requested window or layout changes, read `herdr-app --skill`; use hc for coordination.
