### Personal update / status

AgentCard RFC implementation is mostly complete (see `plan/agent-card-rfc.md`). The main remaining risk is **quality + tests** because the change surface is large and I hit a weekly Codex token cap during implementation and debugging. I tried to cover all observed issues with test cases; now is a good time for another manual REPL test pass.

I’m also starting the **ai-telegram** subsystem as part of migrating from my old `call` repo [`strato-space/ai-telegram/srs.md`](https://github.com/strato-space/ai-telegram). ACP currently connects via stdin/stdout only; for long‑lived sessions I built a small Unix‑socket ACP proxy. It seems like a pragmatic starting point until network ACP transport is addressed.

This could also be a good test if I can plug AgentCards into that project and use them actively.
