# Measuring interactive startup

Measure startup to the first input prompt, separately from authentication and
first-response latency. The benchmark sends Ctrl+D as soon as the prompt renders;
it sends no user message. Normal startup still loads your configuration, starts
configured MCP servers, runs startup hooks, and creates a session.

For repeatable process-to-rendered-prompt measurements on Linux/macOS:

```bash
uv run scripts/benchmark_startup.py --runs 7 --output /tmp/startup.json -- \
  .venv/bin/fast-agent --model copilot.gemini-4-neon -x
```

This runs one excluded warmup and seven fresh processes in a pseudo-terminal,
prints the median/range, and saves individual results plus adjacent terminal logs.
It recognizes the default `❯` input prompt and bracketed-paste activation.
Use `--warmup 0` to retain the first launch in the results. Measurements include
process creation and Python startup, but exclude shutdown and the outer `uv run`.
Warmup does not make these cold-disk measurements: filesystem and bytecode caches
remain warm across launches. Run comparisons sequentially without concurrent tests
or profilers.

Use `fast-agent` instead of `.venv/bin/fast-agent` to measure the installed command.
They may resolve to different packages/interpreters; editing this checkout does
not update a separate `uv tool` installation.

For attribution, collect a separate import trace or cProfile run. Profiling adds
overhead, so do not compare those totals with unprofiled timings:

```bash
uv run scripts/benchmark_startup.py --runs 1 --warmup 0 --output /tmp/imports.json -- \
  .venv/bin/python -X importtime -m fast_agent.cli --model copilot.gemini-4-neon -x

uv run scripts/benchmark_startup.py --runs 1 --warmup 0 --output /tmp/profile.json -- \
  .venv/bin/python -m cProfile -o /tmp/startup.prof -m fast_agent.cli \
  --model copilot.gemini-4-neon -x

uv run python -c "import pstats; pstats.Stats('/tmp/startup.prof').sort_stats('cumulative').print_stats(40)"
```

The cProfile file includes shutdown as well as startup; asynchronous/concurrent
call totals overlap. Use the external benchmark for wall time.

## Initial findings (2026-11-26)

On this Linux checkout with Python 3.15 and the command above, initial five-run
medians were 1.286 s for the installed package and 1.331 s for the checkout. Imports
dominated the profile, especially MCP/Pydantic models and the OpenAI SDK used by
Copilot Chat Completions. No model request was needed to reach the prompt.

Two small improvements were made:

- Startup process discovery reads completed processes' status records without
  loading their specs, output counters/sizes, and session links. This workspace
  had 100 retained process records. Full discovery for `/process` is unchanged;
  active records still reconcile missing/stale supervisors normally.
- Router protobuf imports now occur when building routing instructions, instead
  of when importing the prompt's agent-info display.

Two controlled rounds, reversing measurement order in the second round, used
identical source snapshots except for these changes. Across 14 measured launches
per version, medians were **1.339 s before and 1.306 s after** (about **32 ms /
2.4%**). This is a small, workload-dependent saving, especially dependent on
retained process history. Both snapshots included the timing hooks with reporting
disabled. Source snapshots isolated the comparison from concurrent checkout edits.

Temporary instrumentation measured these stage durations in one optimized run.

| Stage | Milliseconds |
| --- | ---: |
| CLI imports | 76 |
| Command dispatch/config discovery | 267 |
| Runtime imports | 195 |
| Recovery/model selection | 45 |
| Agent configuration | 247 |
| Harness entry | 12 |
| Agent/provider/session initialization | 339 |
| Input setup | 110 |

### Shell construction and SDK imports

With installed skills and `-x`, the agent constructor used the live skill-update
path, which immediately created a shell. Initial shell configuration then created
a replacement with the combined access modes. Preparing the initial skill data
before configuring the shell reduces durable-store construction and pruning from
two passes to one (confirmed with cProfile). Seven fresh-process runs per version,
interleaved after warmup, gave medians of 1.288 s before and 1.281 s after. The
7 ms difference is smaller than the run-to-run variation.

On this Python 3.15 environment, a fresh interpreter took 334 ms to import MCP
2.3.0 alone and 290 ms to import OpenAI 3.52.0 alone (five-run medians, measured
inside Python). Importing both in one interpreter took about 510 ms in either
order because they share dependencies. During actual CLI startup, after earlier
imports, the import trace attributed 198 ms to MCP and 190 ms to OpenAI. MCP
imports client and server code and two protocol schema versions. OpenAI imports
its generated types, including Responses types, even on this Chat Completions
route. The incremental CLI costs depend on import order and are the relevant
figures when judging a deferral.

### Deferring the OpenAI SDK on Copilot Chat Completions

The provider now loads the SDK when preparing the first request. Model selection,
reasoning settings, and tool configuration still run during startup. For
`fast-agent --model copilot.gemini-4-neon -x` in this checkout, the median time
from process launch to the rendered prompt fell from **1.322 s to 1.134 s**
(188 ms, 14.2%). The benchmark alternated between source snapshots, excluded
one warmup per version, and measured seven fresh processes per version.

A second experiment used the same model and shell flag in a temporary working
directory with `--no-home`. It submitted a message as soon as the first prompt
appeared, supplied a synthetic token, and intercepted the HTTP send in memory
to return a streamed Chat Completions response. Each run checked whether
`openai` was loaded at the first prompt and after the response.

| Median over seven runs per version | Before | After |
| --- | ---: | ---: |
| Launch to prompt in the mock session | 1.127 s | 0.944 s |
| Submit to HTTP dispatch | 123 ms | 289 ms |
| Launch to HTTP dispatch | 1.242 s | 1.232 s |

Most of the import cost moves to the first send. Launch-to-dispatch varied by
more than the 9 ms difference between medians. These measurements cover local
request preparation. A live Copilot call also incurs network and inference time.
