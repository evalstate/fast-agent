# Six-hour Terminal-Bench 2.1 experiments

Snapshot: September 5, 2026. These fast-agent experiments use a **21,600-second
agent timeout per trial**, overriding the standard task timeout. They are not
standard Terminal-Bench leaderboard submissions and should not be read as
like-for-like leaderboard comparisons.

Each result covers all 89 tasks with five trial slots each. Scores count missing
rewards as zero, rather than using the Hub overview's non-null-reward denominator.
The selected trials were checked for exactly five slots per task.

| Model (max reasoning) | Rewarded / 445 | Score | Recorded cost estimate | Non-null cost records / 445 selected trials |
| --- | --- | --- | --- | --- |
| GLM-5.3-Flash | 376 | 84.49% | $77.53 | 432 |
| DeepSeek V4 Flash Vision Exp | 370 | 83.15% | $53.68 | 386 |
| GPT-5.6 Luna | 366 | 82.25% | $29.54 | 424 |

## Accounting and selection

Costs are sums of Harbor's recorded source-job costs at the configured token
rates, **not billed API spend**. Null cost records are not assumed to be free;
coverage is incomplete. Totals and token counts include the original replaced
attempts, so their recorded consumption is not silently discarded. Earlier
aborted Luna r4–r6 jobs are not included in the r7 result.

- GLM: r2 Daytona plus local jobs; no replacements.
- DeepSeek: r2 Daytona plus local jobs. The replacement job supplies the
  `merge-diff-arc-agi-task` slot whose original trial
  `25fd3399-9b17-45a3-9a2f-dfb9cf57cd8a` had an `AgentSetupTimeoutError`.
  Three other missing rewards remain zero in the 445-slot score.
- Luna: r7 Daytona plus local jobs. The replacement job supplies the
  `winning-avg-corewars` slot whose original trial
  `3566a1e0-3572-47c1-85ed-c62afcee8687` had a `RuntimeError`.
  Luna used the Codex Responses route; the old Value chart's Flex/API-key
  accounting is not carried forward.

Configured USD rates per million tokens:

| Model | Input | Cached input | Output |
| --- | --- | --- | --- |
| GLM-5.3-Flash | 0.15 | 0.03 | 0.50 |
| DeepSeek V4 Flash Vision Exp | 0.44 | 0.014 | 1.32 |
| GPT-5.6 Luna | 0.20 | 0.02 | 1.20 |

## Source jobs

### GLM-5.3-Flash

- [tb21-glm53-flash-max-r2-long6h-0111-daytona-20260827-113312](https://hub.harborframework.com/jobs/aea81782-d18a-44ca-9417-9f6793342a1b)
- [tb21-glm53-flash-max-r2-long6h-0111-local-20260827-113312](https://hub.harborframework.com/jobs/c368f3bb-2a37-447e-9769-dfa6f1f1cd8a)

### DeepSeek V4 Flash Vision Exp

- [tb21-deepseek-v4-flash-vision-exp-max-r2-long6h-0113-daytona-20260828-222129](https://hub.harborframework.com/jobs/852e75c9-ca0e-4551-9393-21835a9857e2)
- [tb21-deepseek-v4-flash-vision-exp-max-r2-long6h-0113-local-20260828-222129](https://hub.harborframework.com/jobs/5cf4a449-18d4-4f14-80be-3f57581b7f16)
- [tb21-deepseek-v4-flash-vision-exp-max-r2-long6h-0113-replacement-daytona-20260830-105958](https://hub.harborframework.com/jobs/4359bceb-7ca0-478e-9c27-44a19d4d4568)

### GPT-5.6 Luna

- [tb21-luna-max-r7-long6h-0112-daytona-20260828-144748](https://hub.harborframework.com/jobs/059f8d77-1de3-40ea-953e-3261675843cb)
- [tb21-luna-max-r7-long6h-0112-local-20260828-144748](https://hub.harborframework.com/jobs/ae7acad7-64e4-4834-867b-d8c96f12e52c)
- [tb21-luna-max-r7-long6h-0112-replacement-daytona-20260830-104952](https://hub.harborframework.com/jobs/37e6871b-a41d-4e49-adfd-b30206d5c2c0)

## Reproducing the snapshot

Use `uv run harbor hub job list --search long6h --json` to locate jobs,
`uv run harbor hub job show JOB_ID --json` to inspect timeout and pricing config,
and `uv run harbor hub job trials JOB_ID... --limit 500 --json` to inspect trials.
Exclude the two explicitly replaced trial IDs above from score/coverage counts,
retain all other trial slots, and sum recorded costs across the linked jobs.
