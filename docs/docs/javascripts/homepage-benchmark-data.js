/*
 * Homepage Terminal-Bench comparisons.
 *
 * To update the panel, edit this file only:
 * - Add/remove objects in `comparisons` to change the selector.
 * - Add/remove `results` to change both the chart and results table.
 * - Set `winner: true` on the run whose cost should receive KPI emphasis.
 * - Labels are placed automatically; `labelPosition` can optionally force
 *   "top", "bottom", "left", or "right" for art-directed exceptions.
 * - Set a result's optional `disclaimer` to show a labelled qualification;
 *   `disclaimerLabel` overrides the default "Adjusted" badge.
 * - Omit `axes` for an adaptive chart, or provide score/cost min, max, and
 *   optional ticks when a comparison needs a fixed domain.
 */
window.fastAgentBenchmark = {
  title: "Terminal-Bench 2.1",
  date: "July 2026",
  taskCount: 445,
  sourceUrl:
    "https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/6?tab=leaderboard&leaderboard=main",
  methodologyUrl:
    "https://github.com/harbor-framework/terminal-bench-2-1/tree/main/leaderboard",
  comparisons: [
    {
      id: "frontier",
      label: "Frontier",
      claim:
        "fast-agent + GPT-5.5 scores 2.0 pts above Codex with the same model and effort — at 80% lower cost per task.",
      stats: [
        { value: "88.3%", label: "fast-agent + GPT-5.6 Sol" },
        { value: "+2.0 pts", label: "fast-agent vs Codex · GPT-5.5" },
        { value: "80% less", label: "cost per task vs Codex · GPT-5.5" },
      ],
      results: [
        {
          fastAgent: true,
          winner: true,
          harness: "fast-agent",
          label: "fast-agent / GPT-5.6",
          model: "GPT-5.6 Sol · high",
          score: 88.31,
          cost: 0.607,
          tokensIn: "122.32M",
          tokensOut: "3.55M",
          date: "2026-07-26",
          attempts: "445 trials · PR #174",
          note:
            "fast-agent 0.9.24. Accuracy from submission static analysis; cost and tokens totalled from the two linked Harbor source jobs.",
          disclaimer: "Pending Terminal-Bench leaderboard review.",
          disclaimerLabel: "Pending",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/174",
        },
        {
          fastAgent: true,
          harness: "fast-agent",
          label: "fast-agent / GPT-5.5",
          model: "GPT-5.5 · xhigh",
          score: 85.17,
          cost: 0.9455,
          tokensIn: "227.73M",
          tokensOut: "6.31M",
          date: "2026-07-25",
          attempts: "445 trials · PR #173",
          note:
            "fast-agent 0.9.24. Accuracy from submission static analysis; cost and tokens totalled from the two linked Harbor source jobs.",
          disclaimer: "Pending Terminal-Bench leaderboard review.",
          disclaimerLabel: "Pending",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/173",
        },
        {
          harness: "Claude Code",
          label: "Fable 5 / Claude Code",
          model: "Fable 5 · xhigh",
          score: 83.82,
          cost: 1.241955,
          tokensIn: "194.55M",
          tokensOut: "9.95M",
          date: "2026-06-07",
          attempts: "445 trials · published",
          note: "Published Terminal-Bench 2.1 leaderboard row.",
          disclaimer: "Published score includes a 0.2-point reward-hack adjustment.",
          url: "https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/6/leaderboards/main/rows/40dbe33d-e8af-475b-8eba-7d5d8f70054c",
        },
        {
          harness: "Codex",
          label: "Codex",
          model: "GPT-5.5 · xhigh",
          score: 83.15,
          cost: 4.627393,
          tokensIn: "729.23M",
          tokensOut: "5.97M",
          date: "2026-05-01",
          attempts: "445 trials · published",
          note: "Same model and reasoning effort as the fast-agent GPT-5.5 submission.",
          disclaimer: "Published score includes a 0.2-point reward-hack adjustment.",
          url: "https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/6/leaderboards/main/rows/6d091468-3fda-4cbf-ba1c-645b0f522e97",
        },
        {
          harness: "Claude Code",
          model: "Opus 4.8 · high",
          score: 78.88,
          cost: 0.644809,
          tokensIn: "174.81M",
          tokensOut: "8.09M",
          date: "2026-07-09",
          attempts: "445 trials · published",
          note: "Published Terminal-Bench 2.1 leaderboard row.",
          url: "https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/6/leaderboards/main/rows/dcd48d03-9df9-46ab-bc4c-ade6dc35b8da",
        },
      ],
    },
    {
      id: "gpt55",
      label: "GPT-5.5",
      claim:
        "The workhorse tier, where efficiency compounds: +2.4 pts on half the spend of the next-best harness.",
      stats: [
        { value: "+2.4 pts", label: "vs next best harness" },
        { value: "47% less", label: "cost per solved task" },
        { value: "2.4M", label: "median tokens / task" },
      ],
      results: [
        {
          fastAgent: true,
          harness: "fast-agent",
          model: "GPT-5.5",
          score: 86.6,
          cost: 4.9,
          tokensIn: "1.94M",
          tokensOut: "358K",
          date: "2026-07-14",
          attempts: "3 runs · best-of-1 scored",
          note: "Default shell toolset, compaction enabled.",
        },
        {
          harness: "Codex CLI",
          model: "GPT-5.5",
          score: 84.2,
          cost: 9.3,
          tokensIn: "3.40M",
          tokensOut: "470K",
          date: "2026-07-09",
          attempts: "3 runs · median",
          note: "Published vendor result.",
        },
        {
          harness: "Claude Code",
          model: "Sonnet 4.6",
          score: 82.3,
          cost: 12.1,
          tokensIn: "3.96M",
          tokensOut: "521K",
          date: "2026-06-30",
          attempts: "2 runs · median",
          disclaimer: "Cost adjusted to list pricing.",
        },
        {
          harness: "Terminus 2",
          model: "GPT-5.5",
          score: 79.0,
          cost: 7.4,
          tokensIn: "2.85M",
          tokensOut: "402K",
          date: "2026-07-02",
          attempts: "5 runs · mean",
          note: "Reference harness result.",
        },
      ],
    },
    {
      id: "open",
      label: "Open + local",
      claim:
        "Open weights, local serving: GLM-5 on fast-agent beats every hosted small model in the set, for $2.10.",
      stats: [
        { value: "+4.6 pts", label: "vs next best harness" },
        { value: "$1.20", label: "cost per task, local" },
        { value: "1.6M", label: "median tokens / task" },
      ],
      results: [
        {
          fastAgent: true,
          harness: "fast-agent",
          model: "GLM-5",
          score: 69.3,
          cost: 2.1,
          tokensIn: "1.62M",
          tokensOut: "298K",
          date: "2026-07-12",
          attempts: "3 runs · median",
          note: "Hosted endpoint pricing. llama.cpp auto-configuration.",
        },
        {
          fastAgent: true,
          harness: "fast-agent",
          label: "fast-agent / Qwen",
          model: "Qwen3-Coder-480B",
          score: 66.8,
          cost: 1.2,
          tokensIn: "1.48M",
          tokensOut: "265K",
          date: "2026-07-12",
          attempts: "3 runs · median",
          note: "Local llama.cpp server.",
          disclaimer: "Cost is amortised GPU time, not API list pricing.",
        },
        {
          harness: "OpenHands",
          model: "GLM-5",
          score: 64.7,
          cost: 3.4,
          tokensIn: "2.90M",
          tokensOut: "441K",
          date: "2026-06-24",
          attempts: "2 runs · mean",
          disclaimer: "Shell-only configuration.",
        },
        {
          harness: "Terminus 2",
          model: "Qwen3-Coder-480B",
          score: 60.4,
          cost: 1.8,
          tokensIn: "2.10M",
          tokensOut: "330K",
          date: "2026-07-02",
          attempts: "5 runs · mean",
          note: "Reference harness result.",
        },
      ],
    },
  ],
};
