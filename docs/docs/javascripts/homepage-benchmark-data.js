/*
 * Homepage Terminal-Bench comparisons.
 *
 * To update the panel, edit this file only:
 * - Add/remove objects in `comparisons` to change the selector.
 * - Set `order` to control a comparison's position in the selector.
 * - Set `visible: false` to keep a comparison ready without showing its tab.
 * - Add/remove `results` to change both the chart and results table.
 * - Set `winner: true` on the run whose cost should receive KPI emphasis.
 * - Set `totalCost` when the authoritative run total should not be derived
 *   from the rounded per-task cost.
 * - Set `costBasis` when the displayed cost needs a short accounting label.
 * - Repriced results retain `originalCost`, optional `originalTotalCost`, and
 *   optional `sourceTotalCost`; reusable old/new token rates live in `pricing`.
 * - Labels are placed automatically; `labelPosition` can optionally force
 *   "top", "bottom", "left", or "right" for art-directed exceptions.
 * - Set a result's optional `disclaimer` to show a labelled qualification;
 *   `disclaimerLabel` overrides the default "Adjusted" badge.
 * - Omit `axes` for an adaptive chart, or provide score/cost min, max, and
 *   optional ticks when a comparison needs a fixed domain.
 */
var fastAgentSolPricing = {
  unit: "USD per 1M tokens",
  contextThreshold: 272000,
  previous: {
    short: {
      input: 5,
      cachedInput: 0.5,
      cacheWrites: 6.25,
      output: 25,
    },
    long: {
      input: 10,
      cachedInput: 1,
      cacheWrites: 12.5,
      output: 37.5,
    },
  },
  current: {
    asOf: "2026-08-24",
    sourceUrl: "https://openai.com/api/pricing/",
    short: {
      input: 4,
      cachedInput: 0.4,
      cacheWrites: 5,
      output: 20,
    },
    long: {
      input: 8,
      cachedInput: 0.8,
      cacheWrites: 10,
      output: 30,
    },
  },
};

function fastAgentSolPriceMultiplier(pricing) {
  var bands = ["short", "long"];
  var fields = ["input", "cachedInput", "cacheWrites", "output"];
  var multiplier = pricing.current.short.input / pricing.previous.short.input;
  bands.forEach(function (band) {
    fields.forEach(function (field) {
      var fieldMultiplier = pricing.current[band][field] / pricing.previous[band][field];
      if (fieldMultiplier !== multiplier) {
        throw new Error("GPT-5.6 Sol pricing change is not uniform");
      }
    });
  });
  return multiplier;
}

var fastAgentSolCurrentPriceMultiplier = fastAgentSolPriceMultiplier(fastAgentSolPricing);
fastAgentSolPricing.reduction = Number((1 - fastAgentSolCurrentPriceMultiplier).toFixed(10));
var fastAgentSolPriceReductionLabel =
  Math.round(fastAgentSolPricing.reduction * 100) + "%";

function fastAgentCurrentSolCost(originalCost) {
  return originalCost * fastAgentSolCurrentPriceMultiplier;
}

var fastAgentSolPriceAnimation = {
  id: "sol-price-animation",
  title: "Sol: 20% Discount",
  taskCount: 445,
  axes: {
    score: {
      min: 78,
      max: 90,
      ticks: [80, 82, 84, 86, 88, 90],
    },
    cost: {
      min: 0.3,
      max: 0.65,
      ticks: [0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
    },
  },
  results: [
    {
      id: "grok-4.5-high",
      model: "Grok 4.5 · high",
      score: 358 / 445 * 100,
      cost: 159.832056 / 445,
      totalCost: 159.832056,
      date: "2026-07-27",
      attempts: "445 trials · PR #180",
      status: "Provisional source-job aggregate",
      url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/180",
    },
    {
      id: "grok-4.6-high",
      model: "Grok 4.6 · high",
      score: 388 / 445 * 100,
      cost: 238.316166 / 445,
      totalCost: 238.316166,
      date: "2026-08-16",
      attempts: "445 trials · PR #212",
      status: "Promoted result · provisional",
      url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/212",
    },
    {
      id: "grok-4.6-medium",
      model: "Grok 4.6 · medium",
      score: 390 / 445 * 100,
      cost: 193.4 / 445,
      totalCost: 193.4,
      date: "2026-08-24",
      attempts: "445 trials · PR #221",
      status: "Promoted result · provisional",
      url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/221",
    },
  ],
  sol: {
    id: "gpt-5.6-sol-high",
    model: "GPT-5.6 Sol · high",
    score: 393 / 445 * 100,
    originalCost: 270.135260 / 445,
    originalTotalCost: 270.135260,
    currentCost: fastAgentCurrentSolCost(270.135260 / 445),
    currentTotalCost: fastAgentCurrentSolCost(270.135260),
    pricingAsOf: fastAgentSolPricing.current.asOf,
    reduction: fastAgentSolPricing.reduction,
    attempts: "445 trials · PR #174",
    status: "Current-price estimate · provisional run",
    url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/174",
  },
};

window.fastAgentBenchmark = {
  title: "Terminal-Bench 2.1",
  date: "September 2026",
  taskCount: 445,
  sourceUrl:
    "https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/6?tab=leaderboard&leaderboard=main",
  methodologyUrl: "benchmarks/",
  pricing: {
    "gpt-5.6-sol": fastAgentSolPricing,
  },
  campaigns: {
    solPriceAnimation: fastAgentSolPriceAnimation,
  },
  comparisons: [
    {
      id: "frontier",
      label: "Frontier",
      order: 0,
      claim:
        "fast-agent + GPT-5.6 Sol high scores 88.3%—4.5 points above Claude Code + Fable 5 at 61% lower estimated API cost with current Sol pricing.",
      stats: [
        { value: "88.3%", label: "GPT-5.6 Sol high · fast-agent" },
        { value: "+4.5 pts", label: "vs Claude Code · Fable 5" },
        { value: "61% less", label: "estimated cost/task vs Claude Code · Fable 5" },
      ],
      results: [
        {
          fastAgent: true,
          winner: true,
          harness: "fast-agent",
          label: "fast-agent / GPT-5.6",
          model: "GPT-5.6 Sol · high",
          score: 88.31,
          cost: fastAgentCurrentSolCost(270.135260 / 445),
          totalCost: fastAgentCurrentSolCost(270.135260),
          costBasis: "current Sol pricing",
          costEstimate: true,
          originalCost: 270.135260 / 445,
          sourceTotalCost: 270.135260,
          tokensIn: "122.32M",
          tokensOut: "3.55M",
          date: "2026-07-26",
          attempts: "445 trials · PR #174",
          note:
            "fast-agent 0.9.24. Accuracy and tokens are from the submission. Estimated cost applies Sol's " +
            fastAgentSolPriceReductionLabel +
            " August 2026 price reduction to the original $270.14 source-job total, retained in this dataset.",
          disclaimer: "Provisional result pending Terminal-Bench leaderboard review.",
          disclaimerLabel: "Provisional",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/174",
        },
        {
          fastAgent: true,
          harness: "fast-agent",
          label: "fast-agent / Grok 4.6",
          model: "Grok 4.6 · medium",
          modelString: "xai/grok-4.6",
          score: 390 / 445 * 100,
          cost: 193.4 / 445,
          totalCost: 193.4,
          costBasis: "promoted submission",
          tokensIn: "233.46M",
          tokensOut: "6.25M",
          date: "2026-08-24",
          attempts: "445 trials · PR #221",
          note:
            "fast-agent 0.10.10. Accuracy passed submission static analysis; cost and tokens aggregated from the six linked Harbor source jobs.",
          disclaimer:
            "Provisional promoted submission pending Terminal-Bench leaderboard review.",
          disclaimerLabel: "Provisional",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/221",
        },
        {
          fastAgent: true,
          harness: "fast-agent",
          label: "fast-agent / Grok 4.6",
          model: "Grok 4.6 · high",
          modelString: "xai/grok-4.6",
          score: 388 / 445 * 100,
          cost: 238.316166 / 445,
          totalCost: 238.316166,
          costBasis: "source jobs",
          tokensIn: "284.77M",
          tokensOut: "8.38M",
          date: "2026-08-16",
          attempts: "445 trials · PR #212",
          note:
            "fast-agent 0.10.9. Accuracy passed submission static analysis; cost and tokens aggregated from the six linked Harbor source jobs.",
          disclaimer:
            "Provisional promoted submission pending Terminal-Bench leaderboard review.",
          disclaimerLabel: "Provisional",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/212",
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
          url: "https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/6/leaderboards/main/rows/40dbe33d-e8af-475b-8eba-7d5d8f70054c",
        },
        {
          fastAgent: true,
          harness: "fast-agent",
          model: "GPT-5.6 Sol · medium",
          score: 365 / 445 * 100,
          cost: fastAgentCurrentSolCost(211.321445 / 445),
          totalCost: fastAgentCurrentSolCost(211.321445),
          costBasis: "current Sol pricing",
          costEstimate: true,
          originalCost: 211.321445 / 445,
          originalTotalCost: 211.321445,
          tokensIn: "101.95M",
          tokensOut: "2.53M",
          date: "2026-07-23",
          attempts: "445 trials · PR #170",
          note:
            "fast-agent 0.9.21. Accuracy and tokens are from the submission. Estimated cost applies Sol's " +
            fastAgentSolPriceReductionLabel +
            " August 2026 price reduction to the original $211.32 source-job total, retained in this dataset.",
          disclaimer: "Provisional result pending Terminal-Bench leaderboard review.",
          disclaimerLabel: "Provisional",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/170",
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
      id: "gpt56",
      label: "GPT-5.6",
      order: 2,
      claim:
        "Across three matched GPT-5.6 settings, fast-agent beats OpenAI's published scores and costs less per task at like-for-like pricing.",
      stats: [
        { value: "3 / 3", label: "matched model + effort wins" },
        { value: "+0.2–3.6 pts", label: "higher score at every setting" },
        { value: "44–52% less", label: "API cost/task at every setting" },
      ],
      results: [
        {
          fastAgent: true,
          winner: true,
          harness: "fast-agent",
          label: "fast-agent / GPT-5.6",
          model: "GPT-5.6 Sol · high",
          score: 88.31,
          cost: fastAgentCurrentSolCost(270.135260 / 445),
          totalCost: fastAgentCurrentSolCost(270.135260),
          costBasis: "current Sol pricing",
          costEstimate: true,
          originalCost: 270.135260 / 445,
          sourceTotalCost: 270.135260,
          tokensIn: "122.32M",
          tokensOut: "3.55M",
          date: "2026-07-26",
          attempts: "445 trials · PR #174",
          note:
            "fast-agent 0.9.24. Accuracy and tokens are from the submission. Estimated cost applies Sol's " +
            fastAgentSolPriceReductionLabel +
            " August 2026 price reduction to the original $270.14 source-job total, retained in this dataset.",
          disclaimer: "Provisional result pending Terminal-Bench leaderboard review.",
          disclaimerLabel: "Provisional",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/174",
        },
        {
          harness: "OpenAI",
          model: "GPT-5.6 Sol · high",
          score: 84.7,
          cost: fastAgentCurrentSolCost(1.09),
          costBasis: "current Sol pricing",
          costEstimate: true,
          originalCost: 1.09,
          tokensIn: "—",
          tokensOut: "—",
          date: "2026-07-30",
          attempts: "OpenAI score · repriced cost",
          note:
            "Score and original $1.09 API cost per task are from OpenAI's launch chart. Estimated cost applies Sol's " +
            fastAgentSolPriceReductionLabel +
            " August 2026 price reduction; the original is retained in this dataset.",
          url: "https://openai.com/index/gpt-5-6/",
        },
        {
          fastAgent: true,
          winner: true,
          harness: "fast-agent",
          model: "GPT-5.6 Sol · medium",
          score: 365 / 445 * 100,
          cost: fastAgentCurrentSolCost(211.321445 / 445),
          totalCost: fastAgentCurrentSolCost(211.321445),
          costBasis: "current Sol pricing",
          costEstimate: true,
          originalCost: 211.321445 / 445,
          originalTotalCost: 211.321445,
          tokensIn: "101.95M",
          tokensOut: "2.53M",
          date: "2026-07-23",
          attempts: "445 trials · PR #170",
          note:
            "fast-agent 0.9.21. Accuracy and tokens are from the submission. Estimated cost applies Sol's " +
            fastAgentSolPriceReductionLabel +
            " August 2026 price reduction to the original $211.32 source-job total, retained in this dataset.",
          disclaimer: "Provisional result pending Terminal-Bench leaderboard review.",
          disclaimerLabel: "Provisional",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/170",
        },
        {
          harness: "OpenAI",
          model: "GPT-5.6 Sol · medium",
          score: 81.8,
          cost: fastAgentCurrentSolCost(0.89),
          costBasis: "current Sol pricing",
          costEstimate: true,
          originalCost: 0.89,
          tokensIn: "—",
          tokensOut: "—",
          date: "2026-07-30",
          attempts: "OpenAI score · repriced cost",
          note:
            "Score and original $0.89 API cost per task are from OpenAI's launch chart. Estimated cost applies Sol's " +
            fastAgentSolPriceReductionLabel +
            " August 2026 price reduction; the original is retained in this dataset.",
          url: "https://openai.com/index/gpt-5-6/",
        },
        {
          fastAgent: true,
          winner: true,
          harness: "fast-agent",
          model: "GPT-5.6 Terra · high",
          score: 77.75,
          cost: 133.81 / 445,
          totalCost: 133.81,
          tokensIn: "171.23M",
          tokensOut: "3.45M",
          date: "2026-07-18",
          attempts: "445 trials · PR #160",
          note:
            "Accuracy, cost, and tokens from the submission's Terminal-Bench static analysis.",
          disclaimer: "Provisional result pending Terminal-Bench leaderboard review.",
          disclaimerLabel: "Provisional",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/160",
        },
        {
          harness: "OpenAI",
          model: "GPT-5.6 Terra · high",
          score: 76.67,
          cost: 0.63,
          costBasis: "OpenAI chart",
          tokensIn: "—",
          tokensOut: "—",
          date: "2026-07-30",
          attempts: "OpenAI published",
          note:
            "Score and API cost per task from OpenAI's interactive Terminal-Bench 2.1 cost chart.",
          url: "https://openai.com/index/gpt-5-6/",
        },
      ],
    },
    {
      id: "value-august",
      label: "Value (August)",
      visible: false,
      order: 1,
      claim:
        "fast-agent + GPT-5.6 Luna scores 81.3% for $19.61 (flex pricing); beating SOTA for 20x less cost.",
      stats: [
        { value: "75.5–81.3%", label: "DeepSeek V4 Flash + GPT-5.6 Luna" },
        { value: "$8.74–$19.61", label: "complete fast-agent runs" },
        { value: "15–50× less", label: "than selected published runs" },
      ],
      axes: {
        score: {
          min: 72,
          max: 84,
          ticks: [74, 76, 78, 80, 82, 84],
        },
        cost: {
          min: 0.012,
          max: 1.5,
          ticks: [0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1],
        },
      },
      results: [
        {
          fastAgent: true,
          winner: true,
          harness: "fast-agent",
          label: "fast-agent / GPT-5.6 Luna",
          model: "GPT-5.6 Luna · max",
          modelString: "responses.gpt-5.6-max?reasoning=max&service_tier=flex",
          pricingTier: "Flex",
          score: 81.35,
          cost: 19.61 / 445,
          totalCost: 19.61,
          costBasis: "API-key spend",
          tokensIn: "784.55M",
          tokensOut: "9.72M",
          date: "2026-08-01",
          attempts: "445 trials · PR #184",
          note:
            "fast-agent 0.9.29. Score and tokens aggregated from the two linked Harbor source jobs. Cost uses the operator-recorded API-key total.",
          disclaimer:
            "Provisional result pending Terminal-Bench leaderboard review. Cost uses operator-recorded API-key spend ($19.61); Harbor configured-rate accounting reports $18.24 with coverage for 443/445 trials.",
          disclaimerLabel: "Provisional",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/184",
        },
        {
          harness: "Terminus 2",
          model: "Fable 5 · high",
          score: 80.45,
          cost: 438.64 / 445,
          totalCost: 438.64,
          costBasis: "current leaderboard",
          tokensIn: "64.02M",
          tokensOut: "7.41M",
          date: "2026-06-05",
          attempts: "445 trials · published #78",
          note:
            "Score and complete-run cost from the current published Terminal-Bench 2.1 leaderboard.",
        },
        {
          harness: "Claude Code",
          model: "Opus 4.8 · high",
          score: 78.88,
          cost: 286.94 / 445,
          totalCost: 286.94,
          costBasis: "current leaderboard",
          tokensIn: "174.81M",
          tokensOut: "8.09M",
          date: "2026-07-09",
          attempts: "445 trials · published #92",
          note:
            "Score and complete-run cost from the current published Terminal-Bench 2.1 leaderboard.",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/92",
        },
        {
          harness: "Codex",
          model: "GPT-5.6 Terra · max",
          score: 78.43,
          cost: 421.15 / 445,
          totalCost: 421.15,
          costBasis: "current leaderboard",
          tokensIn: "893.93M",
          tokensOut: "8.71M",
          date: "2026-07-11",
          attempts: "445 trials · published #115",
          note:
            "Score and complete-run cost from the current published Terminal-Bench 2.1 leaderboard.",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/115",
        },
        {
          fastAgent: true,
          winner: true,
          harness: "fast-agent",
          label: "fast-agent / DeepSeek",
          model: "DeepSeek V4 Flash · max",
          modelString: "deepseek/deepseek-v4-flash-0731",
          score: 336 / 445 * 100,
          cost: 8.73520228 / 445,
          totalCost: 8.73520228,
          costBasis: "PR calculation",
          tokensIn: "792.23M",
          tokensOut: "17.92M",
          date: "2026-08-02",
          attempts: "445 trials · PR #189",
          note:
            "fast-agent 0.9.30. Score and tokens aggregated from the two linked Harbor source jobs. Cost uses the calculated trial total recorded in the PR.",
          disclaimer: "Provisional result pending Terminal-Bench leaderboard review.",
          disclaimerLabel: "Provisional",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/189",
        },
        {
          harness: "Claude Code",
          model: "Sonnet 5 · high",
          score: 74.61,
          cost: 288.18 / 445,
          totalCost: 288.18,
          costBasis: "current leaderboard",
          tokensIn: "547.36M",
          tokensOut: "11.22M",
          date: "2026-07-09",
          attempts: "445 trials · published #98",
          note:
            "Score and complete-run cost from the current published Terminal-Bench 2.1 leaderboard.",
          url: "https://github.com/harbor-framework/terminal-bench-2-1/pull/98",
        },
      ],
    },
    {
      id: "value",
      axes: {
        score: { min: 80, max: 86, ticks: [80, 82, 84, 86] },
        cost: { min: 0.04, max: 0.3, ticks: [0.05, 0.1, 0.15, 0.25] },
      },
      label: "Value (6hr)",
      order: 1,
      claim:
        "82.2–84.5% for ~$30–$78 per run with Luna max, DeepSeek Vision, and GLM-5.3-Flash · 6hr timeout per trial.",
      runTotalNote:
        "~ marks estimated cost · 89 tasks × 5 trials.",
      stats: [
        { value: "82.2–84.5%", label: "445 trial slots per model" },
        { value: "~$29.54–$77.53", label: "estimated run cost" },
        { value: "6hr", label: "per-trial agent timeout" },
      ],
      results: [
        {
          fastAgent: true,
          harness: "fast-agent",
          label: "GLM-5.3-Flash",
          model: "GLM-5.3-Flash \u00b7 max",
          modelString: "zai.glm-5.3-flash?reasoning=max",
          score: 376 / 445 * 100,
          cost: 77.53332054 / 445,
          totalCost: 77.53332054,
          costEstimate: true,
          costBasis: "configured rates",
          tokensIn: "1802.45M",
          tokensOut: "23.97M",
          date: "2026-08-27",
          attempts: "445 trial slots · 6hr timeout",
          note: "fast-agent 0.10.11. 376/445 rewarded; source jobs and accounting in the 6hr timeout methodology.",
          disclaimer: "Non-standard 6hr timeout. Recorded cost coverage: 432/445 selected trials; not billed spend or a standard leaderboard submission.",
          disclaimerLabel: "6hr timeout",
          url: "https://hub.harborframework.com/jobs/aea81782-d18a-44ca-9417-9f6793342a1b",
        },
        {
          fastAgent: true,
          harness: "fast-agent",
          label: "DeepSeek Vision",
          model: "DeepSeek V4 Flash Vision Exp \u00b7 max",
          modelString: "deepseek.deepseek-v4-flash-vision-exp?reasoning=max",
          score: 370 / 445 * 100,
          cost: 53.68446646 / 445,
          totalCost: 53.68446646,
          costEstimate: true,
          costBasis: "configured rates",
          tokensIn: "1665.55M",
          tokensOut: "19.34M",
          date: "2026-08-28",
          attempts: "445 trial slots · 6hr timeout",
          note: "fast-agent 0.10.13. 370/445 rewarded; source jobs and accounting in the 6hr timeout methodology.",
          disclaimer: "Non-standard 6hr timeout. Recorded cost coverage: 386/445 selected trials; not billed spend or a standard leaderboard submission.",
          disclaimerLabel: "6hr timeout",
          url: "https://hub.harborframework.com/jobs/852e75c9-ca0e-4551-9393-21835a9857e2",
        },
        {
          fastAgent: true,
          harness: "fast-agent",
          winner: true,
          label: "Luna max",
          model: "GPT-5.6 Luna \u00b7 max",
          modelString: "codexresponses.gpt-5.6-luna?reasoning=max",
          score: 366 / 445 * 100,
          cost: 29.54135184 / 445,
          totalCost: 29.54135184,
          costEstimate: true,
          costBasis: "configured rates",
          tokensIn: "647.77M",
          tokensOut: "8.88M",
          date: "2026-08-28",
          attempts: "445 trial slots · 6hr timeout",
          note: "fast-agent 0.10.12. 366/445 rewarded; source jobs and accounting in the 6hr timeout methodology.",
          disclaimer: "Non-standard 6hr timeout. Recorded cost coverage: 424/445 selected trials; not billed spend or a standard leaderboard submission.",
          disclaimerLabel: "6hr timeout",
          url: "https://hub.harborframework.com/jobs/059f8d77-1de3-40ea-953e-3261675843cb",
        },
      ],
    },
    {
      id: "open",
      label: "Open + local",
      order: 3,
      visible: false,
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
