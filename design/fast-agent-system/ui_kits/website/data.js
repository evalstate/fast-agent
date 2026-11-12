// Subset of docs/docs/javascripts/homepage-benchmark-data.js (evalstate/fast-agent). Sol costs use current (−20%) pricing, as upstream.
(() => {
  const sol = c => c * 0.8;
  window.FA_BENCH = {
    title: 'Terminal-Bench 2.1', date: 'September 2026', trials: 445,
    comparisons: [
      { id: 'frontier', label: 'Frontier',
        claim: 'fast-agent + GPT-5.6 Sol high scores 88.3%: 4.5 points above Claude Code + Fable 5, at 61% lower estimated API cost with current Sol pricing.',
        x: { min: 0.2, max: 1.35, ticks: [0.25, 0.5, 0.75, 1, 1.25] }, y: { min: 76, max: 90, ticks: [76, 80, 84, 88] },
        results: [
          { fa: true, winner: true, harness: 'fast-agent', model: 'GPT-5.6 Sol · high', score: 88.31, cost: sol(270.13526 / 445), est: true, tokens: '122.32M / 3.55M', date: '2026-07-26', attempts: '445 trials · PR #174', badge: 'Provisional', note: 'fast-agent 0.9.24. Estimated cost applies Sol’s 20% August 2026 price reduction to the original $270.14 source-job total.', lp: 'top' },
          { fa: true, harness: 'fast-agent', model: 'Grok 4.6 · medium', score: 390 / 445 * 100, cost: 193.4 / 445, tokens: '233.46M / 6.25M', date: '2026-08-24', attempts: '445 trials · PR #221', badge: 'Provisional', note: 'fast-agent 0.10.10. Cost and tokens aggregated from six linked Harbor source jobs.', lp: 'left' },
          { fa: true, harness: 'fast-agent', model: 'Grok 4.6 · high', score: 388 / 445 * 100, cost: 238.316166 / 445, tokens: '284.77M / 8.38M', date: '2026-08-16', attempts: '445 trials · PR #212', badge: 'Provisional', note: 'fast-agent 0.10.9. Cost and tokens aggregated from six linked Harbor source jobs.', lp: 'right' },
          { harness: 'Claude Code', model: 'Fable 5 · xhigh', score: 83.82, cost: 1.241955, tokens: '194.55M / 9.95M', date: '2026-06-07', attempts: '445 trials · published', note: 'Published Terminal-Bench 2.1 leaderboard row.', lp: 'left' },
          { fa: true, harness: 'fast-agent', model: 'GPT-5.6 Sol · medium', score: 365 / 445 * 100, cost: sol(211.321445 / 445), est: true, tokens: '101.95M / 2.53M', date: '2026-07-23', attempts: '445 trials · PR #170', badge: 'Provisional', note: 'fast-agent 0.9.21. Estimated cost applies Sol’s 20% price reduction to the original $211.32 total.', lp: 'right' },
          { harness: 'Claude Code', model: 'Opus 4.8 · high', score: 78.88, cost: 0.644809, tokens: '174.81M / 8.09M', date: '2026-07-09', attempts: '445 trials · published', note: 'Published Terminal-Bench 2.1 leaderboard row.', lp: 'right' },
        ] },
      { id: 'value', label: 'Value (6hr)',
        claim: '82.2–84.5% for ~$30–$78 per run with Luna max, DeepSeek Vision and GLM-5.3-Flash, on a six-hour timeout per trial.',
        x: { min: 0.04, max: 0.2, ticks: [0.05, 0.1, 0.15, 0.2] }, y: { min: 81, max: 86, ticks: [81, 82, 83, 84, 85, 86] },
        results: [
          { fa: true, harness: 'fast-agent', model: 'GLM-5.3-Flash · max', score: 376 / 445 * 100, cost: 77.53332054 / 445, total: 77.53, est: true, tokens: '1802.45M / 23.97M', date: '2026-08-27', attempts: '445 trial slots · 6hr timeout', badge: '6hr timeout', note: 'fast-agent 0.10.11. 376/445 rewarded. Recorded cost coverage 432/445.', lp: 'left' },
          { fa: true, harness: 'fast-agent', model: 'DeepSeek V4 Flash Vision Exp · max', score: 370 / 445 * 100, cost: 53.68446646 / 445, total: 53.68, est: true, tokens: '1665.55M / 19.34M', date: '2026-08-28', attempts: '445 trial slots · 6hr timeout', badge: '6hr timeout', note: 'fast-agent 0.10.13. 370/445 rewarded. Recorded cost coverage 386/445.', lp: 'right' },
          { fa: true, winner: true, harness: 'fast-agent', model: 'GPT-5.6 Luna · max', score: 366 / 445 * 100, cost: 29.54135184 / 445, total: 29.54, est: true, tokens: '647.77M / 8.88M', date: '2026-08-28', attempts: '445 trial slots · 6hr timeout', badge: '6hr timeout', note: 'fast-agent 0.10.12. 366/445 rewarded. Recorded cost coverage 424/445.', lp: 'right' },
        ] },
      { id: 'gpt56', label: 'GPT-5.6',
        claim: 'Across three matched GPT-5.6 settings, fast-agent beats OpenAI’s published scores and costs less per task at like-for-like pricing.',
        x: { min: 0.2, max: 1.0, ticks: [0.25, 0.5, 0.75, 1] }, y: { min: 74, max: 90, ticks: [74, 78, 82, 86, 90] },
        results: [
          { fa: true, winner: true, harness: 'fast-agent', model: 'GPT-5.6 Sol · high', score: 88.31, cost: sol(270.13526 / 445), est: true, tokens: '122.32M / 3.55M', date: '2026-07-26', attempts: '445 trials · PR #174', badge: 'Provisional', note: 'fast-agent 0.9.24.', lp: 'top' },
          { harness: 'OpenAI', model: 'GPT-5.6 Sol · high', score: 84.7, cost: sol(1.09), est: true, tokens: '—', date: '2026-07-30', attempts: 'OpenAI score · repriced cost', note: 'Score and original $1.09 per task from OpenAI’s launch chart, repriced −20%.', lp: 'left' },
          { fa: true, winner: true, harness: 'fast-agent', model: 'GPT-5.6 Sol · medium', score: 365 / 445 * 100, cost: sol(211.321445 / 445), est: true, tokens: '101.95M / 2.53M', date: '2026-07-23', attempts: '445 trials · PR #170', badge: 'Provisional', note: 'fast-agent 0.9.21.', lp: 'right' },
          { harness: 'OpenAI', model: 'GPT-5.6 Sol · medium', score: 81.8, cost: sol(0.89), est: true, tokens: '—', date: '2026-07-30', attempts: 'OpenAI score · repriced cost', note: 'Score and original $0.89 per task from OpenAI’s launch chart, repriced −20%.', lp: 'left' },
          { fa: true, winner: true, harness: 'fast-agent', model: 'GPT-5.6 Terra · high', score: 77.75, cost: 133.81 / 445, tokens: '171.23M / 3.45M', date: '2026-07-18', attempts: '445 trials · PR #160', badge: 'Provisional', note: 'Accuracy, cost and tokens from the submission’s static analysis.', lp: 'right' },
          { harness: 'OpenAI', model: 'GPT-5.6 Terra · high', score: 76.67, cost: 0.63, tokens: '—', date: '2026-07-30', attempts: 'OpenAI published', note: 'Score and API cost per task from OpenAI’s Terminal-Bench 2.1 cost chart.', lp: 'right' },
        ] },
    ],
  };
})();
