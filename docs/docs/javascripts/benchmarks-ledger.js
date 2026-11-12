/*
 * Filters and sort for the build-time ledger (docs/benchmark_ledger.py).
 * The table is complete without this; it only wires up the hidden controls
 * using each row's data-family / data-tier / data-score / data-cost / data-long.
 */
(function () {
  "use strict";

  function enhance(board) {
    if (board.dataset.ready) return;
    board.dataset.ready = "1";
    const controls = board.querySelector(".fb-controls");
    const table = board.querySelector(".fb-ledger");
    const empty = table.querySelector(".fb-empty");
    const items = [...table.querySelectorAll(".fb-group, .fb-row")]; // family order
    const rows = items.filter((n) => n.classList.contains("fb-row"));
    const state = { family: "all", sort: "family", leaderboard: true, claims: true, long: true };

    const shown = (r) =>
      (state.family === "all" || r.dataset.family === state.family) &&
      (state.leaderboard || r.dataset.tier !== "leaderboard") &&
      (state.claims || r.dataset.tier !== "claim") &&
      (state.long || !("long" in r.dataset));
    const sorts = {
      score: (a, b) => b.dataset.score - a.dataset.score,
      cost: (a, b) => a.dataset.cost - b.dataset.cost,
    };

    function apply() {
      rows.forEach((r) => (r.hidden = !shown(r)));
      const grouped = state.sort === "family";
      (grouped ? items : rows.slice().sort(sorts[state.sort])).forEach((n) => table.insertBefore(n, empty));
      table.querySelectorAll(".fb-group").forEach((g) => {
        g.hidden = !grouped || !rows.some((r) => !r.hidden && r.dataset.family === g.dataset.family);
      });
      empty.hidden = rows.some((r) => !r.hidden);
      controls.querySelectorAll("[data-family]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.family === state.family));
      controls.querySelectorAll("[data-sort]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.sort === state.sort));
    }

    controls.addEventListener("click", (evt) => {
      const button = evt.target.closest("button");
      if (!button) return;
      if (button.dataset.family) state.family = button.dataset.family;
      if (button.dataset.sort) state.sort = button.dataset.sort;
      apply();
    });
    controls.addEventListener("change", (evt) => {
      state[evt.target.dataset.toggle] = evt.target.checked;
      apply();
    });
    controls.hidden = false;
  }

  const start = () => document.querySelectorAll("[data-static-ledger]").forEach(enhance);
  if (window.document$) window.document$.subscribe(start);
  else start();
})();
