(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var COST_TICKS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 40, 50];

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function svgElement(tag, attributes, text) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attributes).forEach(function (name) {
      node.setAttribute(name, String(attributes[name]));
    });
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function scoreDomain(results, configured) {
    if (configured) return configured;
    var scores = results.map(function (result) { return result.score; });
    var low = Math.min.apply(null, scores);
    var high = Math.max.apply(null, scores);
    var padding = Math.max(1.4, (high - low) * 0.28);
    var min = Math.floor((low - padding) / 2) * 2;
    var max = Math.ceil((high + padding) / 2) * 2;
    var step = [1, 2, 4, 5, 10].find(function (candidate) {
      return (max - min) / candidate <= 5;
    }) || 10;
    var ticks = [];
    for (var tick = min; tick <= max; tick += step) ticks.push(tick);
    return { min: min, max: max, ticks: ticks };
  }

  function costDomain(results, configured) {
    if (configured) return configured;
    var costs = results.map(function (result) { return result.cost; });
    var min = Math.min.apply(null, costs) / 1.7;
    var max = Math.max.apply(null, costs) * 1.7;
    var ticks = COST_TICKS.filter(function (tick) { return tick >= min && tick <= max; });
    while (ticks.length > 5) ticks.splice(ticks.length % 2 ? 1 : 2, 1);
    return { min: min, max: max, ticks: ticks };
  }

  function formatCost(value) {
    return "$" + value.toFixed(2).replace(/\.?0+$/, "");
  }

  function formatRunCost(result, taskCount) {
    var total = result.totalCost !== undefined ? result.totalCost : result.cost * taskCount;
    return "$" + Math.round(total).toLocaleString("en-US");
  }

  function formatResultCost(result) {
    return (result.costEstimate ? "~" : "") + formatCost(result.cost);
  }

  function formatTableResultCost(result) {
    return (result.costEstimate ? "~" : "") + formatCost(result.cost);
  }

  function formatTableResultRunCost(result, taskCount) {
    return (result.costEstimate ? "~" : "") + formatRunCost(result, taskCount);
  }

  function overlapArea(first, second) {
    var width = Math.max(0, Math.min(first.right, second.right) - Math.max(first.left, second.left));
    var height = Math.max(0, Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top));
    return width * height;
  }

  function labelCandidates(result, pointX, pointY) {
    var width = Math.max(result.model.length * 5.7, result.harness.length * 5.1) + 8;
    var height = 24;
    return [
      {
        name: "top",
        x: pointX,
        y: pointY - 15,
        anchor: "middle",
        box: { left: pointX - width / 2, right: pointX + width / 2, top: pointY - 25, bottom: pointY - 1 },
        preference: 0,
      },
      {
        name: "bottom",
        x: pointX,
        y: pointY + 18,
        anchor: "middle",
        box: { left: pointX - width / 2, right: pointX + width / 2, top: pointY + 8, bottom: pointY + 8 + height },
        preference: 4,
      },
      {
        name: "right",
        x: pointX + 12,
        y: pointY - 4,
        anchor: "start",
        box: { left: pointX + 12, right: pointX + 12 + width, top: pointY - 14, bottom: pointY + 10 },
        preference: 8,
      },
      {
        name: "left",
        x: pointX - 12,
        y: pointY - 4,
        anchor: "end",
        box: { left: pointX - 12 - width, right: pointX - 12, top: pointY - 14, bottom: pointY + 10 },
        preference: 8,
      },
    ];
  }

  function placePointLabels(results, x, y, plot) {
    var points = results.map(function (result) {
      return { x: x(result.cost), y: y(result.score) };
    });
    var order = results.map(function (_, index) { return index; });
    order.sort(function (first, second) {
      function crowding(index) {
        return points.reduce(function (score, point, otherIndex) {
          if (index === otherIndex) return score;
          var dx = points[index].x - point.x;
          var dy = points[index].y - point.y;
          return score + 1 / Math.max(1, Math.sqrt(dx * dx + dy * dy));
        }, 0);
      }
      return crowding(second) - crowding(first);
    });

    var states = [{ score: 0, placements: [] }];
    order.forEach(function (resultIndex) {
      var result = results[resultIndex];
      var candidates = labelCandidates(result, points[resultIndex].x, points[resultIndex].y);
      if (result.labelPosition) {
        candidates = candidates.filter(function (candidate) {
          return candidate.name === result.labelPosition;
        });
      }
      var nextStates = [];
      states.forEach(function (state) {
        candidates.forEach(function (candidate) {
          var score = state.score + candidate.preference;
          var box = candidate.box;
          score += Math.max(0, plot.left + 2 - box.left) * 1000;
          score += Math.max(0, box.right - plot.right + 2) * 1000;
          score += Math.max(0, plot.top + 2 - box.top) * 1000;
          score += Math.max(0, box.bottom - plot.bottom + 2) * 1000;

          points.forEach(function (point) {
            if (
              point.x >= box.left - 5 && point.x <= box.right + 5 &&
              point.y >= box.top - 5 && point.y <= box.bottom + 5
            ) {
              score += 3000;
            }
          });
          state.placements.forEach(function (placement) {
            var area = overlapArea(box, placement.candidate.box);
            if (area > 0) score += 1000 + area * 50;
          });

          nextStates.push({
            score: score,
            placements: state.placements.concat([{ index: resultIndex, candidate: candidate }]),
          });
        });
      });
      nextStates.sort(function (first, second) { return first.score - second.score; });
      states = nextStates.slice(0, 128);
    });

    var placements = [];
    states[0].placements.forEach(function (placement) {
      placements[placement.index] = placement.candidate;
    });
    return placements;
  }

  function renderComparisonRows(comparison, data, sortOrder) {
    var panel = element("div", "fa-bench-sheet");
    var totals = comparison.results.map(function (entry) {
      return entry.totalCost !== undefined ? entry.totalCost : entry.cost * data.taskCount;
    });
    var maxCost = Math.max.apply(null, totals);
    var costStep = Math.pow(10, Math.floor(Math.log10(maxCost)));
    var costCeiling = Math.ceil(maxCost / costStep) * costStep;
    var headings = element("div", "fa-bench-sheet__head");
    headings.appendChild(element("span", "", "Model / harness"));
    var accuracyHeading = element("span", "", "Accuracy");
    accuracyHeading.appendChild(element("small", "", "0–100% · higher is better"));
    headings.appendChild(accuracyHeading);
    var costHeading = element("span", "", "Cost");
    costHeading.appendChild(element("small", "", "$0–$" + costCeiling + " total · lower is better"));
    headings.appendChild(costHeading);
    panel.appendChild(headings);
    var entries = comparison.results.map(function (entry, index) {
      return { entry: entry, total: totals[index], index: index };
    });
    entries.sort(function (first, second) {
      return sortOrder === "cost" ? first.total - second.total : second.entry.score - first.entry.score;
    });
    entries.forEach(function (item) {
      var entry = item.entry;
      var row = element("article", "fa-bench-sheet__row");
      row.style.setProperty("--run-color", ["var(--amber)", "var(--bench-teal)", "var(--orange)"][item.index % 3]);
      var cells = element("div", "fa-bench-sheet__cells");
      var identity = element("div", "fa-bench-sheet__identity");
      identity.appendChild(element("h3", "", entry.model));
      identity.appendChild(element("p", "", entry.harness + " · " + entry.date));
      identity.appendChild(element("small", "", entry.attempts));
      cells.appendChild(identity);
      var accuracy = element("div", "fa-bench-sheet__metric");
      accuracy.appendChild(element("span", "fa-bench-sheet__mobile-label", "Accuracy"));
      accuracy.appendChild(element("strong", "fa-bench-sheet__value", entry.score.toFixed(1) + "%"));
      var scoreBar = element("span", "fa-bench-sheet__track");
      scoreBar.setAttribute("aria-hidden", "true");
      var scoreFill = element("span");
      scoreFill.style.width = entry.score + "%";
      scoreBar.appendChild(scoreFill);
      accuracy.appendChild(scoreBar);
      cells.appendChild(accuracy);
      var cost = element("div", "fa-bench-sheet__metric");
      cost.appendChild(element("span", "fa-bench-sheet__mobile-label", "Total cost"));
      cost.appendChild(element("strong", "fa-bench-sheet__value", (entry.costEstimate ? "~" : "") + "$" + item.total.toFixed(2)));
      var costBar = element("span", "fa-bench-sheet__track");
      costBar.setAttribute("aria-hidden", "true");
      var costFill = element("span");
      costFill.style.width = item.total / costCeiling * 100 + "%";
      costBar.appendChild(costFill);
      cost.appendChild(costBar);
      cost.appendChild(element("small", "", formatResultCost(entry) + " / trial"));
      cells.appendChild(cost);
      row.appendChild(cells);
      var evidence = element("details", "fa-bench-sheet__evidence");
      evidence.appendChild(element("summary", "", "Run evidence & accounting"));
      if (entry.note) evidence.appendChild(element("p", "", entry.note));
      if (entry.disclaimer) evidence.appendChild(element("p", "", entry.disclaimer));
      evidence.appendChild(element("p", "", entry.tokensIn + " input tokens · " + entry.tokensOut + " output tokens" + (entry.costBasis ? " · " + entry.costBasis : "")));
      var source = element("a", "", "Open source run ↗");
      source.href = entry.url || data.sourceUrl;
      source.target = "_blank";
      source.rel = "noopener";
      evidence.appendChild(source);
      row.appendChild(evidence);
      panel.appendChild(row);
    });
    var note = element("p", "fa-bench-sheet__note", comparison.runTotalNote || "~ denotes estimated cost. Totals use submitted costs where available; otherwise cost per trial × " + data.taskCount + ".");
    panel.appendChild(note);
    return panel;
  }

  function render(root, data) {
    var comparisons = data.comparisons.filter(function (comparison) {
      return comparison.visible !== false;
    });
    comparisons.sort(function (first, second) { return first.order - second.order; });
    var sixHour = comparisons.find(function (comparison) { return comparison.id === "value"; });
    var archivedComparisons = comparisons.filter(function (comparison) { return comparison.id !== "value"; });
    comparisons = [
      { id: "terminal-bench-4-subset", label: "Terminal-Bench 4-subset", results: [] },
      Object.assign({}, sixHour, { label: "Terminal-Bench 2.1 (6hr)" }),
      { id: "deepswe-1-1", label: "DeepSWE 1.1", results: [] }
    ].concat(archivedComparisons);
    var selectedComparison = 1;
    var selectedResult = 0;
    var displayMode = "rows";
    var sortOrder = "accuracy";
    var tablePreviewResult = -1;

    function selectResult(index) {
      if (index === selectedResult) return;
      selectedResult = index;
      draw();
    }

    function previewResult(index, active) {
      if (!active && tablePreviewResult !== index) return;
      var nextPreview = active ? index : -1;
      if (tablePreviewResult === nextPreview) return;
      tablePreviewResult = nextPreview;
      root.querySelectorAll(".fa-benchmark__point[data-result-index]").forEach(function (point) {
        point.classList.toggle(
          "is-table-preview",
          Number(point.getAttribute("data-result-index")) === tablePreviewResult
        );
      });
    }

    function draw() {
      var comparison = comparisons[selectedComparison];


      root.replaceChildren();
      root.setAttribute("aria-label", data.title + " benchmark comparisons");

      var comparisonBar = element("div", "fa-benchmark__series");
      var tabs = element("div", "fa-benchmark__tabs");
      tabs.setAttribute("aria-label", "Benchmark series");
      comparisons.slice(0, 3).forEach(function (item, index) {
        var tab = element("button", "fa-benchmark__tab");
        tab.type = "button";
        tab.setAttribute("aria-pressed", String(index === selectedComparison));
        tab.appendChild(element("strong", "", item.label));
        tab.appendChild(element("small", "", item.results.length ? item.results.length + " runs · explore results" : "Results to follow"));
        tab.addEventListener("click", function () {
          selectedComparison = index;
          selectedResult = 0;
          tablePreviewResult = -1;
          draw();
          root.querySelectorAll(".fa-benchmark__tab")[index].focus({ preventScroll: true });
        });
        tabs.appendChild(tab);
      });
      comparisonBar.appendChild(tabs);
      root.appendChild(comparisonBar);

      var toolbar = element("div", "fa-benchmark__toolbar");
      toolbar.appendChild(element("span", "fa-benchmark__eyebrow", "Accuracy / cost"));
      var archiveLabel = element("label", "fa-benchmark__archive", "Earlier comparisons ");
      var archive = element("select");
      archive.appendChild(element("option", "", "Browse archive"));
      archivedComparisons.forEach(function (item, index) {
        var option = element("option", "", item.label);
        option.value = String(index + 3);
        option.selected = selectedComparison === index + 3;
        archive.appendChild(option);
      });
      archive.addEventListener("change", function () {
        if (archive.selectedIndex === 0) return;
        selectedComparison = Number(archive.value);
        selectedResult = 0;
        tablePreviewResult = -1;
        draw();
        root.querySelector("select").focus({ preventScroll: true });
      });
      archiveLabel.appendChild(archive);
      toolbar.appendChild(archiveLabel);
      root.appendChild(toolbar);

      if (!comparison.results.length) {
        var empty = element("div", "fa-benchmark__empty");
        empty.appendChild(element("span", "fa-benchmark__empty-mark", "↗"));
        empty.appendChild(element("h2", "", comparison.label));
        empty.appendChild(element("p", "", "Results and source evidence will appear here when published."));
        root.appendChild(empty);
        return;
      }
      var result = comparison.results[selectedResult] || comparison.results[0];
      var score = scoreDomain(comparison.results, comparison.axes && comparison.axes.score);
      var cost = costDomain(comparison.results, comparison.axes && comparison.axes.cost);
      var heading = element("div", "fa-benchmark__intro");
      heading.appendChild(element("h2", "", comparison.label));
      heading.appendChild(element("p", "", selectedComparison === 1
        ? "89 tasks · 5 trials per task · six-hour agent timeout. Expand a row for source evidence and cost coverage."
        : comparison.claim));
      root.appendChild(heading);

      var viewControls = element("div", "fa-bench-views");
      var viewButtons = element("div", "fa-bench-views__buttons");
      viewButtons.setAttribute("aria-label", "Chart view");
      ["rows", "scatter"].forEach(function (mode) {
        var button = element("button", "", mode === "rows" ? "Comparison rows" : "Accuracy / cost plot");
        button.type = "button";
        button.setAttribute("aria-pressed", String(displayMode === mode));
        button.addEventListener("click", function () {
          displayMode = mode;
          draw();
          root.querySelectorAll(".fa-bench-views__buttons button")[mode === "rows" ? 0 : 1].focus({ preventScroll: true });
        });
        viewButtons.appendChild(button);
      });
      viewControls.appendChild(viewButtons);
      if (displayMode === "rows") {
        var sortLabel = element("label", "", "Sort by ");
        var sortSelect = element("select");
        [["accuracy", "Accuracy"], ["cost", "Total cost"]].forEach(function (optionData) {
          var option = element("option", "", optionData[1]);
          option.value = optionData[0];
          option.selected = sortOrder === optionData[0];
          sortSelect.appendChild(option);
        });
        sortSelect.addEventListener("change", function () {
          sortOrder = sortSelect.value;
          draw();
          root.querySelector(".fa-bench-views select").focus({ preventScroll: true });
        });
        sortLabel.appendChild(sortSelect);
        viewControls.appendChild(sortLabel);
      }
      root.appendChild(viewControls);
      if (displayMode === "rows") {
        root.appendChild(renderComparisonRows(comparison, data, sortOrder));
        return;
      }

      var body = element("div", "fa-benchmark__body");
      var chartArea = element("div", "fa-benchmark__chart-area");

      var svg = svgElement("svg", {
        "class": "fa-benchmark__chart",
        "viewBox": "0 0 680 350",
        "role": "img",
        "aria-label": comparison.label +
          ": Terminal-Bench score versus cost per task, with lower cost farther right",
      });
      var plot = { left: 72, right: 650, top: 22, bottom: 300 };
      var width = plot.right - plot.left;
      var height = plot.bottom - plot.top;
      var x = function (value) {
        var ratio = (Math.log10(value) - Math.log10(cost.min)) /
          (Math.log10(cost.max) - Math.log10(cost.min));
        return plot.right - ratio * width;
      };
      var y = function (value) {
        var ratio = (value - score.min) / (score.max - score.min);
        return plot.bottom - ratio * height;
      };
      var labelPlacements = placePointLabels(comparison.results, x, y, plot);

      cost.ticks.forEach(function (tick) {
        var tickX = x(tick);
        svg.appendChild(svgElement("line", { x1: tickX, y1: plot.top, x2: tickX, y2: plot.bottom, "class": "fa-benchmark__grid" }));
        svg.appendChild(svgElement("text", { x: tickX, y: 325, "class": "fa-benchmark__tick", "text-anchor": "middle" }, formatCost(tick)));
      });
      score.ticks.forEach(function (tick) {
        var tickY = y(tick);
        svg.appendChild(svgElement("line", { x1: plot.left, y1: tickY, x2: plot.right, y2: tickY, "class": "fa-benchmark__grid" }));
        svg.appendChild(svgElement("text", { x: 62, y: tickY + 4, "class": "fa-benchmark__tick", "text-anchor": "end" }, tick + "%"));
      });
      svg.appendChild(svgElement("line", { x1: plot.left, y1: plot.top, x2: plot.left, y2: plot.bottom, "class": "fa-benchmark__axis" }));
      svg.appendChild(svgElement("line", { x1: plot.left, y1: plot.bottom, x2: plot.right, y2: plot.bottom, "class": "fa-benchmark__axis" }));
      svg.appendChild(svgElement("text", {
        x: 17, y: 165, "class": "fa-benchmark__axis-label", transform: "rotate(-90 17 165)", "text-anchor": "middle",
      }, data.title.toUpperCase() + " SCORE (%)"));
      svg.appendChild(svgElement("text", {
        x: (plot.left + plot.right) / 2,
        y: 347,
        "class": "fa-benchmark__axis-label",
        "text-anchor": "middle",
      }, "COST / TASK (USD) · CHEAPER →"));

      comparison.results.forEach(function (entry, index) {
        var pointX = x(entry.cost);
        var pointY = y(entry.score);
        var labelPlacement = labelPlacements[index];
        var group = svgElement("g", {
          "class": "fa-benchmark__point" +
            (entry.fastAgent ? " fa-benchmark__point--fast-agent" : "") +
            (entry.winner ? " fa-benchmark__point--winner" : "") +
            (index === selectedResult ? " is-selected" : "") +
            (index === tablePreviewResult ? " is-table-preview" : ""),
          "role": "button",
          "tabindex": "0",
          "data-result-index": index,
          "aria-label": entry.model + " on " + entry.harness + ", score " + entry.score +
            " percent, " + formatResultCost(entry) + " per task",
        });
        var activate = function () { selectResult(index); };
        var previewRow = function (active) {
          root.classList.toggle("is-point-previewing", active);
          root.querySelectorAll(".fa-benchmark__result[data-result-index]").forEach(function (row) {
            row.classList.toggle(
              "is-preview",
              active && Number(row.getAttribute("data-result-index")) === index
            );
          });
        };
        var promote = function () {
          if (!group.parentNode || group.parentNode.lastElementChild === group) return;
          var restoreFocus = document.activeElement === group;
          group.parentNode.appendChild(group);
          if (restoreFocus) group.focus({ preventScroll: true });
        };
        group.addEventListener("mouseenter", function () {
          promote();
          previewRow(true);
        });
        group.addEventListener("mouseleave", function () { previewRow(false); });
        group.addEventListener("focus", function () {
          promote();
          previewRow(true);
        });
        group.addEventListener("blur", function () { previewRow(false); });
        group.addEventListener("click", activate);
        group.addEventListener("keydown", function (event) {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            activate();
          }
        });
        group.appendChild(svgElement("circle", { cx: pointX, cy: pointY, r: entry.fastAgent ? 7 : 5 }));

        var pointLabel = svgElement("text", {
          x: labelPlacement.x,
          y: labelPlacement.y,
          "class": "fa-benchmark__point-label",
          "text-anchor": labelPlacement.anchor,
        });
        pointLabel.appendChild(svgElement("tspan", {
          x: labelPlacement.x, "class": "fa-benchmark__point-harness",
        }, entry.harness));
        pointLabel.appendChild(svgElement("tspan", {
          x: labelPlacement.x, dy: 11, "class": "fa-benchmark__point-model",
        }, entry.model));
        group.appendChild(pointLabel);

        var tooltipX = pointX > plot.right - 210 ? pointX - 202 : pointX + 12;
        var tooltipY = pointY < plot.top + 92 ? pointY + 14 : pointY - 88;
        var tooltip = svgElement("g", {
          "class": "fa-benchmark__tooltip",
          transform: "translate(" + tooltipX + " " + tooltipY + ")",
          "aria-hidden": "true",
        });
        tooltip.appendChild(svgElement("rect", { width: 190, height: 76, rx: 2 }));
        tooltip.appendChild(svgElement("text", {
          x: 10, y: 18, "class": "fa-benchmark__tooltip-model",
        }, entry.model));
        tooltip.appendChild(svgElement("text", {
          x: 10, y: 34, "class": "fa-benchmark__tooltip-harness",
        }, entry.harness));
        tooltip.appendChild(svgElement("text", {
          x: 10, y: 58, "class": "fa-benchmark__tooltip-score",
        }, entry.score.toFixed(1) + "% score"));
        tooltip.appendChild(svgElement("text", {
          x: 180, y: 58, "class": entry.winner
            ? "fa-benchmark__tooltip-cost fa-benchmark__tooltip-cost--winner"
            : "fa-benchmark__tooltip-cost",
          "text-anchor": "end",
        }, formatResultCost(entry) + "/task"));
        group.appendChild(tooltip);
        svg.appendChild(group);
      });
      chartArea.appendChild(svg);

      var detail = element("div", "fa-benchmark__detail");
      var detailMain = element("div", "fa-benchmark__detail-main");
      detailMain.appendChild(element("strong", "", result.harness));
      detailMain.appendChild(element("span", "", result.model));
      detailMain.appendChild(element("span", "", result.date + " · " + result.attempts));
      if (result.disclaimer) {
        var badge = element(
          "span",
          "fa-benchmark__status-badge",
          result.disclaimerLabel || "Adjusted"
        );
        badge.title = result.disclaimer;
        badge.setAttribute("aria-label", badge.textContent + ": " + result.disclaimer);
        detailMain.appendChild(badge);
      }
      var runLink = element("a", "", "Source evidence ↗");
      runLink.href = result.url || data.sourceUrl;
      runLink.target = "_blank";
      runLink.rel = "noopener";
      detailMain.appendChild(runLink);
      detail.appendChild(detailMain);
      var metrics = element("p", "fa-benchmark__metrics");
      metrics.appendChild(document.createTextNode(result.score.toFixed(1) + "% score · "));
      metrics.appendChild(element(
        "span",
        result.winner ? "fa-benchmark__winning-cost" : "",
        formatResultCost(result) + "/task"
      ));
      if (result.costBasis) {
        metrics.appendChild(document.createTextNode(" (" + result.costBasis + ")"));
      }
      metrics.appendChild(document.createTextNode(
        " · " + result.tokensIn + " tokens in · " + result.tokensOut + " tokens out"
      ));
      detail.appendChild(metrics);
      chartArea.appendChild(detail);
      body.appendChild(chartArea);

      var results = element("div", "fa-benchmark__results");
      results.appendChild(element("h2", "fa-benchmark__results-title", "Compare runs"));
      var resultHeader = element("div", "fa-benchmark__result fa-benchmark__result--header");
      ["Model / harness", "Score", "$/task", "Total $"].forEach(function (label) {
        resultHeader.appendChild(element("span", "", label));
      });
      results.appendChild(resultHeader);
      comparison.results.forEach(function (entry, index) {
        var row = element("button", "fa-benchmark__result" +
          (entry.fastAgent ? " fa-benchmark__result--fast-agent" : "") +
          (index === selectedResult ? " is-selected" : ""));
        row.type = "button";
        row.setAttribute("data-result-index", String(index));
        var name = element("span", "fa-benchmark__result-name");
        name.appendChild(element("strong", "", entry.model));
        name.appendChild(element("small", "", entry.harness));
        row.appendChild(name);
        row.appendChild(element("span", "fa-benchmark__result-score", entry.score.toFixed(1)));
        row.appendChild(element(
          "span",
          entry.winner
            ? "fa-benchmark__result-cost fa-benchmark__result-cost--winner"
            : "fa-benchmark__result-cost",
          formatTableResultCost(entry)
        ));
        row.appendChild(element("span", "", formatTableResultRunCost(entry, data.taskCount)));
        row.addEventListener("mouseenter", function () { previewResult(index, true); });
        row.addEventListener("mouseleave", function () { previewResult(index, false); });
        row.addEventListener("focus", function () { previewResult(index, true); });
        row.addEventListener("blur", function () { previewResult(index, false); });
        row.addEventListener("click", function () { selectResult(index); });
        results.appendChild(row);
      });
      var hasEstimatedCosts = comparison.results.some(function (entry) {
        return entry.costEstimate;
      });
      var runTotalNote = comparison.runTotalNote || (hasEstimatedCosts
        ? "~ marks Sol estimates at current pricing; other totals use submitted totals when " +
          "provided or cost/task × " + data.taskCount + " tasks"
        : "Run total = submitted total when provided; otherwise cost/task × " +
          data.taskCount + " tasks");
      results.appendChild(element("p", "fa-benchmark__run-total", runTotalNote));
      body.appendChild(results);
      root.appendChild(body);

      var footer = element("div", "fa-benchmark__footer");
      footer.appendChild(element("span", "fa-benchmark__legend fa-benchmark__legend--fast-agent", "fast-agent"));
      footer.appendChild(element("span", "fa-benchmark__legend", "other harnesses"));
      footer.appendChild(element("span", "", "Higher and further right is better."));
      var methodology = element("a", "", "Methodology & disclaimers");
      methodology.href = "#benchmark-notes";
      footer.appendChild(methodology);
      root.appendChild(footer);


    }

    draw();
  }

  function start() {
    var root = document.querySelector("[data-fa-benchmark]");
    if (root && window.fastAgentBenchmark) render(root, window.fastAgentBenchmark);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(start);
})();
