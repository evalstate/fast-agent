(function () {
  "use strict";

  var CARDS = {
    spotlight: {
      focusModel: "GPT-5.6 Luna · max",
      identity: "luna max",
      contextModels: [
        "Grok 4.6 · high",
        "Fable 5 · xhigh",
        "GPT-5.6 Luna · max",
        "Opus 4.8 · high",
      ],
      trialQualifier: "flex pricing",
      spendCopy: "Operator-recorded API-key spend · 362 rewarded trials out of 445.",
      footerNote: "luna cost: operator-recorded API-key spend",
      footerAccent: "PR #184 · pending leaderboard review",
      className: "",
    },
    grok: {
      focusModel: "Grok 4.6 · high",
      identity: "grok 4.6 high",
      contextModels: [
        "GPT-5.6 Sol · high",
        "Grok 4.6 · high",
        "Fable 5 · xhigh",
        "Opus 4.8 · high",
      ],
      trialQualifier: "source-job cost",
      spendCopy: "388 rewarded trials out of 445.",
      footerNote: "Grok cost: aggregated configured source-job cost",
      footerAccent: "PR #212 · static analysis passed",
      className: " spotlight--grok",
      swapCostEmphasis: true,
      primaryCostDigits: 2,
    },
    deepseek: {
      focusModel: "DeepSeek V4 Flash · max",
      identity: "deepseek v4 flash 0731",
      contextModels: [
        "Fable 5 · high",
        "Opus 4.8 · high",
        "GPT-5.6 Terra · max",
        "DeepSeek V4 Flash · max",
        "Sonnet 5 · high",
      ],
      trialQualifier: "PR calculation",
      spendCopy: "PR-calculated trial cost · 336 rewarded trials out of 445.",
      footerNote: "DeepSeek cost: calculated trial total in PR #189",
      footerAccent: "PR #189 · pending leaderboard review",
      className: " spotlight--deepseek",
    },
  };

  function svgElement(tag, attributes, text) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attributes).forEach(function (name) {
      node.setAttribute(name, String(attributes[name]));
    });
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function runTotal(result, taskCount) {
    return result.totalCost !== undefined ? result.totalCost : result.cost * taskCount;
  }

  function formatExactTotal(value) {
    return "$" + value.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }

  function formatTrial(value) {
    return "$" + value.toFixed(3);
  }

  function displayName(result) {
    if (result.model.indexOf("Luna") >= 0) return "luna max";
    if (result.model.indexOf("Fable") === 0) return "Fable 5";
    if (result.model.indexOf("DeepSeek") === 0) return "DeepSeek V4 Flash 0731";
    return result.model.replace(" · ", " ");
  }

  function allResults() {
    var data = window.fastAgentBenchmark;
    return data.comparisons.reduce(function (results, comparison) {
      return results.concat(comparison.results);
    }, []);
  }

  function cardResults(config) {
    var data = window.fastAgentBenchmark;
    var results = allResults();
    return config.contextModels.map(function (model) {
      var result = results.find(function (entry) { return entry.model === model; });
      return {
        fastAgent: Boolean(result.fastAgent),
        model: result.model,
        modelString: result.modelString || "",
        name: displayName(result),
        harness: result.harness,
        status: result.disclaimer ? "Provisional" : "Published",
        score: result.score,
        costPerTrial: result.cost,
        total: runTotal(result, data.taskCount),
      };
    });
  }

  function masthead(statusCopy, brandSource) {
    return [
      '<header class="masthead">',
      '<img class="brand" src="' + (brandSource || "../wordmark.svg") + '" alt="fast-agent">',
      '<span class="benchmark-name">Terminal-Bench 2.1</span>',
      '<span class="status">' + (statusCopy || "fast-agent result · provisional") + "</span>",
      "</header>",
    ].join("");
  }

  function footer(config) {
    return [
      '<footer class="footer">',
      '<span class="primary">445 trials · complete run cost</span>',
      "<span>" + config.footerNote + "</span>",
      '<span class="spacer"></span>',
      '<span class="accent">' + config.footerAccent + "</span>",
      "</footer>",
    ].join("");
  }

  function contextRows(results, focus) {
    return results.map(function (result) {
      var classes = "context-row" + (result.model === focus.model ? " featured" : "");
      return [
        '<div class="' + classes + '">',
        "<div><strong>" + result.name + "</strong><small>" + result.harness +
          ' · <span class="run-status ' + result.status.toLowerCase() + '">' +
          result.status + "</span></small></div>",
        '<span class="cost">' + formatExactTotal(result.total) + "</span>",
        '<span class="score">' + result.score.toFixed(1) + "%</span>",
        '<span class="trial-cost">' + formatTrial(result.costPerTrial) + "</span>",
        "</div>",
      ].join("");
    }).join("");
  }

  function spotlightMain(focus, config) {
    var primaryCost = config.swapCostEmphasis
      ? "$" + focus.costPerTrial.toFixed(config.primaryCostDigits || 3)
      : formatExactTotal(focus.total);
    var primaryCostLabel = config.swapCostEmphasis ? "cost / trial" : "complete run cost";
    var supportingCost = config.swapCostEmphasis
      ? formatExactTotal(focus.total)
      : formatTrial(focus.costPerTrial);
    var supportingCostLabel = config.swapCostEmphasis ? "complete run cost" : "/ trial";
    return [
      '<section class="spotlight-main">',
      '<div class="spotlight-identity"><strong>' + config.identity +
        "</strong><span>fast-agent</span></div>",
      '<div class="spotlight-primary">',
      '<p class="spotlight-primary-score"><strong>' + focus.score.toFixed(1) +
        "%</strong><span>score</span></p>",
      '<span class="spotlight-primary-divider">|</span>',
      '<p class="spotlight-primary-cost"><strong>' + primaryCost +
        "</strong><span>" + primaryCostLabel + "</span></p>",
      "</div>",
      '<p class="spotlight-trial"><strong>' + supportingCost +
        "</strong> " + supportingCostLabel + " <span>· " +
        config.trialQualifier + "</span></p>",
      '<p class="spotlight-copy">' + config.spendCopy + "</p>",
      '<code class="spotlight-model">' + focus.modelString + "</code>",
      '<p class="spotlight-benchmark">Terminal-Bench <strong>2.1</strong></p>',
      "</section>",
    ].join("");
  }

  function renderSpotlight(root, config) {
    var results = cardResults(config);
    var focus = results.find(function (result) { return result.model === config.focusModel; });
    root.innerHTML = [
      '<article class="card spotlight' + config.className + '">',
      masthead(),
      '<div class="content">',
      spotlightMain(focus, config),
      '<section class="spotlight-context">',
      '<div class="context-header"><span>Model / harness</span><span>Cost</span>' +
        "<span>Score</span><span>$/trial</span></div>",
      contextRows(results, focus),
      "</section>",
      "</div>",
      footer(config),
      "</article>",
    ].join("");
  }

  function renderPricing(root) {
    var data = window.fastAgentBenchmark;
    var results = allResults();
    var luna = results.find(function (result) {
      return result.model === "GPT-5.6 Luna · max";
    });
    var deepseekOffPeak = (
      782508928 * 0.007 +
      9723987 * 0.22 +
      17923720 * 0.66
    ) / 1000000;
    var deepseekPrevious = (
      782508928 * 0.0028 +
      9723987 * 0.14 +
      17923720 * 0.28
    ) / 1000000;
    var series = [
      {
        className: "openai",
        name: "GPT-5.6 Luna",
        normal: runTotal(luna, data.taskCount) * 2,
        special: runTotal(luna, data.taskCount),
        xOffset: -10,
      },
      {
        className: "deepseek",
        name: "DeepSeek V4 Flash 0731",
        normal: deepseekOffPeak * 2,
        special: deepseekOffPeak,
        xOffset: 10,
      },
    ];
    root.innerHTML = [
      '<article class="card pricing-chart">',
      masthead(
        "pricing comparison",
        "../../docs/assets/brand/fast-agent-lockup-light.svg"
      ),
      '<div class="content">',
      '<div class="pricing-heading">',
      '<p class="pricing-kicker">Terminal-Bench 2.1 full run cost</p>',
      '<div class="pricing-legend">',
      '<span class="openai">OpenAI · GPT-5.6 Luna</span>',
      '<span class="deepseek">DeepSeek V4 Flash 0731</span>',
      "</div>",
      "</div>",
      '<svg viewBox="0 0 1100 430" role="img" aria-label="Normal pricing versus ' +
        'Flex and off-peak pricing"></svg>',
      "</div>",
      '<footer class="footer"><span class="primary">445 trials · complete run cost</span>',
      '<span>OpenAI Flex · DeepSeek off-peak</span>',
      '<span class="spacer"></span><span>PR #184 · PR #189</span>',
      "</footer>",
      "</article>",
    ].join("");

    var svg = root.querySelector("svg");
    var plot = { left: 120, right: 1035, top: 65, bottom: 335 };
    var normalX = 300;
    var specialX = 850;
    var y = function (value) {
      return plot.bottom - value / 42 * (plot.bottom - plot.top);
    };

    [0, 10, 20, 30, 40].forEach(function (tick) {
      var tickY = y(tick);
      svg.appendChild(svgElement("line", {
        x1: plot.left, y1: tickY, x2: plot.right, y2: tickY, "class": "pricing-grid",
      }));
      svg.appendChild(svgElement("text", {
        x: plot.left - 18, y: tickY + 4, "class": "pricing-tick", "text-anchor": "end",
      }, "$" + tick));
    });
    svg.appendChild(svgElement("line", {
      x1: plot.left, y1: plot.bottom, x2: plot.right, y2: plot.bottom,
      "class": "pricing-axis",
    }));
    svg.appendChild(svgElement("text", {
      x: 24, y: 200, "class": "pricing-axis-label",
      transform: "rotate(-90 24 200)", "text-anchor": "middle",
    }, "COMPLETE RUN COST (USD)"));

    series.forEach(function (item) {
      var normalY = y(item.normal);
      var specialY = y(item.special);
      var itemNormalX = normalX + item.xOffset;
      var itemSpecialX = specialX + item.xOffset;
      svg.appendChild(svgElement("line", {
        x1: itemNormalX, y1: normalY, x2: itemSpecialX, y2: specialY,
        "class": "pricing-series " + item.className,
      }));
      [
        { x: itemNormalX, y: normalY, value: item.normal, special: false },
        { x: itemSpecialX, y: specialY, value: item.special, special: true },
      ].forEach(function (point) {
        svg.appendChild(svgElement("circle", {
          cx: point.x, cy: point.y, r: 6,
          "class": "pricing-point " + item.className,
        }));
        var above = item.className === "openai";
        var modelY = point.y + (above ? -44 : 25);
        var valueY = point.y + (above ? -17 : 52);
        svg.appendChild(svgElement("text", {
          x: point.x, y: modelY,
          "class": "pricing-series-name",
          "text-anchor": "middle",
        }, item.name));
        svg.appendChild(svgElement("text", {
          x: point.x, y: valueY,
          "class": "pricing-value " + item.className,
          "text-anchor": "middle",
        }, formatExactTotal(point.value)));
      });
    });

    var previousX = normalX + 10;
    var previousY = y(deepseekPrevious);
    svg.appendChild(svgElement("circle", {
      cx: previousX, cy: previousY, r: 6,
      "class": "pricing-point previous",
    }));
    svg.appendChild(svgElement("text", {
      x: previousX, y: previousY - 44,
      "class": "pricing-series-name previous",
      "text-anchor": "middle",
    }, "Previous DeepSeek price"));
    svg.appendChild(svgElement("text", {
      x: previousX, y: previousY - 17,
      "class": "pricing-value previous",
      "text-anchor": "middle",
    }, formatExactTotal(deepseekPrevious)));

    [
      {
        x: normalX,
        title: "NORMAL PRICING",
        detail: "OPENAI STANDARD · DEEPSEEK PEAK",
      },
      {
        x: specialX,
        title: "FLEX / OFF PEAK",
        detail: "OPENAI FLEX · DEEPSEEK OFF-PEAK",
      },
    ].forEach(function (tick) {
      var label = svgElement("text", {
        x: tick.x, y: 385, "class": "pricing-x-label", "text-anchor": "middle",
      });
      label.appendChild(svgElement("tspan", { x: tick.x }, tick.title));
      label.appendChild(svgElement("tspan", {
        x: tick.x, dy: 19, "class": "pricing-x-detail",
      }, tick.detail));
      svg.appendChild(label);
    });
  }

  function renderDeepseekCacheCost(root) {
    var scenarios = [
      { className: "official", label: "$0.0028", detail: "cache read", value: 8.74 },
      { className: "", label: "$0.028", detail: "cache read", value: 28.81 },
      { className: "", label: "$0.14", detail: "NO CACHE", value: 118.03 },
    ];
    var maximum = Math.max.apply(null, scenarios.map(function (scenario) {
      return scenario.value;
    }));
    var rows = scenarios.map(function (scenario) {
      var detail = scenario.detail
        ? ' <span class="cache-cost-detail">(' + scenario.detail + ")</span>"
        : "";
      var width = (scenario.value / maximum * 100).toFixed(2);
      return [
        '<div class="cache-cost-row ' + scenario.className + '">',
        '<div class="cache-cost-row-heading">',
        '<p class="cache-cost-label">' + scenario.label + detail + "</p>",
        '<strong class="cache-cost-value">' + formatExactTotal(scenario.value) + "</strong>",
        "</div>",
        '<div class="cache-cost-track"><span style="width:' + width + '%"></span></div>',
        "</div>",
      ].join("");
    }).join("");

    root.innerHTML = [
      '<article class="card pricing-chart cache-cost-chart">',
      masthead(
        "run cost",
        "../../docs/assets/brand/fast-agent-lockup-light.svg"
      ),
      '<div class="content">',
      '<h1 class="cache-cost-title">',
      "<strong>DeepSeek V4 Flash 0731 · TB-2.1 run cost</strong>",
      "</h1>",
      '<div class="cache-cost-rows">' + rows + "</div>",
      "</div>",
      '<footer class="footer"><span class="primary">445 trials · complete run cost</span>',
      '<span>DeepSeek V4 Flash 0731</span>',
      '<span class="spacer"></span><span>PR #189</span>',
      "</footer>",
      "</article>",
    ].join("");
  }

  function renderTaskSwings(root) {
    var tasks = [
      { task: "configure-git-webserver", grok: 5, sol: 1 },
      { task: "pytorch-model-recovery", grok: 5, sol: 1 },
      { task: "gcode-to-text", grok: 4, sol: 1 },
      { task: "gpt2-codegolf", grok: 0, sol: 5 },
      { task: "extract-elf", grok: 1, sol: 4 },
      { task: "video-processing", grok: 0, sol: 3 },
    ];

    function trialDots(passes) {
      var dots = [];
      for (var index = 0; index < 5; index += 1) {
        dots.push('<i class="' + (index < passes ? "pass" : "fail") + '"></i>');
      }
      return [
        '<span class="task-swing-outcome" aria-label="' + passes + ' passes out of 5">',
        '<span class="task-swing-dots" aria-hidden="true">' + dots.join("") + "</span>",
        "<strong>" + passes + "/5</strong>",
        "</span>",
      ].join("");
    }

    var rows = tasks.map(function (task, index) {
      var grokAhead = task.grok > task.sol;
      var edge = Math.abs(task.grok - task.sol);
      return [
        '<div class="task-swing-row ' + (index === 3 ? "starts-sol" : "") + '">',
        '<strong class="task-swing-task">' + task.task + "</strong>",
        trialDots(task.grok),
        trialDots(task.sol),
        '<strong class="task-swing-edge ' + (grokAhead ? "grok" : "sol") + '">',
        (grokAhead ? "GROK" : "SOL") + " +" + edge,
        "</strong>",
        "</div>",
      ].join("");
    }).join("");

    root.innerHTML = [
      '<article class="card pricing-chart task-swings-chart">',
      masthead(
        "task comparison",
        "../../docs/assets/brand/fast-agent-lockup-light.svg"
      ),
      '<div class="content">',
      '<div class="task-swings-heading">',
      "<h1>Major task swings</h1>",
      "<p>Grok 4.6 high vs GPT-5.6 Sol high · fast-agent</p>",
      '<strong>|Δ| ≥ 3 of 5 trials</strong>',
      "</div>",
      '<div class="task-swings-table">',
      '<div class="task-swing-header"><span>Task</span><span>Grok 4.6</span>',
      "<span>GPT-5.6 Sol</span><span>Δ Passes</span></div>",
      rows,
      "</div>",
      "</div>",
      '<footer class="footer"><span class="primary">89 tasks · five trials each</span>',
      "<span>56 tasks were 5/5 for both</span>",
      '<span class="spacer"></span><span>PR #212 · PR #174</span>',
      "</footer>",
      "</article>",
    ].join("");
  }

  function renderGrokShare(root, swapped) {
    var results = cardResults(CARDS.grok).sort(function (first, second) {
      return second.score - first.score;
    });
    var focus = results.find(function (result) {
      return result.model === CARDS.grok.focusModel;
    });

    root.innerHTML = [
      '<article class="card grok-share' + (swapped ? " grok-share--swapped" : "") + '">',
      masthead("frontier result · provisional"),
      '<div class="content">',
      '<section class="grok-share-hero">',
      '<p class="grok-share-eyebrow">445 trials · frontier comparison</p>',
      "<h1>Grok 4.6 high</h1>",
      '<p class="grok-share-agent">fast-agent</p>',
      '<div class="grok-share-metrics">',
      '<p class="grok-share-score"><strong>' + focus.score.toFixed(1) +
        "%</strong><span>score</span></p>",
      '<div class="grok-share-costs">',
      '<p><strong>' + formatCostForShare(focus.costPerTrial) +
        '</strong><span>per task</span></p>',
      '<p><strong>$' + Math.round(focus.total) + '</strong><span>full run</span></p>',
      "</div>",
      "</div>",
      '<p class="grok-share-benchmark">Terminal-Bench <strong>2.1</strong></p>',
      "</section>",
      '<section class="spotlight-context grok-share-context">',
      '<div class="context-header"><span>Model / harness</span><span>Cost</span>' +
        "<span>Score</span><span>$/trial</span></div>",
      contextRows(results, focus),
      "</section>",
      "</div>",
      '<footer class="footer"><span class="primary">388 / 445 rewarded trials</span>',
      "<span>fast-agent 0.10.9 · Grok 4.6 high</span>",
      '<span class="spacer"></span><span class="accent">PR #212 · static analysis passed</span>',
      "</footer>",
      "</article>",
    ].join("");
  }

  function formatCostForShare(value) {
    return "$" + value.toFixed(2);
  }

  function start() {
    var root = document.getElementById("storyboard");
    var variant = new URLSearchParams(window.location.search).get("variant") || "spotlight";
    try {
      if (variant === "pricing-convergence") renderPricing(root);
      else if (variant === "deepseek-cache-cost") renderDeepseekCacheCost(root);
      else if (variant === "task-swings") renderTaskSwings(root);
      else if (variant === "grok-share") renderGrokShare(root, false);
      else if (variant === "grok-share-swapped") renderGrokShare(root, true);
      else renderSpotlight(root, CARDS[variant] || CARDS.spotlight);
    } catch (error) {
      root.innerHTML = '<div class="error">Unable to load benchmark data: ' +
        String(error) + "</div>";
    }
  }

  start();
})();
