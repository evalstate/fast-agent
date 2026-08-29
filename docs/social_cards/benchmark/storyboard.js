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
      costSnapshot: "original",
    },
    "grok-medium": {
      focusModel: "Grok 4.6 · medium",
      identity: "grok 4.6 medium",
      contextModels: [
        "GPT-5.6 Sol · high",
        "Grok 4.6 · medium",
        "Grok 4.6 · high",
        "Fable 5 · xhigh",
      ],
      footerAccent: "PR #221 · static analysis passed",
      share: {
        className: " grok-share--medium",
        status: "promoted result · provisional",
        eyebrow: "445 trials · promoted submission",
        footerPrimary: "390 / 445 rewarded trials",
        footerModel: "fast-agent 0.10.10 · Grok 4.6 medium",
      },
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
      var useOriginalCost =
        config.costSnapshot === "original" && result.originalCost !== undefined;
      var costPerTrial = useOriginalCost ? result.originalCost : result.cost;
      var total = useOriginalCost
        ? (result.originalTotalCost !== undefined
          ? result.originalTotalCost
          : result.sourceTotalCost !== undefined
            ? result.sourceTotalCost
            : costPerTrial * data.taskCount)
        : runTotal(result, data.taskCount);
      return {
        fastAgent: Boolean(result.fastAgent),
        costEstimate: Boolean(result.costEstimate) && !useOriginalCost,
        model: result.model,
        modelString: result.modelString || "",
        name: displayName(result),
        harness: result.harness,
        status: result.disclaimer ? "Provisional" : "Published",
        score: result.score,
        costPerTrial: costPerTrial,
        total: total,
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
      var classes = "context-row" +
        (result.model === focus.model ? " featured" : "") +
        (result.costEstimate ? " repriced" : "");
      var repricedLabel = result.costEstimate
        ? ' · <span class="repriced-label">repriced</span>'
        : "";
      return [
        '<div class="' + classes + '">',
        "<div><strong>" + result.name + "</strong><small>" + result.harness +
          ' · <span class="run-status ' + result.status.toLowerCase() + '">' +
          result.status + "</span>" + repricedLabel + "</small></div>",
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

  function renderGrokShare(root, config, swapped) {
    var share = config.share || {};
    var results = cardResults(config).sort(function (first, second) {
      return second.score - first.score;
    });
    var focus = results.find(function (result) {
      return result.model === config.focusModel;
    });

    root.innerHTML = [
      '<article class="card grok-share' + (share.className || "") +
        (swapped ? " grok-share--swapped" : "") + '">',
      masthead(share.status || "frontier result · provisional"),
      '<div class="content">',
      '<section class="grok-share-hero">',
      '<p class="grok-share-eyebrow">' +
        (share.eyebrow || "445 trials · frontier comparison") + "</p>",
      "<h1>" + config.identity + "</h1>",
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
      '<footer class="footer"><span class="primary">' +
        (share.footerPrimary || "388 / 445 rewarded trials") + "</span>",
      "<span>" + (share.footerModel || "fast-agent 0.10.9 · Grok 4.6 high") + "</span>",
      '<span class="spacer"></span><span class="accent">' + config.footerAccent + "</span>",
      "</footer>",
      "</article>",
    ].join("");
  }

  function formatCostForShare(value) {
    return "$" + value.toFixed(2);
  }

  function animationFrame(params, name, fallback) {
    var rawValue = params.get(name);
    if (rawValue === null || rawValue.trim() === "") return fallback;
    var value = Number(rawValue);
    return Number.isFinite(value) ? value : fallback;
  }

  function renderSolPriceAnimation(root, frameOverride, fpsOverride) {
    document.documentElement.classList.add("sol-price-animation-page");
    document.body.classList.add("sol-price-animation-page");
    var campaign = window.fastAgentBenchmark.campaigns.solPriceAnimation;
    var params = new URLSearchParams(window.location.search);
    var fps = Math.max(
      1,
      Math.round(fpsOverride === undefined ? animationFrame(params, "fps", 30) : fpsOverride)
    );
    var pulseFrames = Math.round(0.6 * fps);
    var motionFrames = Math.round(3.5 * fps);
    var holdFrames = Math.round(4 * fps);
    var motionEndFrame = pulseFrames + motionFrames;
    var totalFrames = motionEndFrame + holdFrames;
    var frame = Math.max(
      0,
      Math.min(
        totalFrames - 1,
        Math.round(
          frameOverride === undefined ? animationFrame(params, "frame", 0) : frameOverride
        )
      )
    );
    var movementFrame = Math.max(0, frame - pulseFrames);
    var progress = Math.min(movementFrame / motionFrames, 1);
    var eased = progress < 0.5
      ? 4 * progress * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 3) / 2;
    var revealProgress = Math.min(movementFrame / Math.round(0.5 * fps), 1);
    var priceReveal = 1 - Math.pow(1 - revealProgress, 3);
    var pulseIntensity = 0;
    if (frame < pulseFrames) {
      var pulseCycleFrames = Math.max(1, Math.round(pulseFrames / 3));
      var pulsePhase = (frame % pulseCycleFrames) / pulseCycleFrames;
      if (pulsePhase < 1 / 3) pulseIntensity = 1;
      else if (pulsePhase < 2 / 3) pulseIntensity = 2 - pulsePhase * 3;
    }
    var movingCost = campaign.sol.originalCost +
      (campaign.sol.currentCost - campaign.sol.originalCost) * eased;
    var scoreAxis = campaign.axes.score;
    var costAxis = campaign.axes.cost;
    var plot = { left: 112, right: 1144, top: 154, bottom: 608 };
    var width = plot.right - plot.left;
    var height = plot.bottom - plot.top;
    var x = function (cost) {
      var ratio = (Math.log10(cost) - Math.log10(costAxis.min)) /
        (Math.log10(costAxis.max) - Math.log10(costAxis.min));
      return plot.right - ratio * width;
    };
    var y = function (score) {
      return plot.bottom -
        ((score - scoreAxis.min) / (scoreAxis.max - scoreAxis.min)) * height;
    };
    var oldX = x(campaign.sol.originalCost);
    var currentX = x(campaign.sol.currentCost);
    var movingX = x(movingCost);
    var solY = y(campaign.sol.score);

    root.innerHTML = [
      '<article class="card sol-price-animation">',
      masthead("Sol pricing · 20% lower"),
      '<div class="content"></div>',
      '<footer class="footer">',
      '<span class="primary">445 trials · provisional submissions</span>',
      "<span>Current Sol pricing · Aug 24, 2026</span>",
      '<span class="spacer"></span>',
      '<span class="accent">PR #180 · #212 · #221 · #174</span>',
      "</footer>",
      "</article>",
    ].join("");

    var svg = svgElement("svg", {
      viewBox: "0 0 1200 740",
      role: "img",
      "aria-label":
        "Terminal-Bench score versus cost per task. GPT-5.6 Sol high moves from " +
        "$0.607 to $0.486 per task while retaining an 88.3 percent score.",
    });
    svg.appendChild(svgElement("text", {
      x: 54, y: 52, "class": "animation-title",
    }, campaign.title));

    var priceShift = svgElement("text", {
      x: 760, y: 55, "class": "animation-price-shift",
    });
    priceShift.appendChild(svgElement("tspan", {
      "class": "animation-price-old",
    }, "$" + campaign.sol.originalCost.toFixed(3)));
    priceShift.appendChild(svgElement("tspan", {
      dx: 16, opacity: priceReveal, "class": "animation-price-arrow",
    }, "→"));
    priceShift.appendChild(svgElement("tspan", {
      dx: 16, opacity: priceReveal, "class": "animation-price-new",
    }, "$" + campaign.sol.currentCost.toFixed(3)));
    priceShift.appendChild(svgElement("tspan", {
      dx: 8, opacity: priceReveal, "class": "animation-price-unit",
    }, "/ TASK"));
    svg.appendChild(priceShift);
    svg.appendChild(svgElement("text", {
      x: 1144, y: 84, opacity: priceReveal,
      "class": "animation-reduction", "text-anchor": "end",
    }, Math.round(campaign.sol.reduction * 100) + "% LOWER API PRICE"));

    scoreAxis.ticks.forEach(function (tick) {
      var tickY = y(tick);
      svg.appendChild(svgElement("line", {
        x1: plot.left, y1: tickY, x2: plot.right, y2: tickY,
        "class": "animation-grid",
      }));
      svg.appendChild(svgElement("text", {
        x: plot.left - 16, y: tickY + 5, "class": "animation-tick",
        "text-anchor": "end",
      }, tick + "%"));
    });
    costAxis.ticks.forEach(function (tick) {
      var tickX = x(tick);
      svg.appendChild(svgElement("line", {
        x1: tickX, y1: plot.top, x2: tickX, y2: plot.bottom,
        "class": "animation-grid",
      }));
      svg.appendChild(svgElement("text", {
        x: tickX, y: plot.bottom + 27, "class": "animation-tick",
        "text-anchor": "middle",
      }, "$" + tick.toFixed(2)));
    });
    svg.appendChild(svgElement("line", {
      x1: plot.left, y1: plot.top, x2: plot.left, y2: plot.bottom,
      "class": "animation-axis",
    }));
    svg.appendChild(svgElement("line", {
      x1: plot.left, y1: plot.bottom, x2: plot.right, y2: plot.bottom,
      "class": "animation-axis",
    }));
    svg.appendChild(svgElement("text", {
      x: 26, y: (plot.top + plot.bottom) / 2,
      "class": "animation-axis-label",
      transform: "rotate(-90 26 " + ((plot.top + plot.bottom) / 2) + ")",
      "text-anchor": "middle",
    }, "TERMINAL-BENCH 2.1 SCORE (%) · HIGHER ↑"));
    svg.appendChild(svgElement("text", {
      x: (plot.left + plot.right) / 2,
      y: 696,
      "class": "animation-axis-label",
      "text-anchor": "middle",
    }, "COST / TASK (USD) · CHEAPER →"));

    var labelPositions = {
      "grok-4.5-high": { dx: 0, dy: -45, anchor: "middle" },
      "grok-4.6-high": { dx: -16, dy: 40, anchor: "end" },
      "grok-4.6-medium": { dx: 16, dy: 40, anchor: "start" },
    };
    campaign.results.forEach(function (result) {
      var pointX = x(result.cost);
      var pointY = y(result.score);
      var label = labelPositions[result.id];
      var group = svgElement("g", { "class": "animation-point animation-point--grok" });
      group.appendChild(svgElement("circle", { cx: pointX, cy: pointY, r: 9 }));
      var text = svgElement("text", {
        x: pointX + label.dx,
        y: pointY + label.dy,
        "class": "animation-point-label",
        "text-anchor": label.anchor,
      });
      text.appendChild(svgElement("tspan", {
        x: pointX + label.dx, "class": "animation-point-model",
      }, result.model.replace(" · ", " ") + " · " + result.score.toFixed(1) + "%"));
      text.appendChild(svgElement("tspan", {
        x: pointX + label.dx, dy: 18, "class": "animation-point-numbers",
      }, "$" + result.cost.toFixed(3) + " / TASK"));
      group.appendChild(text);
      svg.appendChild(group);
    });

    svg.appendChild(svgElement("line", {
      x1: oldX, y1: solY, x2: movingX, y2: solY,
      "class": "animation-sol-trail",
    }));
    svg.appendChild(svgElement("circle", {
      cx: oldX, cy: solY, r: 11, "class": "animation-sol-original",
    }));
    svg.appendChild(svgElement("text", {
      x: oldX, y: solY + 36, "class": "animation-sol-original-label",
      "text-anchor": "middle",
    }, "ORIGINAL $" + campaign.sol.originalCost.toFixed(3)));
    svg.appendChild(svgElement("circle", {
      cx: currentX, cy: solY, r: 14, "class": "animation-sol-target",
    }));
    svg.appendChild(svgElement("circle", {
      cx: movingX, cy: solY,
      r: 16 + progress * 5 + pulseIntensity * 10,
      opacity: 0.35 + pulseIntensity * 0.55,
      "class": "animation-sol-halo",
    }));
    svg.appendChild(svgElement("circle", {
      cx: movingX, cy: solY, r: 10 + pulseIntensity * 2,
      "class": "animation-sol-point",
    }));
    var solLabelX = movingX + 16;
    var solLabel = svgElement("text", {
      x: solLabelX, y: solY - 38, "class": "animation-sol-label",
      "text-anchor": "start",
    });
    solLabel.appendChild(svgElement("tspan", {
      x: solLabelX, "class": "animation-sol-model",
    }, "GPT-5.6 SOL HIGH · 88.3%"));
    svg.appendChild(solLabel);

    root.querySelector(".content").appendChild(svg);
  }

  function playSolPriceAnimation(root) {
    var fps = 30;
    var pulseFrames = Math.round(0.6 * fps);
    var motionFrames = Math.round(3.5 * fps);
    var motionEndFrame = pulseFrames + motionFrames;
    var durationMs = 8100;
    var startedAt;
    var lastFrame = -1;
    var requestId;

    function draw(now) {
      if (startedAt === undefined) startedAt = now;
      var elapsed = (now - startedAt) % durationMs;
      var frame = Math.min(Math.floor(elapsed * fps / 1000), motionEndFrame);
      if (frame !== lastFrame) {
        renderSolPriceAnimation(root, frame, fps);
        lastFrame = frame;
      }
      requestId = window.requestAnimationFrame(draw);
    }

    function restart() {
      startedAt = undefined;
      lastFrame = -1;
      if (requestId !== undefined) window.cancelAnimationFrame(requestId);
      requestId = window.requestAnimationFrame(draw);
    }

    root.addEventListener("click", restart);
    document.addEventListener("keydown", function (event) {
      if (event.key === " " || event.key.toLowerCase() === "r") {
        event.preventDefault();
        restart();
      }
    });
    restart();
  }

  function start() {
    var root = document.getElementById("storyboard");
    var params = new URLSearchParams(window.location.search);
    var variant = params.get("variant") || "spotlight";
    try {
      if (variant === "pricing-convergence") renderPricing(root);
      else if (variant === "deepseek-cache-cost") renderDeepseekCacheCost(root);
      else if (variant === "task-swings") renderTaskSwings(root);
      else if (variant === "sol-price-animation") {
        if (params.has("frame")) renderSolPriceAnimation(root);
        else playSolPriceAnimation(root);
      }
      else if (variant === "grok-medium") renderGrokShare(root, CARDS["grok-medium"], false);
      else if (variant === "grok-share") renderGrokShare(root, CARDS.grok, false);
      else if (variant === "grok-share-swapped") renderGrokShare(root, CARDS.grok, true);
      else renderSpotlight(root, CARDS[variant] || CARDS.spotlight);
    } catch (error) {
      root.innerHTML = '<div class="error">Unable to load benchmark data: ' +
        String(error) + "</div>";
    }
  }

  start();
})();
