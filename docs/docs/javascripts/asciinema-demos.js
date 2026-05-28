(function () {
  var players = new WeakMap();

  function colorScheme() {
    return (
      document.body.getAttribute("data-md-color-scheme") ||
      document.documentElement.getAttribute("data-md-color-scheme") ||
      ""
    );
  }

  function siteTheme() {
    var scheme = colorScheme();
    if (scheme === "slate") return "dark";
    if (scheme === "default") return "light";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function selectedMode(container) {
    var override = container.dataset.faTerminalThemeOverride || "auto";
    return override === "auto" ? siteTheme() : override;
  }

  function currentTheme(container) {
    return selectedMode(container) === "dark" ? "fast-agent-dark" : "fast-agent-light";
  }

  function updateButtons(container) {
    var override = container.dataset.faTerminalThemeOverride || "auto";
    container.querySelectorAll("[data-fa-terminal-theme]").forEach(function (button) {
      var active = button.getAttribute("data-fa-terminal-theme") === override;
      button.toggleAttribute("aria-pressed", active);
    });
  }

  function bindButtons(container) {
    container.querySelectorAll("[data-fa-terminal-theme]").forEach(function (button) {
      if (button.dataset.faTerminalThemeBound === "true") return;
      button.dataset.faTerminalThemeBound = "true";
      button.addEventListener("click", function () {
        container.dataset.faTerminalThemeOverride =
          button.getAttribute("data-fa-terminal-theme") || "auto";
        updateButtons(container);
        renderDemo(container, true);
      });
    });
  }

  function renderDemo(container, force) {
    var target = container.querySelector("[data-fa-asciinema-target]");
    var cast = container.getAttribute("data-fa-asciinema-cast");
    if (!target || !cast || !window.AsciinemaPlayer) return;

    bindButtons(container);
    updateButtons(container);

    var theme = currentTheme(container);
    if (!force && target.dataset.faAsciinemaLoaded === "true" && target.dataset.faAsciinemaTheme === theme) {
      return;
    }

    var previous = players.get(target);
    if (previous && typeof previous.dispose === "function") {
      previous.dispose();
    }

    target.dataset.faAsciinemaLoaded = "true";
    target.dataset.faAsciinemaTheme = theme;
    target.textContent = "";

    players.set(
      target,
      window.AsciinemaPlayer.create(cast, target, {
        cols: Number(container.getAttribute("data-fa-asciinema-cols") || 96),
        rows: Number(container.getAttribute("data-fa-asciinema-rows") || 22),
        preload: true,
        autoPlay: container.getAttribute("data-fa-asciinema-autoplay") === "true",
        poster: container.getAttribute("data-fa-asciinema-poster") || "npt:0:03",
        speed: Number(container.getAttribute("data-fa-asciinema-speed") || 1),
        idleTimeLimit: Number(container.getAttribute("data-fa-asciinema-idle-time-limit") || 1.3),
        fit: container.getAttribute("data-fa-asciinema-fit") || "width",
        theme: theme
      })
    );
  }

  function renderAll(force) {
    document.querySelectorAll("[data-fa-asciinema-cast]").forEach(function (container) {
      renderDemo(container, force);
    });
  }

  function observeTheme(target) {
    if (!target || target.dataset.faAsciinemaThemeObserver === "true") return;
    target.dataset.faAsciinemaThemeObserver = "true";
    new MutationObserver(function () { renderAll(false); }).observe(target, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"]
    });
  }

  function start() {
    observeTheme(document.documentElement);
    observeTheme(document.body);
    renderAll(false);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }

  if (window.document$ && window.document$.subscribe) {
    window.document$.subscribe(start);
  }
})();
