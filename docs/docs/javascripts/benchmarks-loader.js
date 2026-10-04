/*
 * Loads benchmarks assets only on pages that use them, on first load and after
 * every instant-navigation swap (document$):
 *   [data-fa-bench]       client-rendered views: data + renderer
 *   [data-static-ledger]  build-time ledger: filters/sort only
 * Both get benchmarks.css first, so nothing paints unstyled.
 */
(function () {
  "use strict";

  var ROOT = document.currentScript.src.replace(/javascripts\/benchmarks-loader\.js.*$/, "");
  var requested = {};
  var styled = null;

  function styles() {
    if (!styled) {
      styled = new Promise(function (resolve) {
        var css = document.createElement("link");
        css.rel = "stylesheet";
        css.href = ROOT + "stylesheets/benchmarks.css";
        css.onload = resolve;
        // Instant navigation rebuilds <head>; a stylesheet at the end of <body> survives it.
        document.body.appendChild(css);
      });
    }
    return styled;
  }

  function scripts(paths) {
    paths.forEach(function (path) {
      if (requested[path]) return;
      requested[path] = true;
      var script = document.createElement("script");
      script.src = ROOT + path;
      script.async = false; // execute in order
      document.head.appendChild(script);
    });
  }

  function load() {
    var rendered = document.querySelector("[data-fa-bench]");
    var ledger = document.querySelector("[data-static-ledger]");
    if (!rendered && !ledger) return;
    styles().then(function () {
      if (rendered) scripts(["javascripts/benchmark-data.js", "javascripts/benchmarks.js"]);
      if (ledger) scripts(["javascripts/benchmarks-ledger.js"]);
    });
  }

  if (window.document$) window.document$.subscribe(load);
  else load();
})();
