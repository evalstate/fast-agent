/*
 * Loads the benchmarks styles, data and renderer only on pages with a
 * [data-fa-bench] mount point. Runs for the first page and after every
 * instant-navigation swap (document$), so in-site navigation works too.
 */
(function () {
  "use strict";

  var ROOT = document.currentScript.src.replace(/javascripts\/benchmarks-loader\.js.*$/, "");
  var loaded = false;

  function load() {
    if (loaded || !document.querySelector("[data-fa-bench]")) return;
    loaded = true;
    var css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = ROOT + "stylesheets/benchmarks.css";
    // Render only once the styles are in, so the charts never paint unstyled.
    css.onload = function () {
      ["javascripts/benchmark-data.js", "javascripts/benchmarks.js"].forEach(function (path) {
        var script = document.createElement("script");
        script.src = ROOT + path;
        script.async = false; // execute in order: data, then renderer
        document.head.appendChild(script);
      });
    };
    // Instant navigation rebuilds <head>; a stylesheet at the end of <body> survives it.
    document.body.appendChild(css);
  }

  if (window.document$) window.document$.subscribe(load);
  else load();
})();
