/*
 * Click to enlarge for dense figures: wrap an image in <a class="fa-zoom" href="full.png">.
 * Without JS the link opens the image. With JS it opens in a full-window dialog that
 * fits the image to the viewport; clicking the image toggles actual size (scrollable).
 * One delegated listener on document, so it survives instant-navigation page swaps.
 */
(function () {
  "use strict";

  var dialog = null;

  function build() {
    dialog = document.createElement("dialog");
    dialog.className = "fa-zoom-dialog";
    dialog.innerHTML =
      '<button class="fa-zoom-dialog__close" type="button" aria-label="Close">×</button>' +
      '<div class="fa-zoom-dialog__frame"><img alt=""></div>';
    dialog.querySelector(".fa-zoom-dialog__close").addEventListener("click", function () {
      dialog.close();
    });
    dialog.querySelector("img").addEventListener("click", function (event) {
      event.stopPropagation();
      dialog.classList.toggle("is-actual");
    });
    dialog.querySelector(".fa-zoom-dialog__frame").addEventListener("click", function () {
      dialog.close();
    });
    dialog.addEventListener("close", function () {
      document.documentElement.classList.remove("fa-zoom-open");
    });
    document.body.appendChild(dialog);
  }

  document.addEventListener("click", function (event) {
    var link = event.target.closest("a.fa-zoom");
    if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
    // Capture phase + stopPropagation: instant navigation would otherwise treat it as a page link.
    event.preventDefault();
    event.stopPropagation();
    if (!dialog || !dialog.isConnected) build();
    var img = dialog.querySelector("img");
    var thumb = link.querySelector("img");
    img.src = link.href;
    img.alt = thumb ? thumb.alt : "";
    dialog.classList.remove("is-actual");
    document.documentElement.classList.add("fa-zoom-open");
    dialog.showModal();
  }, true);
})();
