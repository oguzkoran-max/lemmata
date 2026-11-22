/* P017 · The Living Corpus — reversible scroll state machine.
   Input: normalized progress P (0..1) over the single pinned chapter.
   Output: envelope custom properties consumed by CSS transforms only.
   No IntersectionObserver reveals; no scroll hijack; passive listeners. */
(function () {
  "use strict";
  var de = document.documentElement;
  de.classList.add("js");

  /* Configurable workbench destination. */
  var meta = document.querySelector('meta[name="delta-workbench-url"]');
  if (meta && meta.content) {
    var links = document.querySelectorAll("a[data-workbench]");
    for (var i = 0; i < links.length; i += 1) links[i].setAttribute("href", meta.content);
  }

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (reduced.matches) return; /* stacked mode is fully composed without JS state */

  var scene = document.querySelector("[data-scene]");
  if (!scene) return;

  function env(p, a, b) {
    var v = (p - a) / (b - a);
    return v < 0 ? 0 : v > 1 ? 1 : v;
  }

  var heads = document.querySelectorAll(".cine .h-mega, .cine .h-act");

  /* Hero motion layer: cine-only, never on Save-Data; poster remains the LCP. */
  var vid = document.querySelector(".keyvis video");
  var conn = navigator.connection || {};
  /* the cine chapter is display:none on mobile; never fetch the 3.8 MB hero video then */
  var cineVisible = (function () {
    var c = document.querySelector(".cine");
    return c && getComputedStyle(c).display !== "none";
  })();
  if (vid && vid.dataset.src && !conn.saveData && cineVisible) {
    vid.src = vid.dataset.src;
  }
  /* visibility windows must mirror the CSS trapezoids (see .b1-.b6) */
  var WINDOWS = [[0, 0.155], [0.165, 0.35], [0.36, 0.56], [0.555, 0.725], [0.735, 0.868], [0.875, 2]];
  var blks = [];
  var blkNodes = document.querySelectorAll(".cine .blk");
  for (var b = 0; b < blkNodes.length; b += 1) {
    blks.push({ el: blkNodes[b], links: blkNodes[b].querySelectorAll("a"), win: WINDOWS[b] || [0, 2] });
  }
  var motionButton = document.querySelector(".pin > .motion-toggle");
  var summaryLink = document.querySelector(".cine-summary");
  var sceneCounter = document.querySelector(".scene-counter");
  var ticking = false;

  function paint() {
    ticking = false;
    var r = scene.getBoundingClientRect();
    var span = r.height - window.innerHeight;
    var visible = span > 0 && r.bottom > 0 && r.top < window.innerHeight && getComputedStyle(scene.parentElement).display !== "none";
    summaryLink.hidden = !visible;
    sceneCounter.hidden = !visible;
    if (span <= 0) return; /* cine hidden (mobile) or degenerate viewport */
    var p = Math.min(1, Math.max(0, -r.top / span));
    var act = 0;
    for (var a = 1; a < blks.length; a += 1) if (p >= blks[a].win[0]) act = a;
    sceneCounter.textContent = (act + 1) + " / " + blks.length;

    var settle = env(p, 0, 0.13);
    var sep = env(p, 0.06, 0.22) - env(p, 0.37, 0.45);
    var col = env(p, 0.39, 0.51) - env(p, 0.55, 0.65);
    var mat = env(p, 0.55, 0.67);
    var dial = env(p, 0.72, 0.86);
    var fold = env(p, 0.88, 0.985);

    var s = de.style;
    s.setProperty("--p", p.toFixed(4));
    s.setProperty("--settle", settle.toFixed(4));
    s.setProperty("--sep", sep.toFixed(4));
    s.setProperty("--col", col.toFixed(4));
    s.setProperty("--mat", mat.toFixed(4));
    s.setProperty("--dial", dial.toFixed(4));
    s.setProperty("--fold", fold.toFixed(4));

    /* typographic x-ray: variable weight breathes with the narrative */
    var w = (560 - p * 90).toFixed(0);
    for (var h = 0; h < heads.length; h += 1) {
      heads[h].style.fontVariationSettings = '"opsz" 72,"wght" ' + w;
    }

    /* Match the keyvis opacity window: it is fully hidden at p >= 0.125. */
    if (motionButton) motionButton.hidden = p >= 0.125;

    /* hero video runs only while Act I is on stage */
    if (vid && vid.src) {
      if (p > 0.16) {
        if (!vid.paused) vid.pause();
      } else if (vid.paused && vid.dataset.userPaused !== "true") {
        var pr = vid.play();
        if (pr && pr.catch) pr.catch(function () {});
      }
    }

    /* links inside visually hidden copy blocks must not be invisible tab stops,
       and focus must not pin a closed act on screen (double-exposure bug) */
    for (var k = 0; k < blks.length; k += 1) {
      var inW = p >= blks[k].win[0] && p <= blks[k].win[1];
      blks[k].el.classList.toggle("is-off", !inW); /* the opaque focus plate is only for an act that is off stage */
      for (var l = 0; l < blks[k].links.length; l += 1) {
        blks[k].links[l].style.pointerEvents = inW ? "auto" : "none";
      }
    }
  }

  function onScroll() {
    if (!ticking) {
      ticking = true;
      window.requestAnimationFrame(paint);
    }
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll, { passive: true });
  paint();
})();
