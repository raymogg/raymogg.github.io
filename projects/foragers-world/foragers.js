/* Side-by-side replay of pre-recorded foraging runs.
 *
 * Each run is a JSON trajectory exported from the JAX simulation: food is one
 * base64 byte per cell, foragers are flat cell indices. Nothing is simulated
 * here -- the page replays what JAX produced, so the numbers on the page are
 * the numbers the environment actually generates.
 */
(function () {
    var STEP_MS = 90;
    // Chart ceiling, shared by all panels. Comfortably above the highest
    // population any of the recorded runs reaches.
    var Y_MAX = 240;
    var INK = "#333";
    var FOOD = "#8fb08f";
    var BAND = "#e6e6e6";

    function decodeFood(b64) {
        var bin = atob(b64);
        var out = new Uint8Array(bin.length);
        for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out;
    }

    function fitCanvas(canvas, cssHeight) {
        var ratio = window.devicePixelRatio || 1;
        var cssWidth = canvas.clientWidth;
        canvas.width = Math.round(cssWidth * ratio);
        canvas.height = Math.round(cssHeight * ratio);
        var ctx = canvas.getContext("2d");
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        return { ctx: ctx, w: cssWidth, h: cssHeight };
    }

    function drawGrid(canvas, run, frame) {
        var n = run.grid;
        var fit = fitCanvas(canvas, canvas.clientWidth);
        var ctx = fit.ctx;
        var px = fit.w / n;

        ctx.clearRect(0, 0, fit.w, fit.h);

        // Food: alpha scales with amount, so dense patches read darker.
        var food = frame.food;
        ctx.fillStyle = FOOD;
        for (var i = 0; i < food.length; i++) {
            var v = food[i];
            if (v === 0) continue;
            ctx.globalAlpha = 0.15 + 0.85 * (v / run.maxFood);
            ctx.fillRect((i % n) * px, Math.floor(i / n) * px, px, px);
        }
        ctx.globalAlpha = 1;

        // Foragers, slightly inset so they read as dots on the food layer.
        ctx.fillStyle = INK;
        var pad = px * 0.18;
        for (var j = 0; j < frame.p.length; j++) {
            var cell = frame.p[j];
            ctx.fillRect(
                (cell % n) * px + pad,
                Math.floor(cell / n) * px + pad,
                px - 2 * pad,
                px - 2 * pad
            );
        }
    }

    function drawChart(canvas, run, upto) {
        var fit = fitCanvas(canvas, 54);
        var ctx = fit.ctx;
        var w = fit.w;
        var h = fit.h;
        var total = run.frames.length;
        // Shared y-scale across all three runs so the charts are comparable,
        // but set from the data rather than MAX_POPULATION -- scaling to 256
        // squashes the band into a sliver no one can read.
        var yMax = run.yMax;

        ctx.clearRect(0, 0, w, h);

        // Target band, as a shaded strip.
        var yLow = h - (run.band[0] / yMax) * h;
        var yHigh = h - (run.band[1] / yMax) * h;
        ctx.fillStyle = BAND;
        ctx.fillRect(0, yHigh, w, yLow - yHigh);

        ctx.strokeStyle = INK;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (var i = 0; i <= upto; i++) {
            var x = (i / Math.max(total - 1, 1)) * w;
            var y = h - (run.frames[i].n / yMax) * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    function setup(panel, run) {
        var grid = panel.querySelector(".fw-grid");
        var chart = panel.querySelector(".fw-chart");
        var stat = panel.querySelector(".fw-stat");
        var frame = 0;

        function render() {
            var f = run.frames[frame];
            drawGrid(grid, run, f);
            drawChart(chart, run, frame);
            stat.textContent =
                "step " + frame + " — " + f.n + " foragers";
        }

        render();
        return { render: render, advance: function () {
            frame = (frame + 1) % run.frames.length;
            render();
        } };
    }

    function load(panel) {
        var src = panel.getAttribute("data-run");
        return fetch(src)
            .then(function (r) {
                if (!r.ok) throw new Error(src + ": " + r.status);
                return r.json();
            })
            .then(function (run) {
                // Decode once, up front: these are short runs and decoding per
                // frame would stutter the animation.
                run.frames.forEach(function (f) {
                    f.food = decodeFood(f.f);
                });
                run.yMax = Y_MAX;
                panel.querySelector(".fw-cost").textContent =
                    run.moveCost.toFixed(2);
                panel.querySelector(".fw-regrowth").textContent =
                    run.regrowth.toFixed(2);
                panel.querySelector(".fw-return").textContent =
                    run.return + "/" + (run.frames.length - 1);
                return setup(panel, run);
            })
            .catch(function (err) {
                panel.querySelector(".fw-stat").textContent =
                    "could not load run";
                if (window.console) console.error(err);
                return null;
            });
    }

    function init() {
        var panels = [].slice.call(document.querySelectorAll(".fw-panel"));
        if (!panels.length) return;

        Promise.all(panels.map(load)).then(function (views) {
            var live = views.filter(Boolean);
            if (!live.length) return;

            // One shared clock, so the three runs stay in step with each other.
            setInterval(function () {
                live.forEach(function (v) { v.advance(); });
            }, STEP_MS);

            var resizeTimer;
            window.addEventListener("resize", function () {
                clearTimeout(resizeTimer);
                resizeTimer = setTimeout(function () {
                    live.forEach(function (v) { v.render(); });
                }, 120);
            });
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
