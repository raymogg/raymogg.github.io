(function () {
    var MAX_GENERATIONS = 100;
    var MAX_POPULATION = 400;
    var STEP_MS = 380;

    // Knuth's multiplicative method for sampling Poisson(mu).
    function poisson(mu) {
        if (mu <= 0) return 0;
        // Guard against underflow of Math.exp(-mu) for larger mu by
        // splitting the draw into chunks of at most 500.
        var remaining = mu;
        var total = 0;
        while (remaining > 0) {
            var step = Math.min(remaining, 500);
            remaining -= step;
            var limit = Math.exp(-step);
            var k = 0;
            var p = 1;
            do {
                k++;
                p *= Math.random();
            } while (p > limit);
            total += k - 1;
        }
        return total;
    }

    function createSim() {
        return { generation: 0, population: 1, history: [1], dead: false };
    }

    function step(sim, mu) {
        if (sim.dead || sim.generation >= MAX_GENERATIONS) return sim;
        var next = 0;
        for (var i = 0; i < sim.population; i++) {
            next += poisson(mu);
            if (next > MAX_POPULATION) { next = MAX_POPULATION; break; }
        }
        sim.generation++;
        sim.population = next;
        sim.history.push(next);
        if (next === 0) sim.dead = true;
        return sim;
    }

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = { poisson: poisson, createSim: createSim, step: step,
                           MAX_GENERATIONS: MAX_GENERATIONS, MAX_POPULATION: MAX_POPULATION };
        return;
    }

    var root = document.getElementById('branching');
    if (!root) return;

    var canvas = root.querySelector('canvas');
    var ctx = canvas.getContext('2d');
    var slider = root.querySelector('input[type=range]');
    var muLabel = root.querySelector('.bp-mu');
    var stat = root.querySelector('.bp-stat');

    var mu = parseFloat(slider.value);
    var sim = createSim();
    var timer = null;

    // Seed a short run so the widget shows a trace before it is scrolled
    // into view, rather than sitting empty at generation 0.
    function seed() {
        var s = createSim();
        for (var attempt = 0; attempt < 12; attempt++) {
            s = createSim();
            for (var i = 0; i < 6 && !s.dead; i++) step(s, mu);
            if (!s.dead) return s;
        }
        return s;
    }
    sim = seed();

    function cssVar(name, fallback) {
        var v = getComputedStyle(root).getPropertyValue(name);
        return v ? v.trim() : fallback;
    }

    function resize() {
        var ratio = window.devicePixelRatio || 1;
        var w = canvas.clientWidth;
        var h = canvas.clientHeight;
        canvas.width = w * ratio;
        canvas.height = h * ratio;
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        draw();
    }

    function draw() {
        var w = canvas.clientWidth;
        var h = canvas.clientHeight;
        ctx.clearRect(0, 0, w, h);

        var hist = sim.history;
        var peak = 1;
        for (var i = 0; i < hist.length; i++) peak = Math.max(peak, hist[i]);
        // Log scale: exponential growth otherwise flattens all early
        // generations into an unreadable smudge against the final bar.
        var scale = Math.log(peak + 1);

        var padX = 2;
        var padY = 6;
        var plotW = w - padX * 2;
        var plotH = h - padY * 2;
        // Always scale the x-axis to the full run so the trace grows
        // left-to-right instead of rescaling every step.
        var dx = plotW / MAX_GENERATIONS;

        // Baseline.
        ctx.strokeStyle = '#eee';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padX, h - padY + 0.5);
        ctx.lineTo(w - padX, h - padY + 0.5);
        ctx.stroke();

        var barW = Math.max(1.5, dx - 1.5);
        ctx.fillStyle = sim.dead ? '#c8c8c8' : cssVar('--bp-ink', '#333');
        for (var g = 0; g < hist.length; g++) {
            var v = hist[g];
            if (v === 0) continue;
            var barH = Math.max(1.5, (Math.log(v + 1) / scale) * plotH);
            var x = padX + g * dx;
            ctx.fillRect(x, h - padY - barH, barW, barH);
        }
    }

    function render() {
        // Two decimals only when the 0.05 step needs it.
        muLabel.textContent = (Math.round(mu * 100) % 10 === 0) ?
            mu.toFixed(1) : mu.toFixed(2);
        var word = sim.population === 1 ? 'individual' : 'individuals';
        if (sim.dead) {
            stat.textContent = 'extinct at generation ' + sim.generation;
        } else if (sim.generation >= MAX_GENERATIONS) {
            stat.textContent = 'survived ' + MAX_GENERATIONS + ' generations';
        } else if (sim.population >= MAX_POPULATION) {
            stat.textContent = 'generation ' + sim.generation + ' — capped at ' + MAX_POPULATION;
        } else {
            stat.textContent = 'generation ' + sim.generation + ' — ' +
                sim.population + ' ' + word;
        }
        draw();
    }

    function tick() {
        if (sim.dead || sim.generation >= MAX_GENERATIONS) {
            sim = createSim();
        } else {
            step(sim, mu);
        }
        render();
    }

    function start() {
        if (timer) clearInterval(timer);
        timer = setInterval(tick, STEP_MS);
    }

    slider.addEventListener('input', function () {
        mu = parseFloat(slider.value);
        sim = seed();
        render();
    });

    window.addEventListener('resize', resize);

    // Pause when scrolled out of view so we don't animate off-screen.
    if (window.IntersectionObserver) {
        new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
                if (e.isIntersecting) start();
                else if (timer) { clearInterval(timer); timer = null; }
            });
        }, { rootMargin: '120px', threshold: 0 }).observe(root);
    } else {
        start();
    }

    resize();
    render();
})();
