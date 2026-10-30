# Floor control experiments

Running log of attempts to control the population in a low band without it
crashing to zero. Each experiment changes one thing against a fixed baseline,
has its own config in `experiments/`, and writes its results to
`results/<name>/`. Add new experiments at the end.

## Goal

Keep the population inside the **floor band (0–30)** for as much of the
500-step episode as possible, without it going extinct. Extinction ends the
episode, so every remaining step is lost.

## How experiments are compared

Unless an experiment says otherwise, everything is held fixed:

- **Environment:** 32×32 grid; 16 starting foragers with 10 energy each; food
  regrows with probability 0.2 per tile per step, capped at 5; foragers breed at
  4 energy.
- **Agent:** PPO, 5M training steps, training seed 0, PureJaxRL default
  hyperparameters.
- **Evaluation:** the same 256 worlds (evaluation seed 0). Results are greedy
  PPO unless marked sampled.
- **Metrics:** return is steps in band, out of 500. Extinctions are split into
  **early** (within the first 20 steps) and **late** (after step 20), because
  the two have different causes (see the baseline).

All numbers below come from each experiment's `episodes.csv` and `raw/` files.

---

## Baseline: band_sweep, floor band

- **Config:** `experiments/band_sweep.toml` (the floor band is one of four
  bands in it).
- **Results:** `results/band_sweep/`.
- **Observation:** 3 features: total food, population, mean energy.

| | PPO greedy | PPO sampled | Best fixed (cost 1.75) | Random |
|---|---|---|---|---|
| Mean return | 349.0 | 239.3 | 83.6 | 37.1 |
| Median return | 455 | 233.5 | 79 | 27 |
| Extinct (of 256) | 68 (27%) | 166 (65%) | 177 (69%) | 94% |
| …within the first 20 steps | 56 | 72 | | |
| …after step 20 | 12 | 94 | | |
| Return in worlds that survive | 461.5 | 457.5 | 124.5 | |

**What we learned:**
- **The returns split into two groups.** 74% of worlds score above 400 and 22%
  score below 50. Surviving worlds are controlled well (about 460/500). The
  mean is pulled down by extinctions.
- **Most extinctions happen at the start.** 82% of greedy extinctions (56 of
  68) happen in the first 20 steps, and the median extinction step is 9.
- **Start-up crash mechanism.** All 16 starting foragers have 10 energy, so
  they breed at once and the population jumps to about 50 within 2–3 steps,
  well above the band. The agent responds with its harshest cull (cost 2.5) and
  keeps it up while the population falls through the band. Example, world 3:

  ```
  population  16 → 32 → 51 → 30 → 19 → 3 → 0
  move cost        2.5  2.5  2.5  2.5  2.5  2.5
  ```

  The peak at the start is the same (median 51) in worlds that die and in worlds
  that survive. The difference is whether a few foragers make it through the
  crash.
- **The learned rule** is a hard cull: cost 2.5 above a population of about 10,
  and a mix of 0 and 2.5 below it. The switch point is about 12, close to the
  band's middle (15).

---

## Experiment 1: add a growth observation

- **Config:** `experiments/floor_growth_obs.toml`.
- **Results:** `results/floor_growth_obs/`.
- **Change:** the agent also observes **growth**, the population's fractional
  change since the last step, `(population − previous population) / previous`.
  Each forager has at most one child per step, so growth always lies between
  −1 and +1. Everything else is identical to the baseline.
- **Hypothesis:** the baseline agent sees only the current population, so it
  can't tell a crashing population from a recovering one. Seeing the trend
  should let it ease off the cull while the population is falling, and so
  prevent the crash.

### Results

| | Baseline (3 obs) | + growth |
|---|---|---|
| **Greedy** mean return | 349.0 | **374.9** |
| Greedy median return | 455 | 457 |
| Greedy extinct (of 256) | 68 (27%) | **48 (19%)** |
| …within the first 20 steps | 56 | 48 |
| …after step 20 | 12 | **0** |
| Greedy return in surviving worlds | 461.5 | 459.4 |
| Greedy mean max overshoot | 24.4 | 25.0 |
| **Sampled** mean return | 239.3 | 259.5 |
| Sampled median return | 233.5 | 331.0 |
| Sampled extinct (of 256) | 166 | 142 |
| …within the first 20 steps | 72 | 66 |
| …after step 20 | 94 | 76 |

### Findings

- **Late extinctions went from 12 to 0** (greedy). Once the population is in
  the band, seeing its trend is enough to stop it drifting into extinction. This
  is the one clear effect, and it shows up in the sampled policy too (94 to 76).
- **The early crash barely changed: 56 to 48 worlds** (22% to 19%). That
  difference is within run-to-run noise: roughly ±2.5 percentage points on 256
  worlds, and there is only one training seed per setup.
- **Which worlds crash looks largely random.** Of the early crashes, only 18
  worlds crash in both runs, 38 crash only in the baseline and 30 only with
  growth. So *whether* a given world survives the start looks mostly like
  chance; the *rate* is the stable quantity.
- **The start-up crash plays out exactly as before.** The agent now sees the
  drop but keeps a high cost anyway. Example, world 3:

  | Step | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|---|
  | Population seen | 16 | 32 | 51 | 30 | 19 | 4 | 5 |
  | Growth seen | 0.0 | +1.0 | +0.59 | −0.41 | −0.37 | −0.79 | +0.25 |
  | Cost chosen | 2.5 | 2.5 | 2.5 | 2.5 | 2.25 | 2.25 | 2.25 |
  | Population after | 32 | 51 | 30 | 19 | 4 | 5 | 0 |

  After two steps of falling population (growth −0.41, then −0.37), the agent
  eases only from 2.5 to 2.25.

- **Surviving worlds score the same** (about 460), so the gain in the mean comes
  entirely from fewer extinctions.

### Interpretation (hypothesis, later disproved)

> **Update, Experiment 2:** this explanation was wrong. Rescaling the population
> observation cut early crashes from 56 to 13 with the same starting conditions,
> so the crash *was* avoidable. The agent lacked resolution, not information.

The start-up crash may be decided before it is visible in the population. After
the simultaneous first round of breeding, parents and children hold about 5
energy each. At cost 2.5 that lasts about two steps. By the time the
population visibly falls, the survivors are already nearly out of energy, so
lowering the cost then is too late. If that is right, the crash comes from the
starting conditions, not from what the agent can observe, and no observation
change will fix it.

### Caveats

- One training seed per setup, so the 349 vs 375 difference is not conclusive.
  The late-extinction result (12 vs 0) is the strong one.
- Hyperparameters are PureJaxRL defaults. The learning rate decays to zero over
  5M steps, so the learning curves flatten partly by design.

---

## Experiment 2: scale the population observation to the band

- **Config:** `experiments/floor_band_scaled_pop.toml`.
- **Results:** `results/floor_band_scaled_pop/`.
- **Change:** the `population` observation (population / 256) is replaced by
  `population_band`, `(population − band low) / (band high − band low)`, which
  for this band is population / 30. The band now spans 0–1 of the input
  instead of 0–0.12. No growth observation, so this changes exactly one thing
  against the baseline.
- **Hypothesis:** at the baseline's scale, one forager is a change of 0.004 in
  the input, so the network can barely distinguish 5 foragers from 15. Finer
  resolution inside the band should give it finer control.

### Results

| | Baseline | Exp 1: + growth | **Exp 2: band-scaled** |
|---|---|---|---|
| **Greedy** mean return | 349.0 | 374.9 | **389.4** |
| Greedy median return | 455 | 457 | **472** |
| Greedy extinct (of 256) | 68 | 48 | 84 |
| …within the first 20 steps | 56 | 48 | **13** |
| …steps 20–99 | 2 | 0 | 13 |
| …step 100 or later | 10 | 0 | 58 |
| Greedy median extinction step | 9 | 11 | 179 |
| Greedy return in surviving worlds | 461.5 | 459.4 | **477.7** |
| Greedy return in extinct worlds | 37.8 | 9.0 | 208.8 |
| **Sampled** mean return | 239.3 | 259.5 | **403.7** |
| Sampled median return | 233.5 | 331.0 | **463** |
| Sampled extinct (of 256) | 166 | 142 | **55** |
| …within the first 20 steps | 72 | 66 | **14** |
| …step 20 or later | 94 | 76 | 41 |

### Findings

- **The start-up crash is mostly gone: early extinctions fell from 56 to 13**
  (greedy) and from 72 to 14 (sampled), with the same starting conditions. The
  crash was avoidable; the baseline agent just couldn't resolve small
  populations.
- **This is the best floor result so far.** Greedy scores 389.4 and sampled
  403.7, and surviving worlds score 477.7 of 500, the tightest control yet.
- **But greedy now dies late instead.** There are 71 extinctions after step 20
  (vs 12 in the baseline), with a median extinction step of 179. Total greedy
  extinctions *rose*, from 68 to 84, but each one happens much later, so far less
  reward is lost: extinct worlds still average 208.8.
- **Every late greedy extinction is a single-step wipeout.** In all 71, the
  population is at 10 or more (median 15) one step and at 0 the next. In 68 of
  the 71, the agent had just chosen cost 2.5. Example, world 3, the last steps
  before death:

  ```
  population  29 24 29 34 31 22 14 12 13 19 22 19 → 0
  move cost   every step 2.5
  ```

- **Why one step can wipe out the population.** Foragers pay the move cost
  *before* they eat (`step` runs before `eat` in `env_step`). Cost 2.5 therefore
  kills every forager holding 2.5 energy or less, whatever food is on its tile.
  On the fatal steps, the mean energy the agent saw was about 1.86 (median),
  against 2.12 across all live steps, so most of the population was below the
  line. The agent uses cost 2.5 on 61% of steps and doesn't back off when
  energy is low.
- **For the first time, sampled beats greedy** (403.7 vs 389.4) and has fewer
  extinctions (55 vs 84). Greedy is deterministic, so whenever the state looks
  "cull", it culls every time. Sampling sometimes picks a lower cost, and that
  occasionally saves a population on the edge. Sampled late extinctions are
  not all one-step wipeouts: only 51% start from 10 or more.
- **The learned rule is sharper.** Mean cost by population, after the start:
  0.19 at 0–4 foragers, 1.28 at 5–9, 2.36 at 10–14, and 2.5 from 15 upward.
  The switch is around 7–10, lower in the band than the baseline's 12.

### Interpretation (hypotheses, not tested)

- **Resolution fixed the start-up.** With the band spanning a full unit of
  input, the agent can tell 5 foragers from 15 and eases off before the
  population reaches zero.
- **The late wipeouts are a rare, sharp event the agent hasn't learned to
  avoid.** Whether cost 2.5 is fatal depends on how many foragers hold 2.5
  energy or less. The agent sees only the mean energy, not that distribution.
  Wipeouts are also rare in training, so there is little signal to learn from.
  The discount (γ = 0.99, roughly a 100-step horizon) also makes a wipeout late
  in an episode look cheap compared with the reward gained by holding the
  population low.

### Caveats

- One training seed per setup. The 13 vs 56 early-crash drop is large enough
  to trust; the smaller differences are not.
- The greedy/sampled reversal could be a quirk of this one trained policy.

---

## Next experiments

1. **Combine both observations** (`population_band` + `growth`). Growth removed
   the late extinctions in Experiment 1, and band scaling removed the early
   ones here. The question is whether together they remove both.
2. **Give the agent the energy information the cull depends on,** for example
   the fraction of foragers at or below each move cost, or at least the minimum
   or low-percentile energy instead of only the mean. That directly targets the
   one-step wipeout.
3. **Several training seeds** for each setup, to put error bars on these
   differences.
4. **A hand-written rule as a reference,** for example "cost 2.5 only when mean
   energy is high and the population is above X", to see how close a simple
   controller gets now that we know the failure modes.

---

## Data for the website

Everything the website needs comes from stored results, not one-off scripts.

- **`results/floor_control/comparison.csv`:** one row per experiment × policy
  (baseline, growth, band_scaled × PPO greedy, PPO sampled, best fixed,
  random). It has every number in this log's tables: mean and median return,
  extinctions bucketed by step (0–19, 20–99, 100–499), median extinction step,
  return for surviving and extinct worlds, and late-extinction details (how
  many started from 10 or more foragers, how many followed a cost of 2.5, the
  mean energy seen on the fatal step). Generated by `foragers-compare`.
- **`results/floor_control/web/<experiment>/`:** compact episodes for the
  site's viewer, plus an `index.json` with each episode's metrics. For each
  experiment, there are the median and worst PPO-greedy episodes and the median
  best-fixed episode for the floor band. The `baseline/` folder also has the
  other three bands from band_sweep. This folder is gitignored and can be
  regenerated.
- **Per-experiment detail:** `results/<experiment>/episodes.csv` (every world)
  and `raw/*.npz` (every step).

Not yet built: figures comparing the experiments. Also, none of the recorded
episodes shows an Experiment 2 late wipeout. The recorded worst world is an
early crash, so showing a wipeout would need a "late extinction" episode
selector in the config.

---

## Reproducing

```bash
uv run foragers-experiment experiments/band_sweep.toml --force       # baseline (all four bands)
uv run foragers-experiment experiments/floor_growth_obs.toml --force # experiment 1
uv run foragers-experiment experiments/floor_band_scaled_pop.toml --force # experiment 2

# comparison table and web episodes
uv run foragers-compare results/floor_control/comparison.csv --band floor \
    baseline=results/band_sweep growth=results/floor_growth_obs \
    band_scaled=results/floor_band_scaled_pop
for e in baseline:band_sweep growth:floor_growth_obs band_scaled:floor_band_scaled_pop; do
  uv run python src/foragers_world_rl/export_web.py results/floor_control/web/${e%%:*} \
      --experiment results/${e#*:}
done
```

Each output directory has `meta.json` (resolved config, git commit, versions)
and, if the working tree had uncommitted changes, `git.diff`.
