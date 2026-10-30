# Band sweep — context for the write-up

This is a handoff for writing a blog post about this experiment. It contains
everything needed to write it without re-running anything: what was built, what
was measured, the numbers, what each figure shows, and what the data does and
does not support.

The post is for Ray's personal website (the project is a portfolio piece). It
will show the figures below and embed raw episodes in an animated viewer.
Tone, length and audience are Ray's call; check with him rather than assuming.

Every number here comes from `summary.csv`, `episodes.csv`,
`training/*.csv` or `figures/data/*.csv` in this directory. If you need a
number not listed here, compute it from those files rather than estimating.

---

## 1. One-paragraph summary

A population of simulated foragers lives on a 32×32 grid. The only control
is a single global setting, the energy each forager pays to move. Raise it and
foragers starve, lower it and they breed. An RL agent sees three numbers about
the world (total food, population, mean energy) and picks one of 12 move costs
each step. It is rewarded for every step the population stays inside a target
band. PPO was trained separately for four bands: wide (80–140), tight (75–85),
near the floor (0–30) and near the ceiling (120–154). In every band it beats
the best fixed setting by a wide margin, because it learns a simple threshold
controller that flips the move cost up when the population is high and down
when it is low. How well that works depends heavily on the band. It is
near-perfect for a wide band in the comfortable middle of the range, and much
weaker when the band is narrower than the population's step-to-step swing
(tight) or sits next to extinction (floor).

---

## 2. Setup

### Environment (`src/foragers_world_rl/world.py`)

| Setting | Value |
|---|---|
| Grid | 32 × 32 |
| Max population (array slots) | 256 |
| Initial population | 16 foragers, 10 energy each |
| Episode length | 500 steps; ends early on extinction |
| Movement | random walk, one tile per step |
| Eating | foragers on a tile split its food evenly; the tile is then emptied |
| Food regrowth | each tile gains 1 food with probability 0.2 per step, capped at 5 |
| Starvation | a forager dies at 0 energy |
| Breeding | asexual: at ≥ 4 energy a forager splits, giving the child half its energy |

The environment is written entirely in JAX: a pure step function, `vmap` over
foragers and over parallel worlds, `lax.scan` over steps, `jit` throughout. 256
worlds × 500 steps run in about 1.5 s on a laptop CPU.

### The control problem

- **Observation (3 floats):** total food (fraction of maximum), population
  (fraction of 256), mean energy (relative to the breeding threshold). Global
  aggregates only, with no spatial information and no trend or history.
- **Action:** one of 12 move costs, 0.0 to 2.75 in steps of 0.25, applied to
  every forager for that step.
- **Reward:** +1 for each step the population is inside the band (inclusive),
  otherwise 0. **Return = number of steps in band, out of 500.**
- **Band:** a parameter of the environment; one policy is trained per band.

### Agent (`src/foragers_world_rl/ppo.py`)

PPO in the style of PureJaxRL (github.com/luchris429/purejaxrl). The whole update
(rollout, advantages and gradient steps) runs on device as one jitted function.

| Setting | Value |
|---|---|
| Network | separate actor and critic MLPs, 2 × 64 tanh, hand-written in JAX |
| Optimiser | Adam (optax), lr 2.5e-4 annealed linearly to 0, grad-norm clip 0.5 |
| Rollout | 256 parallel worlds × 128 steps per update |
| Training | 5M env steps (152 updates), about 2 min per band on CPU |
| PPO | 4 epochs × 4 minibatches, clip 0.2, γ 0.99, GAE λ 0.95, entropy 0.01, value coef 0.5 |
| Seed | training seed 0 |

Two implementation details are worth mentioning if the post goes into how it
works:

- **Hitting the step limit is not treated as death.** Extinction ends the
  episode with no future reward. Reaching step 500 instead bootstraps from the
  critic's value of the final state, because the observation has no clock and
  the critic cannot know an end is coming. Standard PureJaxRL treats both the
  same way.
- **Rewards are scaled by (1 − γ) = 0.01 during training,** so value targets
  stay between 0 and 1.

### Evaluation protocol

- 256 evaluation worlds per band, evaluation seed 0. Every policy in a band
  sees the same 256 starting worlds.
- **Policies compared:**
  - **PPO (greedy):** always takes the most likely action. This is the main result.
  - **PPO (sampled):** samples from the policy, which is how it behaved during training.
  - **Best fixed action:** the single move cost with the highest mean return
    for that band, found by scoring all 12. This is the bar a learned policy must
    clear to show it is reacting to the observation rather than picking one good
    setting.
  - **Uniform random.**
- Steps after an episode ends (extinction) are excluded from every metric.

---

## 3. Headline results

Return is steps in band out of 500, as a mean over 256 worlds (± standard
deviation).

| Band | PPO greedy | PPO sampled | Best fixed action | Random |
|---|---|---|---|---|
| **wide** 80–140 | **496.0** ± 1.7 | 489.8 ± 3.7 | 297.6 ± 37.5 (cost 1.5) | 36.3 |
| **ceiling** 120–154 | **448.2** ± 9.3 | 419.8 ± 11.7 | 203.0 ± 29.7 (cost 1.25) | 14.6 |
| **floor** 0–30 | **349.0** ± 192.4 | 239.3 ± 196.0 | 83.6 ± 48.4 (cost 1.75) | 37.1 |
| **tight** 75–85 | **270.4** ± 11.8 | 218.0 ± 11.7 | 65.6 ± 11.2 (cost 1.5) | 9.3 |

| Band | Extinct (PPO greedy / sampled / best fixed / random) |
|---|---|
| wide | 0% / 0% / 0% / 94% |
| ceiling | 0% / 0% / 0% / 94% |
| floor | 27% / 65% / 69% / 94% |
| tight | 0% / 0% / 0% / 94% |

Largest deviation from the band, per episode, after the population first
enters it (mean over worlds, in foragers):

| Band | Overshoot: PPO greedy vs best fixed | Undershoot: PPO greedy vs best fixed |
|---|---|---|
| wide | 0.4 vs 47.4 | 3.0 vs 54.1 |
| ceiling | 23.4 vs 83.4 | 15.5 vs 56.6 |
| floor | 24.4 vs 76.0 | n/a (see caveats) |
| tight | 22.0 vs 102.4 | 16.8 vs 49.1 |

---

## 4. Findings, band by band

### Wide (80–140): essentially solved
- Greedy PPO is in band for 496 of 500 steps. The ~4 missed steps are the
  start-up: the population needs about 3.4 steps on average to grow from 16 to
  80 (`mean_steps_to_enter` 3.35), and no policy can skip that.
- The worst episode still scores 490. Mean largest overshoot is 0.4 foragers.
- The best fixed action (cost 1.5) gets 297.6. The population oscillates around
  the band under any constant cost, spending about 160 steps below it and 38
  above.

### Ceiling (120–154): works, with more slack
- 448.2 vs 203.0 for the best fixed action. Never extinct.
- Deviations are bigger than in the wide band (overshoot 23.4, undershoot 15.5).
  A population near 135 swings more per step than one near 100, and the band is
  about half as wide (35 vs 61).
- Uses only costs 1.0, 1.25 and 1.5. It never needs a high cost, because a large
  population mostly needs to be kept fed.

### Tight (75–85): about four times better than fixed, but limited
- 270.4, so about 54% of steps in band, vs 65.6 for the best fixed action.
- The band is 11 foragers wide, but PPO's typical worst swing is 22 above and
  17 below. A population of about 80 often changes by 10 or more in a single
  step, so even good switching overshoots. (The close-up in figure 04b shows this.)
- **Probably undertrained.** This is the only band whose learning curve is
  clearly still rising at 5M steps (training-episode return: 208.7 at
  3.5–4M steps, 218.5 over the last 10 updates). A longer run might score
  higher, but that hasn't been tested.

### Floor (0–30): bimodal, and the failure is at the start
- Mean 349.0, but the mean hides two groups. **74% of worlds score above 400,
  and 22% score below 50.** The median is 455. Worlds that survive average
  461.5.
- **27% of worlds go extinct, and 82% of those extinctions (56 of 68) happen in
  the first 20 steps.** After the start, only 12 of 256 worlds (about 5%) die.
- **The mechanism visible in the raw data:** the 16 starting foragers all have
  10 energy, so they breed immediately and the population roughly triples to
  about 50 within 2–3 steps, well over the band. The agent responds with its
  maximum cull (cost 2.5) and keeps it up while the population falls through
  the band. Whether a few foragers survive the crash is close to chance. The
  peak at the start is the same (median 51) in worlds that die and in worlds
  that survive.
  - Example extinct world 3: population 32, 51, 30, 19, 3, 0 under cost 2.5 for
    six steps.
  - A surviving world bottoms out at 4–5 and recovers.
- Once established, the controller holds a small population inside 0–30 well.
  This is the "steady state" visible in figures 03 and 04.
- Its action use is extreme: cost 2.5 on 65% of steps, 2.0 on 23%, and
  **0.0 on 12%**, the only band where PPO uses the no-cost action. The learned
  rule is "cull hard unless nearly dead". Only at a population of about 0–5
  does it switch to cost 0.
- **Sampled is much worse here** (65% extinct vs 27%). Most sampled extinctions
  happen later (median step 25.5, vs 9 for greedy): one random high-cost step
  when the population is in single digits can finish it.
- **Why the floor is hard (these are hypotheses, not tested):**
  - With the population observed as a fraction of 256, the whole band is only
    0.00–0.12 of the input range.
  - The observation has no trend, so the agent can't tell a crashing population
    from a recovering one.
  - At around 5–15 foragers, chance deaths and births dominate.

### Across bands
- **PPO wins every band,** by 1.7× (wide) to 4.2× (floor and tight) over the
  best fixed action.
- **It learns a threshold controller.** Figure 08 plots the average move cost
  PPO picks at each population. Each band has a low-cost level and a high-cost
  level, with a sharp switch between them, **and the switch sits at or near the
  middle of the band:**

  | Band | Low cost | High cost | Switches at a population of about | Band middle |
  |---|---|---|---|---|
  | wide | 1.0 | 1.75 | 100 | 110 |
  | tight | 1.0 | 2.0 | 78 | 80 |
  | ceiling | 1.0 | 1.5 | 137 | 137 |
  | floor | about 1.4 (a mix of 0 and 2.5) | 2.5 | 12 | 15 |

  The switch point is where the average cost crosses halfway between the two
  levels, in 5-forager bins (`figures/data/08_policy_map.csv`). In plain terms,
  the agent has learned to aim for the middle of whatever band it is given.
- It switches action on most steps: about 290–350 switches per 500-step episode
  in the wide, tight and ceiling bands, and 193 in the floor band. It behaves
  like a thermostat, not a smooth controller.
- **Greedy beats sampled in every band.** The gap is small where the margin is
  comfortable (wide: 496 vs 490) and large near extinction (floor: 349 vs 239).
- **PPO passes the best fixed action early,** after 0.7–1.0M of the 5M training
  steps in every band (wide 0.69M, tight 0.75M, ceiling 0.92M, floor 0.98M).

---

## 5. Figures

All figures are in `figures/`, each as `<name>_light.png`, `<name>_dark.png`,
`<name>_light.svg` and `<name>_dark.svg`. The exact plotted numbers are in
`figures/data/<name>.csv`; use them for tables or for redrawing a chart
interactively. Colour always means policy: **blue = PPO, orange = PPO
sampled, green = best fixed action**. A grey shaded region is always the
target band.

**Ray's primary figures are `01_returns` and `08_policy_map`;** both were
deliberately simplified for the post. The rest are supporting material. Use
them only where they add something, and don't let them crowd the post.

The primary figures carry plain, descriptive titles only ("Average return of
PPO vs best fixed move cost", "PPO policy: average move cost vs population").
That is Ray's preference: **the interpretation belongs in the post's text or
captions, not in the figure.**

| File | Shows | Main takeaway |
|---|---|---|
| **`01_returns`** (primary) | Grouped bars: mean steps in band (of 500) per band, PPO (greedy) vs the best fixed move cost, value on each bar | PPO beats the best fixed setting in every band: 496 vs 298, 270 vs 66, 349 vs 84, 448 vs 203. The bars are means; the floor band's spread (bimodal, 27% extinct) is not shown, so mention it in the text |
| `02_time_allocation` | Each policy's 500 steps split into: before first entering the band, in band, above, below, and after extinction | Where the missing reward goes. Fixed actions spend most of their time above or below the band. In the floor band the loss is mostly extinction |
| `03_population_over_time` | Median population across 256 worlds per step, with a 10th–90th percentile ribbon, PPO greedy vs best fixed | PPO keeps the population in a narrow ribbon inside the band. Under the fixed action in the floor band, the median population falls to 0 around step 320 |
| `04_example_episodes` | One median PPO-greedy episode per band (the same world as the embedded episode): population above, chosen cost below, full 500 steps | The overall shape of control. The cost panel is dense on purpose, showing constant switching |
| `04b_example_zoom` | The same episodes, steps 200–259, with each step marked | Read this to see individual decisions: the population ticks up, the cost goes up the next step, and vice versa |
| `05_deviation` | Distribution of each episode's largest overshoot and undershoot (middle 50%, 10th–90th percentile, median) | PPO cuts the mean worst deviation about 3–5× in the ceiling, floor and tight bands, and almost removes it in the wide band (overshoot 47 → 0.4). The floor band's undershoot is marked n/a |
| `06_training` | Return of training episodes per update, with the best fixed action as a dashed line | All bands pass the fixed baseline by about 1M steps. Tight is still rising at 5M |
| `07_action_usage` | Heatmap of the share of steps at each of the 12 costs (PPO greedy); the best fixed cost is outlined | PPO uses 2–4 costs per band, roughly bracketing the best fixed cost. The floor band uses the extremes |
| **`08_policy_map`** (primary) | One panel per band: the average move cost PPO picks at each population (line), with the band shaded | The learned rule is a switch: low cost below a threshold, high cost above it, with the threshold near the middle of the band |

Notes for describing figures accurately:
- **`06_training` shows training episodes,** which use sampled actions, so its
  final values sit below the greedy evaluation numbers. Don't compare its
  y-values directly with figure 01.
- **`08_policy_map` is an average over population only,** but the policy also
  sees food and mean energy. Where the line sits between the two levels, the
  agent picks low on some steps and high on others depending on those inputs.
  It isn't random: the greedy policy is deterministic given its observation.
  The first 10 steps of each episode are left out, because the population is
  still growing from 16 and passes through levels it never returns to.
  Populations with fewer than 20 steps of data are not drawn, which is why each
  line covers only the range that band actually visits.
- **Percentiles in 03 and 05 are across worlds,** not across time.

---

## 6. Embeddable episodes (`web/`)

`web/` holds compact episodes for the site's viewer, listed with their metrics
in `web/index.json`. For each band:

| Episode | What it is |
|---|---|
| `<band>__ppo_greedy__median` | the world whose return is closest to the median (the typical case) |
| `<band>__ppo_greedy__worst` | the lowest-return world |
| `<band>__best_fixed__median` | the typical world under the best fixed action |

**These are the exact worlds that were scored,** not re-simulations, so any
number quoted alongside an episode is that episode's real metric (it's in
`index.json` under `metrics`).

Notable ones:
- **`floor__ppo_greedy__worst`:** extinct in 5 steps (return 2). This is the
  start-up crash described above, and a good illustration of the floor-band
  failure.
- **`floor__ppo_greedy__median`:** a surviving world, return 455.
- **`wide__ppo_greedy__median`:** return 496, the "it works" example.
- **`tight__best_fixed__median` vs `tight__ppo_greedy__median`:** return 65 vs
  271, the starkest side-by-side comparison.

Format: the same as the site's existing compact runs.
- Each frame has `f` (base64 food grid, one byte per cell), `p` (cell index of
  each living forager) and `n` (population).
- New fields:
  - `actions` and `rewards`: one per step, so length = frames − 1.
  - `moveCosts`: maps an action index to its cost.
  - `band`, `bandName`, `policy`, `policyDetail`, `world`, `which`, `return`.
- `action` and `moveCost` are null, because the action changes every step. The
  site's player may need updating to read `actions`.

---

## 7. Metric definitions

From `episodes.csv` (one row per band × policy × world). `summary.csv` holds
the means and extremes over the 256 worlds.

| Metric | Definition |
|---|---|
| `return` | steps with the population in band (inclusive), out of 500 |
| `length` | steps until the episode ended: 500, or the extinction step + 1 |
| `steps_to_enter` | 0 if the starting population (16) is already in band (only the floor band); t + 1 if it first entered after step t; −1 if never |
| `max_overshoot` / `max_undershoot` | the largest distance above the top / below the bottom of the band, **after first entering**, in foragers. NaN if it never entered |
| `steps_above_after_entry` / `steps_below_after_entry` | steps spent above / below the band after first entering |
| `extinct`, `extinct_step` | whether, and at which step, the population hit 0 |
| `mean_cost`, `action_switches`, `n_action_<k>` | the average move cost chosen, the number of step-to-step action changes, and how many steps used each action |

The raw per-step data for every world is in `raw/<band>__<policy>.npz`:
- arrays `pops`, `actions`, `rewards`, `dones`, `live` and `obs`, each of shape
  (256, 500);
- `pops[w, t]` is the population after step t;
- `obs[w, t]` is what the agent saw when choosing `actions[w, t]`.

---

## 8. Caveats: what not to claim

- **Floor-band undershoot is 0 by definition,** since nothing is below 0. Don't
  report it as perfect control; extinction is the floor band's failure on the
  low side.
- **The floor-band reward includes the extinction step,** because 0 is within
  0–30. The episode then ends, so extinction always costs all the remaining
  reward; there is no reward hacking. But "dying" earns one reward on its way
  out.
- **One training seed per band.** The differences between bands are large
  compared with the spread across evaluation worlds, but nothing here measures
  the variation between training seeds. Avoid claims like "PPO reliably
  learns X" without that caveat.
- **One fixed set of hyperparameters,** tuned for nothing in particular
  (PureJaxRL defaults). The tight band looks undertrained.
- **The best fixed action is chosen on the evaluation worlds themselves,** which
  slightly favours the baseline. That only strengthens the PPO-beats-fixed
  conclusion.
- **The "why" claims in section 4 are hypotheses** (step-to-step swing vs band
  width, input scale, demographic noise, no trend in the observation). They
  are consistent with the data but untested. Present them as likely
  explanations or as next experiments.
- **The observation is global only.** The agent has no spatial information,
  and the action is a single global setting.

---

## 9. Engineering notes the post could use

These are things that happened while building this, if the post wants a "how it
was built" or "lessons" section:

- **The 5× speed-up wasn't where it was expected.** Evaluation took 8 s per
  policy, which looked like JAX recompiling every call. Timing compile and run
  separately showed compiling took 0.5 s. The real cost was the breeding step:
  it paired parents with free slots using a 256×256 matching matrix
  (O(N²) per step, about 8 billion element operations per evaluation).
  Replacing it with an O(N) lookup and scatter (`jnp.nonzero(..., size=N)` and
  `.at[].set(mode="drop")`) brought it to 1.6 s, with identical results.
- **An evaluation bug that only this experiment would have exposed:** JAX's
  `scan` has a fixed length, so a world keeps being stepped after it goes
  extinct. With a band that includes 0, every dead step would have scored as
  in band. Returns are now counted only up to the first `done`.
- **Recorded episodes now replay exactly the scored worlds.** Originally, the
  viewer recording used different random keys from the batched evaluation, so
  the episode you watched was a lookalike. Now `record(world=i)` derives keys
  the same way as `evaluate`, and a test checks that they match step for step.
- **The PPO is about 400 lines,** with the environment, rollout, advantages and
  update all in JAX. The 5M-step training takes about 2 minutes on a laptop CPU,
  and passes the best fixed action within roughly the first 15–25 seconds.

---

## 10. Reproducing

```bash
uv sync --group plots
uv run foragers-experiment experiments/band_sweep.toml --force    # ~12 min: train, evaluate, record
uv run --group plots foragers-plots results/band_sweep           # figures + figures/data
uv run python src/foragers_world_rl/export_web.py results/band_sweep/web --experiment results/band_sweep
```

Provenance is in `meta.json` (resolved config, environment constants, library
versions and git commit) and `config.toml`. **This run was made from a
working tree with uncommitted changes:** `meta.json` records base commit
`3340172` with `dirty: true`, and the changes are saved in `git.diff`. The plan
is to commit and re-run before publishing, so the results point at a clean
commit. An earlier identical run reproduced every number exactly, so the
figures should not change. If they do, the numbers in this document need
updating.

## 11. File map

```
results/band_sweep/
  WRITEUP_CONTEXT.md        this file
  config.toml               the experiment config that was run
  meta.json                 provenance: resolved config, env, versions, git, best fixed action per band
  git.diff                  uncommitted changes at run time
  summary.csv               one row per band x policy (all 12 fixed actions included)
  episodes.csv              one row per band x policy x world: every metric in section 7
  training/<band>.csv       learning curve per update
  raw/<band>__<policy>.npz  per-step arrays, all 256 worlds
  params/<band>.pkl         trained weights
  figures/                  9 figures x light/dark x png/svg
  figures/data/             the plotted numbers per figure
  web/                      compact embeddable episodes + index.json (gitignored, regenerable)
  episodes/                 full viewer-format recordings (gitignored, regenerable)
```

To browse the full recordings locally, run `uv run python -m http.server` from
the repo root and open `localhost:8000/viewer/?dir=../results/band_sweep/episodes/`.

## 12. Possible follow-ups (if the post ends with "next")

- **A trend observation** (change in population since the last step). This is
  the most direct test of the floor and tight hypotheses.
- **Population scaled to the band** instead of to 256, for the floor band.
- **Longer training for the tight band.**
- **Several training seeds per band,** to put error bars on the band-to-band
  comparison.
- **A harder environment:** spatial actions (poison or plant food in one
  region), per-region observations, and heritable traits that let the
  population evolve while it is being controlled.
