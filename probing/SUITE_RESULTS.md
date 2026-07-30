# Objective-grounded probe suite — quick pass (2026-07-27)

`probing/suite.py` — linear **and** non-linear probes over the training-step range,
both encoders + the predictor, with random-encoder / raw-input controls.

- **Models**: today's run at step 5k / 20k / 50k, plus the Jul-26 run's
  `ALARM_step18450_collapse` snapshot, plus `rand` (same architecture,
  random init) and `raw` (flattened 10x110 input window, action columns removed
  because the model never sees actions; hidden-object columns zeroed for masked
  probes). All checkpoints loaded with config inferred from the state dict —
  every one is the same 6-block/3-block architecture.
- **Data**: probe-train = 24,000 windows (shard_00310), probe-val = 8,000
  windows (shard_00312, different replays). Windows collected exactly like
  training (`gap=2`, physical norm, drop_noise), 10 visible states = 1.8 s,
  plus 10 future states for transfer targets. Horizons are true seconds
  (the older `probes.py` labels are 2x off: its "+0.5s/+1s" are +1s/+2s).
- **Probe families** per target: `lin` ridge/logistic - `knn` (k=32, cosine) -
  `rff` RBF kernel ridge (2048 random features) - `mlp` (1x512). Regression =
  val R^2, binary = val ROC-AUC.
- **Substrates**: context encoder (receives masked views, gets gradients), EMA
  target encoder (produces the training targets; `tgt_*` pools), and the
  predictor's outputs decoded to physical units (`pred*`), extracted under the
  exact training mask (anchor at state 0, object hidden states 1-9).

Raw JSON: `probing/suite_results/*.json`. Runtime: ~5 min for the whole pass.

---

## 1. What the objective actually optimizes (what is fair to test)

Per sample, ONE object (self .10 / opp .35 / ball .45 / env .05 / boost .05) is
hidden in states 1–9; the context encoder sees the other objects' **full 1.8 s
trajectories** plus the hidden object's state-0 anchor; the predictor must
output the EMA target encoder's latents at the hidden slots (smooth-L1).

**Directly optimized ("direct" probes):**
- cross-object routing: visible tokens must carry the hidden object's
  within-window trajectory (mostly ball, then opp);
- the predictor's reconstruction of that trajectory *on the target-latent
  manifold*, given everyone else's full window.

This is **completion / inpainting-with-context, not forecasting**: the model
always sees every *other* object up to state 9. Nothing in the loss rewards
representing what happens **after** the window, or predicting any object from
*past-only* information.

**Never optimized ("transfer" probes — failure here is not a training bug, but
this is what PPO consumes):** forecasting beyond the window; game events;
actions (controller inputs are not model inputs — self-action decode = BC
readout, opp-action decode = intent prior); world-frame pose; metric
calibration of latents; goals (drop_noise removes them from every window).

**Structural pressure worth naming:** an object's target latents are only
constrained by the loss *when that object is masked* — and being a target
rewards being **predictable**. So masking pressure pushes the most-masked
object's latents toward *lower information* (ball, 45%), while the never-masked
slots (env/boost, 5%) have no reason to retain their own content at all. Both
predictions are confirmed below.

---

## 2. Geometry / run health

Val-set geometry of the EMA target encoder (my numbers are cross-model
comparable within this pipeline; not comparable to the training panel, which
reads the online encoder on training batches):

| model | erank(pool) | top1 share | cos | per-object token erank (self/opp/ball/env/boost) | prenorm RMS ctx/tgt |
|---|---|---|---|---|---|
| step5k        | **2.6** | 0.78 | 0.999 | 2.9 / 2.6 / 2.6 / 1.8 / 3.4 | **4.3 / 5.4** |
| step20k       | 62.0 | 0.08 | 0.38 | 58 / 53 / 46 / 38 / 56 | 1.12 / 1.12 |
| step50k       | 61.0 | 0.06 | 0.42 | 58 / **37** / 45 / 43 / 58 | 1.05 / 1.06 |
| collapse18450 | **78.2** | 0.05 | **0.18** | 74 / 69 / 68 / 67 / 74 | 1.48 / 1.23 |
| rand          | 26.9 | 0.13 | 0.999 | 15 / 18 / 9 / 6 / 9 | 1.48 / 1.48 |

- **Today's run was in the trivial basin at 5k** (erank 2.6, cos 0.999,
  predictor-target smooth-L1 at the ~0.0003 trivial floor for every object,
  inflated pre-norm scale). It escaped between 5k and 20k — i.e. around the end
  of LR warmup (10k). This explains why today's run produced **zero ALARM
  snapshots**: alarms stay disarmed until "healthy", and the arming logic never
  saw a healthy baseline early; after escape, erank stayed ~61 so nothing
  tripped.
- **After 20k the geometry is static (erank 62 -> 61) while probes decay** —
  the erank/cos panel is blind to this failure mode. CKA(step20k, step50k) =
  0.62: the representation keeps rotating/reorganizing under a constant
  spectrum. CKA of anything to rand ~ 0.44; CKA(step5k, step50k) = 0.14.
- The Jul-26 warn snapshot (`collapse18450`) has the **healthiest target
  encoder of all four** (its rupture hit the online encoder; EMA lag shielded
  the target). Its run reached erank 78 / cos 0.18 by 18k — better geometry
  than today's run ever achieved.
- Opp token erank fell 53 -> 37 over 20k->50k; ball predictor loss *halved*
  (0.0037 -> 0.0019) over the same span in which every physical read of ball
  info fell — **the loss improves by making targets emptier, not predictions
  better.**

## 3. Directly-optimized capabilities — the objective works, peaks ~15-20k

val R^2, probe-train n=24k. Controls: rand / raw shown as `r:`/`w:`.

| probe (family) | 5k | 20k | 50k | collapse18450 | controls |
|---|---|---|---|---|---|
| masked ball -> ball pos, pooled (lin) | 0.448 | **0.543** | 0.438 | 0.514 | r 0.279, w 0.193 |
| ... (mlp) | 0.287 | **0.621** | 0.546 | 0.565 | r 0.278, w 0.331 |
| masked ball -> ball pos, s9-tokens cat (lin) | 0.518 | **0.701** | 0.667 | 0.648 | r 0.509 |
| masked opp -> opp pos, pooled (lin/mlp) | 0.48/0.30 | **0.65/0.72** | 0.57/0.67 | 0.54/0.60 | r 0.39/0.35 |
| masked opp -> opp pos, cat (lin) | 0.554 | 0.839 | **0.843** | 0.730 | r 0.565, w 0.221 |
| masked self -> own boost (lin) | 0.461 | 0.752 | **0.751** | 0.632 | r 0.512, w 0.671 |
| masked self -> local vel, cat (lin) | 0.566 | 0.743 | **0.796** | 0.701 | r 0.566 |
| predictor(ball) s9 -> ball pos (lin) | 0.234 | **0.665** | 0.470 | 0.590 | r 0.223, w 0.193 |
| predictor(ball) s1/s5 -> pos@s1/s5 (lin) | 0.16/0.18 | **0.86/0.71** | 0.59/0.45 | 0.62/0.58 | r 0.47/0.31 |
| predictor(opp) s9 -> opp pos (lin) | 0.250 | **0.828** | 0.801 | 0.609 | r 0.324 |

Reads:
1. **The core JEPA claim is confirmed at 20k**: the full trained pathway
   (context encoder + predictor) reconstructs the hidden ball at R^2 0.67–0.86
   across the window, far above every control; masked-recovery margins over
   rand are large for ball/opp/self. The objective does what it says.
2. **Everything ball-related peaks at ~15–20k and then decays** (predictor ball
   decode 0.665 -> 0.470; pooled masked-ball 0.543 -> 0.438) while the
   *training loss on ball keeps improving* — target-information erosion, not
   optimization progress. Opp holds up much better (pred-opp 0.83 -> 0.80; cat
   0.84 flat) — consistent with erosion pressure scaling with masking share
   (ball 45%).
3. **Linear vs non-linear**: after the peak, MLP probes lose less than linear
   ones (masked-ball pooled at 50k: lin 0.438 vs mlp 0.546; routing lin 0.395
   vs mlp 0.509) — part of the information is retreating into non-linear code,
   part is genuinely gone. kNN is far below everything — the latent *metric*
   is poor even when content is decodable; nearest-neighbour retrieval on
   these latents would not work.
4. **Token readouts >> pooled means, everywhere** (cat vs pooled: +0.16 to
   +0.27 R^2), and the pooled readout decays fastest. Any downstream consumer
   must take tokens, not means.
5. Self recovery works via the documented frame leak (local vel 0.80 from the
   trilateration inversion) — as designed/predicted, it is an easy task.

## 4. Information preservation — training destroys the identity map

Linear decode of an object's own state from its own (target-encoder) token:

| token -> its own feature | 5k | 20k | 50k | collapse18450 | rand |
|---|---|---|---|---|---|
| ball -> rel pos | 0.997 | 0.853 | **0.705** | 0.883 | 1.000 |
| ball -> rel vel | 0.999 | 0.702 | **0.599** | 0.745 | 1.000 |
| opp -> rel pos  | 0.989 | 0.948 | 0.922 | 0.917 | 0.995 |
| self -> own boost | 0.996 | 0.929 | 0.858 | 0.866 | 0.998 |
| env -> secs remaining | 1.000 | **0.046** | 0.027 | 0.214 | 1.000 |
| boost -> 3 pad timers | 0.777 | **0.121** | 0.158 | 0.110 | 0.927 |

The never-masked token slots (env 5%, boost 5%) are **wiped** — the encoder
overwrites them as workspace for the dominant task, exactly as the objective
incentivizes. The most-masked token (ball) erodes steadily; opp (35%) erodes
slowest. A random encoder preserves everything at ~1.0 for free.

## 5. Transfer battery — nothing the objective didn't optimize was learned

Across **all four probe families** and **all checkpoints**, the trained encoder
never beats the random-init control on a single forward-looking target:

| probe | best trained (any ckpt/family) | rand | raw |
|---|---|---|---|
| ball rel pos +0.4s (lin) | 0.957 | **0.986** | 0.223 |
| ball rel pos +1s (lin)   | 0.899 (5k) | **0.901** | 0.202 |
| ball rel pos +2s (lin)   | 0.727 | **0.730** | 0.176 |
| ball local vel +1s — bounce/touch dynamics (lin) | 0.377 | 0.367 | 0.092 |
| opp rel pos +1s (lin)    | 0.941 | **0.945** | 0.210 |
| own displacement +1s (lin) | 0.744 | **0.752** | 0.294 |
| ball grounded <=1s (AUC, base .38) | 0.774 | 0.768 | **0.855** |
| opp closer to ball @+1s (AUC, base .50) | 0.861 | **0.909 (mlp)** | 0.798 |
| SELF throttle+steer (BC readout, R^2) | 0.257 | 0.261 | 0.146 |
| OPP throttle+steer (intent, R^2) | 0.127 | **0.179 (mlp)** | 0.146 |
| OPP boost held (AUC, base .22) | **0.828 (50k)** | 0.701 | 0.754 |

- The one (small) trained edge: opp-boost-held at 50k (+0.07 AUC over raw).
  Everything else: ties or losses to rand, at every step count.
- **Sample efficiency moves the wrong way**: with 2k probe labels, rand reads
  ball+1s at 0.886; step20k at 0.773; step50k at 0.677. Training makes the
  representation *harder* to read from few labels — the opposite of what an RL
  policy/value head needs.
- The real featurizer here is the **architecture prior**: object tokens +
  agent-frame features + attention pooling give future-ball R^2 0.90 linearly
  *at random init* (vs 0.20 from the raw window). Training as currently posed
  subtracts from that prior on every transfer axis.
- (Probe defect to fix in the full pass: `ball_impulse` base rate is 0.963 —
  threshold too loose; tighten before reading that row.)

## 6. Collapse forensics — two distinct failure modes

**Mode A (Jul-25 / Jul-26 runs): abrupt rank rupture.** The Jul-26 attribution
log shows erank climbing healthily 75 -> 164 through 18.2k, then in <=200 steps:
top-eigenshare 0.10 -> 0.38, loss 0.0115 -> 0.178 (15x), cos 0.17 -> 0.38.
Precursors: an EMA-divergence spike (0.033 -> 0.096) ~2.5k steps earlier at
~15.8k, after which predictor context-sensitivity (`pred_cos`) degraded
steadily from 0.28 -> 0.55 into the rupture. The alarm caught it (by design).

**Mode B (today's run): slow information erosion under healthy-looking
geometry.** Born in the trivial basin (never armed -> no alarms), escaped at
warmup end, peaked ~15-20k, then eroded: linear readability decays everywhere,
loss keeps improving, erank/cos stay flat. The existing panel **cannot see
Mode B** — it needs an information metric, not a geometry metric. The cheapest
in-training canary: every N k steps, ridge-decode ball-rel-pos from the ball
token and from the masked-visible pool on a fixed cached batch; alarm on
sustained decline. (This suite is the offline version.)

## 7. Is this objective right for PPO? — No, not as-is

What a PPO featurizer must provide: linearly (or near-linearly) readable
current state, **forward-predictive** features (value/policy look-ahead),
opponent intent, and all of it **sample-efficiently**, because RL gradients
are weak supervision. What this objective provably builds: within-window
cross-object completion — a skill PPO never invokes (the policy always sees
all objects) — and it builds it while *degrading* state readability, forecast
readability, and sample efficiency below the random-init architecture prior.

Ranked changes (evidence for each above):

1. **Add a forecast masking mode (highest leverage).** Some fraction of
   samples, mask **all objects for states k..9** (temporal suffix / "tube")
   and predict forward from a past-only prefix. This makes P(future | past) —
   the quantity PPO consumes — the training signal instead of an incidental
   byproduct. Keep single-object completion masks as a minority mode for the
   routing skill. (Closest analog to V-JEPA's full-temporal-extent multiblock
   masks; the current always-visible anchor + full-context design is the root
   reason forecasting never emerges.)
2. **Action-condition the predictor** (feed self controller inputs for the
   masked span). Right now self-dynamics is unlearnable in principle (own
   future depends on unobserved inputs) and ball-after-touch is stochastic
   given state alone. With actions, the predictor becomes a latent world
   model — the standard route from JEPA pretraining to control (V-JEPA-2-AC).
   The shards already store both cars' actions; they are currently dropped at
   tokenization.
3. **Stop the target-erosion loop.** Candidates, cheapest first:
   (a) per-object loss floors diverging + identity-decode canary in the panel;
   (b) V-JEPA's predictor-variance regularizer `mean(relu(1 - std(z)))`;
   (c) revisit the wd ramp 0.04 -> 0.4 — that is a ViT-L/300-epoch recipe
   applied to a ~5M-param model, and the erosion window coincides with the
   ramp + post-warmup clipping becoming active; (d) a small auxiliary
   "token decodes its own features" reconstruction head on the *online*
   encoder (keeps the latent full-rank in information, not just spectrum).
4. **Rebalance masking pressure.** 45% ball masking concentrates
   make-targets-predictable pressure exactly on the object PPO cares most
   about. Under a suffix-mask regime this pressure redistributes naturally;
   if completion masks are kept, cap any single object's share.
5. **Downstream consumption**: take state-9 (or all) tokens, never the pooled
   mean; if an encoder must ship today, use **step15k-20k**, not the latest.

## 8. Caveats

- One train shard / one val shard; margins of ~±0.01–0.02 are noise. MLP
  probes are single-seed, 25 epochs, 1x512 — a capacity ladder (deeper MLPs)
  is a full-pass item.
- `collapse18450` attention-head count assumed 4 (not recoverable from
  shapes); all other config inferred and load was clean (no missing keys).
- Geometry here reads the EMA target encoder on val windows — not comparable
  to the training panel's online-encoder-on-train-batch numbers; within-suite
  comparisons only.
- `rand` shares the trivial-basin high-cos signature (cos 0.999) yet probes
  fine — cos/erank alone don't measure information, which is the whole point
  of the suite.
- Old `results_rjepa_*.json` future-probe labels are 2x off in horizon (their
  windows were collected with gap=2 but labeled as gap=1 seconds).

## Next (full pass, ~30-40 min)

- All 10 step checkpoints + collapse + interrupt for a dense trajectory of
  every table above (`--models all`).
- Fix `ball_impulse` threshold; add prefix-only ("causal") encoding probe —
  encode with only states 0..k visible for all objects and decode s9/future:
  measures forecast-through-encoder directly, no retraining needed.
- 2-3 probe shards per split for tighter CIs; MLP capacity ladder; a second
  seed for MLP probes.
