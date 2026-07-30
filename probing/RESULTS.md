# Frozen-encoder probe results — `rjepa_interrupt.pt` (2026-07-23)

`probes.py` freezes the EMA target encoder and fits a ladder of probes on
cached latents. All targets are in the agent's **state-9 local frame** (the
model's own `build_obs` convention) — world-frame readouts are intentionally
not probed, since the feature frame is agent-centric and world pose is
irrelevant by design. Probe-train = shard_00310 (35,362 windows), probe-val =
shard_00312 (8,000 windows, different replays).

Controls per probe — the encoder *learned* something only where jepa beats both:

- **rand** — same probe on a randomly-initialized frozen encoder (what the
  architecture + a random projection of the inputs gives for free);
- **raw** — linear probe on the flattened 10x110 input window (task triviality).

Scores: val R² (regression) / ROC-AUC (binary).

| tier | probe | jepa lin | jepa mlp | rand lin | rand mlp | raw lin |
|---|---|---|---|---|---|---|
| L1 | ball token → ball rel pos (now)   | 0.998 | — | 1.000 | — | 0.318 |
| L1 | self token → own boost (now)      | 0.997 | — | 0.999 | — | 1.000 |
| L2 | car tokens → ball rel pos (now)   | **0.932** | 0.898 | 0.675 | 0.734 | 0.318 |
| L3 | ball MASKED → ball rel pos (now)  | **0.659** | 0.652 | 0.367 | 0.428 | 0.286 |
| L3 | ball MASKED → ball rel vel (now)  | **0.721** | 0.658 | 0.425 | 0.445 | 0.299 |
| L3 | opp MASKED → opp rel pos (now)    | **0.610** | 0.579 | 0.465 | 0.505 | 0.249 |
| L3 | opp MASKED → opp rel vel (now)    | **0.488** | 0.456 | 0.319 | 0.363 | 0.224 |
| L4 | s9 tokens → ball rel pos +0.5s    | 0.968 | 0.906 | 0.967 | 0.965 | 0.283 |
| L4 | s9 tokens → ball rel pos +1s      | 0.873 | 0.828 | 0.870 | 0.876 | 0.245 |
| L4 | s9 tokens → ball local vel +1s    | 0.379 | 0.284 | 0.369 | 0.371 | 0.121 |
| L5 | s9 tokens → opp rel pos +0.5s     | 0.987 | 0.961 | 0.987 | 0.984 | 0.234 |
| L5 | s9 tokens → opp rel pos +1s       | 0.934 | 0.887 | 0.932 | 0.934 | 0.209 |
| L5 | s9 tokens → own displacement +1s  | 0.733 | 0.627 | 0.722 | 0.729 | 0.342 |
| L5 | all tokens → ball grounded in 1s (AUC, base 0.46)  | 0.757 | 0.794 | 0.740 | 0.779 | **0.908** |
| L5 | all tokens → opp burns boost in 1s (AUC, base 0.44) | 0.706 | 0.725 | 0.715 | 0.744 | **0.782** |

## Evaluation

**Healthy but immature: the JEPA objective is demonstrably doing its job on
the encoder, and nothing is collapsed — but the latents don't yet contain
dynamics or intent beyond what raw kinematics gives linearly.**

1. **No collapse, no information loss (L1).** Object tokens remain perfectly
   linearly decodable to their own inputs after 6 blocks of mixing.

2. **Cross-object routing is strongly learned (L2, L3) — the core JEPA
   claim confirmed.** Car tokens linearly carry ball position (0.93 vs 0.73
   best random control). Under the actual training condition (object hidden
   in states 1–9), visible-token pools recover the hidden object's position
   AND velocity with large, consistent margins over both controls: ball
   +0.23/+0.28 R², opp +0.11/+0.13. Velocity margins beating position margins
   means it's encoding hidden *state*, not just interpolating a location.
   Opp margins < ball margins, consistent with the masking analysis (opp is
   stochastic intent; ball is physics).

3. **Future prediction sits exactly at the kinematic ceiling (L4, L5).**
   On every local-frame future probe, jepa ties the random encoder to within
   noise (e.g. ball +1s: 0.873 vs 0.876; opp +1s: 0.934 vs 0.934). Both just
   preserve current pos+vel, which a linear probe extrapolates. The decisive
   row is **ball local vel +1s** — dominated by bounces/touches — where both
   encoders are stuck at ~0.37. Learned ball dynamics in the encoder latents
   would push jepa above rand here; it hasn't happened yet.

4. **No semantic/intent edge yet (L5 binaries).** Ball-grounded-soon and
   opp-burns-boost sit at or below the random control.

Interpretation: after a partial first epoch of a 210k-step schedule, the
encoder has learned *communication* (visible objects describe hidden ones —
which is literally the training loss, so this is the objective working), but
not yet *prediction* (that skill currently lives only in the predictor). This
matches early-training expectations; the JEPA bet is that 2→3 transfers with
more steps.

**Scorecard to track across checkpoints** (re-run on `rjepa_step25k/50k/...`;
point `CKPT` at the snapshot and delete/rename the feature cache):
- L3 margins over rand → should keep widening (objective progress);
- L4 "ball local vel +1s" jepa vs rand → the first sign of real learned
  dynamics in the latents;
- L5 opp rows + boost binary → the anticipation/intent prior that is meant to
  transfer to PPO.

## Caveats

- Mean-pooling is lossy: the concat state-9 readout beats the 50-token mean
  on every linear probe. Downstream consumers should take tokens, not means.
- Probes read the EMA target encoder.
- MLP probes occasionally score below linear (small val set + light tuning);
  treat linear as the primary number.
