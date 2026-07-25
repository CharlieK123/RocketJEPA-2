import torch
import torch.nn.functional as F
from collections import deque
from training.functions import (
    effective_rank, collapse_metrics, lr_schedule, wd_schedule, save_checkpoint,
    grad_norm, module_grad_norms, attach_prenorm_probes,
    predictor_context_sensitivity, ema_param_divergence, covariance_metrics,
    explain_collapse,
)

EPOCHS = 1

# --- early-warning thresholds for auto-snapshotting around a sudden collapse ---
# Collapse here is abrupt and at a non-repeatable step, so we can't pre-place a
# snapshot. Instead we watch cheap per-step signals and dump a tagged checkpoint
# THE MOMENT something trips, giving a clean pre/post-collapse pair to autopsy.
#
# CRITICAL: healthy cos/erank are MODEL-SPECIFIC (this one plateaus at cos~0.35,
# erank~85), so absolute thresholds mis-fire. Collapse is therefore declared
# RELATIVE to the best healthy values THIS run has reached (tracked after arming).
GRAD_SPIKE_MULT = 5.0     # grad norm > this * trailing median -> spike
PRENORM_FLOOR = 0.30      # pre-out_norm activation RMS below this -> scale dying (healthy ~1.0+)
ALARM_COOLDOWN = 500      # min steps between auto-snapshots so we don't spam disk
ERANK_WARN_FRAC = 0.60    # erank fell below this fraction of its healthy PEAK -> warn+snapshot
ERANK_COLLAPSE_FRAC = 0.35  # ... below this -> confirmed collapse
COS_WARN_RISE = 0.20      # cos climbed this far back above its healthy FLOOR -> warn+snapshot
COS_COLLAPSE_RISE = 0.40  # ... this far above -> confirmed collapse

# --- ARMING ---------------------------------------------------------------
# An untrained model looks collapsed: shared positional encoding gives a large
# common component (high cos, low erank, tiny pre-norm scale) before the encoder
# learns to spread states apart. Firing collapse alarms on that is a false
# positive AND poisons the attribution baseline. So alarms stay DISARMED until
# the model first reaches a healthy PLATEAU; only a departure from that counts.
# erank (not cos) is the arming signal: it cleanly separates collapsed (~1.5)
# from healthy (85+), whereas healthy cos is model-specific and may never cross
# a fixed low threshold.
ARM_ERANK = 20.0          # min erank to be considered out of the trivial-solution basin
ARM_PLATEAU_TOL = 0.15    # relative spread of erank over recent evals to call it "stable"
ARM_DEADLINE = 15_000     # not armed by here -> warn: likely STUCK in the basin from init
SPIKE_COOLDOWN = 250      # rate-limit grad-spike logging (spikes come in clusters)


def train(model, loader, optim, lr, warmup_steps, wd, device='cuda'):
    model.train()
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    step = 0

    # pre-out_norm activation RMS probes; `prenorm` is refreshed every forward pass
    prenorm, _handles = attach_prenorm_probes(model)
    grad_hist = deque(maxlen=100)   # trailing pre-clip TOTAL grad norms for spike detection
    mg_hist = {k: deque(maxlen=100) for k in           # per-module trailing norms, for
               ("enc.embed", "enc.attn", "enc.ffn", "pred.mask", "pred.rest")}  # localized spikes
    # rolling per-step metric history for post-hoc CAUSAL attribution. Bounded so
    # it can't grow across a 210k run; 20k steps is ample lead to (a) establish a
    # healthy baseline and (b) capture a sudden collapse's onset at step resolution.
    hist = deque(maxlen=20_000)
    last_alarm = -ALARM_COOLDOWN
    reported = False   # definitive attribution fires exactly once, cooldown-independent
    armed = False      # collapse alarms disarmed until the model first looks healthy
    arm_step = None    # step at which health was first established (baseline origin)
    arm_warned = False # one-shot warning if the model never escapes the basin
    erank_hist = deque(maxlen=8)   # recent eval eranks, for plateau detection
    best_erank, best_cos, best_top = 0.0, 1.0, 1.0   # best healthy values seen (post-arming)
    last_spike = -SPIKE_COOLDOWN   # rate-limit transient grad-spike logging
    n_spikes = 0                   # running count of grad-spike events (frequency matters)

    def armed_hist():
        # attribution only sees post-arming (healthy-baseline) history
        return [r for r in hist if arm_step is not None and r["step"] >= arm_step]

    def snapshot(reason, attribute):
        # save a checkpoint for autopsy. attribute=True ALSO runs the collapse
        # narrative -- reserved for REAL health degradation, NOT transient grad
        # spikes (which recover and are not collapses; running the narrative on them
        # produces false "root cause before the collapse" reports).
        nonlocal last_alarm
        if not armed:            # never fire on an untrained (natively degenerate) model
            return
        if step - last_alarm < ALARM_COOLDOWN:
            return
        last_alarm = step
        path = save_checkpoint(model, f"checkpoints/rjepa_ALARM_step{step}_{reason}.pt",
                               optim=optim, epoch=0)
        print(f"  !! ALARM [{reason}] step={step} -> snapshot {path}")
        if attribute:
            # rank every signal by when it first left its baseline; earliest = root.
            explain_collapse(armed_hist(), log_path=f"checkpoints/rjepa_ALARM_step{step}_{reason}.jsonl")

    for epoch in range(EPOCHS):

        for window in loader:
            window = window.to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                z_hat, z, masked_idx = model(window)  # run the forward pass

                loss = F.smooth_l1_loss(z_hat, z)

            # warmup+cosine LR and cosine-up weight-decay, set before the optimizer step
            for g in optim.param_groups:
                g["lr"] = lr_schedule(step, lr, warmup_steps)
                g["weight_decay"] = wd_schedule(step, wd, lr[3])

            optim.zero_grad(set_to_none=True)
            loss.backward()

            # pre-clip grad norms EVERY step: the trigger of a sudden collapse is
            # almost always a grad event, and we must catch it before clipping
            # (which happens only after warmup) masks the magnitude. Per-module too,
            # so a spike LOCALIZED to one module (e.g. mask_token) is attributable
            # even when it barely moves the global total.
            gnorm = grad_norm(model.parameters())
            mg = module_grad_norms(model)   # {enc.embed, enc.attn, enc.ffn, pred.mask, pred.rest}
            if armed and len(grad_hist) >= 20:
                med = sorted(grad_hist)[len(grad_hist) // 2]
                # spike if the global total OR any single module jumps 5x its recent median
                mg_med = {k: sorted(v)[len(v) // 2] for k, v in mg_hist.items()} if len(grad_hist) >= 20 else {}
                mod_spike = next((k for k, v in mg.items() if v > GRAD_SPIKE_MULT * (mg_med.get(k, 0) + 1e-8)), None)
                if (gnorm > GRAD_SPIKE_MULT * (med + 1e-8) or mod_spike) and step - last_spike >= SPIKE_COOLDOWN:
                    last_spike = step
                    n_spikes += 1
                    worst = max(mg, key=mg.get)
                    # a grad spike is an EVENT, not a collapse: post-warmup clipping
                    # usually absorbs it and the model recovers. We log it (spikes are
                    # the likely eventual-collapse trigger) but do NOT run the collapse
                    # narrative unless health has actually degraded (checked in eval).
                    print(f"  ~ grad spike #{n_spikes} step={step} gnorm={gnorm:.2f} "
                          f"(median={med:.2f}) worst={worst}={mg[worst]:.2f}"
                          + (f" module_spike={mod_spike}" if mod_spike else ""))
            grad_hist.append(gnorm)
            for k, v in mg.items():
                mg_hist[k].append(v)

            # record the cheap per-step signals into the attribution buffer. The
            # eval block below enriches THIS SAME row with the expensive signals.
            row = {"step": step, "gnorm": gnorm,
                   "gm_embed": mg["enc.embed"], "gm_attn": mg["enc.attn"], "gm_ffn": mg["enc.ffn"],
                   "gm_mask": mg["pred.mask"], "gm_predrest": mg["pred.rest"],
                   "pn_enc": prenorm.get("enc"), "pn_pred": prenorm.get("pred"),
                   "pn_tgt": prenorm.get("tgt")}
            hist.append(row)

            # scale dying behind out_norm is invisible to the loss — guard it per step.
            # This IS a real collapse mode, so attribute it.
            if prenorm.get("enc", 1.0) < PRENORM_FLOOR or prenorm.get("tgt", 1.0) < PRENORM_FLOOR:
                snapshot("prenorm", attribute=True)

            if step >= warmup_steps:  # V-JEPA: clip only after warmup
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optim.step()
            model.update_target_params(step)

            # periodic checkpointing: rolling latest (crash recovery) +
            # tagged snapshots every 25k for downstream probe comparisons
            if step and step % 5_000 == 0:
                save_checkpoint(model, "checkpoints/rjepa_latest.pt", optim=optim, epoch=epoch)
            if step and step % 25_000 == 0:
                save_checkpoint(model, f"checkpoints/rjepa_step{step // 1000}k.pt")

            # eval
            if step % 50 == 0:
                with torch.no_grad():
                    tok = model.encoder.pos(model.encoder.embedding.build(window))
                    rep = model.encoder(tok).mean(1)               # [B, D] per-sample pooled
                    # collapse panel on the online encoder's per-sample reps
                    erank, cos, vstd, dead = collapse_metrics(rep)
                    # target-side rank: the loss chases these EMA/normalized latents,
                    # so if THEY collapse the whole objective is degenerate. Pool the
                    # masked target positions per sample and take RankMe.
                    tgt_erank = effective_rank(z.float().mean(1))
                    # per-object loss: which object each sample masked, bucketed smooth-L1.
                    # expected: env ~0 fast (canary), self low (frame leak),
                    # boost floors slowly, ball/opp keep a persistent floor. A collapse
                    # tell is these CONVERGING and diving toward the ~0.001 trivial floor.
                    obj = masked_idx[:, 0] % model.objects
                    per = F.smooth_l1_loss(z_hat.float(), z.float(), reduction="none").mean(dim=(1, 2))
                    names = ("self", "opp", "ball", "env", "boost")
                    per_obj = " ".join(
                        f"{n}={per[obj == i].mean().item():.4f}" if (obj == i).any() else f"{n}=n/a"
                        for i, n in enumerate(names))

                    # --- CAUSAL panel: which mechanism is failing ---
                    redundancy, top_share = covariance_metrics(rep)   # dimensional-collapse leading indicators
                    pred_rel, pred_cos = predictor_context_sensitivity(model, window)  # constant-map test
                    ema_div = ema_param_divergence(model)             # EMA asymmetry health
                    # (mg = per-module grad norms already computed this step)

                    # enrich this step's attribution row with the expensive signals
                    row.update(dict(cos=cos, erank=erank, top_share=top_share,
                                    redun=redundancy, pred_rel=pred_rel, pred_cos=pred_cos,
                                    ema_div=ema_div, loss=loss.item()))

                    print(f"itr: {step}, loss={loss.item():.5f} erank={erank:.1f} tgt_erank={tgt_erank:.1f} "
                          f"cos={cos:.3f} vstd={vstd:.4f} dead={dead:.2f} | {per_obj}")
                    print(f"      CAUSE| prenorm(enc/pred/tgt)={prenorm.get('enc', 0):.2f}/"
                          f"{prenorm.get('pred', 0):.2f}/{prenorm.get('tgt', 0):.2f} "
                          f"pred_sens(rel/cos)={pred_rel:.3f}/{pred_cos:.3f} "
                          f"ema_div={ema_div:.4f} redun={redundancy:.3f} top_share={top_share:.3f} "
                          f"gnorm={gnorm:.2f} spikes={n_spikes} | grad "
                          + " ".join(f"{k}={v:.2f}" for k, v in mg.items()))

                    erank_hist.append(erank)
                    if not armed:
                        # arm once erank is high AND has plateaued (stopped climbing):
                        # a stable healthy regime, not a point mid-escape.
                        plateaued = (len(erank_hist) == erank_hist.maxlen and
                                     (max(erank_hist) - min(erank_hist)) /
                                     (sum(erank_hist) / len(erank_hist)) < ARM_PLATEAU_TOL)
                        if erank > ARM_ERANK and plateaued:
                            armed = True
                            arm_step = step
                            best_erank, best_cos, best_top = erank, abs(cos), top_share
                            print(f"   armed collapse detection @ step {step} — healthy baseline "
                                  f"(erank={erank:.1f} cos={cos:.3f} top_share={top_share:.3f})")
                        elif step >= ARM_DEADLINE and not arm_warned:
                            arm_warned = True
                            print(f"  !! NEVER ARMED by step {step}: erank still {erank:.1f} "
                                  f"(< {ARM_ERANK}). The model likely never escaped the trivial-"
                                  f"solution basin it starts in — a BORN-COLLAPSED run, not a "
                                  f"mid-training collapse. Suspect EMA momentum too high early, "
                                  f"predictor too strong, or LR warmup. (No baseline to attribute "
                                  f"against, so the causal engine stays off.)")
                    else:
                        # track the best healthy values; collapse = departure FROM them
                        best_erank = max(best_erank, erank)
                        best_cos = min(best_cos, abs(cos))
                        best_top = min(best_top, top_share)
                        warn = (erank < ERANK_WARN_FRAC * best_erank or
                                abs(cos) > best_cos + COS_WARN_RISE or
                                top_share > best_top + COS_WARN_RISE)
                        collapse = (erank < ERANK_COLLAPSE_FRAC * best_erank or
                                    abs(cos) > best_cos + COS_COLLAPSE_RISE or
                                    top_share > best_top + COS_COLLAPSE_RISE)
                        if warn:
                            snapshot("collapse", attribute=True)
                        # DEFINITIVE attribution, once, NOT gated by the snapshot cooldown:
                        # an early warn-snapshot precedes the full collapse, so its report
                        # has no outcome to measure lead time against. This fires once the
                        # collapse has materialized, giving the complete root-cause timeline.
                        if not reported and collapse:
                            print("  == CONFIRMED COLLAPSE: running definitive attribution ==")
                            explain_collapse(armed_hist(),
                                             log_path=f"checkpoints/rjepa_CONFIRMED_step{step}.jsonl")
                            reported = True

            step += 1
