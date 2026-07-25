import math
import json

import torch
import torch.nn.functional as F
from pathlib import Path


def lr_schedule(step, lr, warmup_steps):
    """lr = (start, peak, final, total_steps).
    Linear warmup start -> peak over warmup_steps, cosine decay peak -> final
    over the remaining steps, then hold at final."""
    start, peak, final, total = lr
    if step < warmup_steps:
        return start + (peak - start) * (step + 1) / warmup_steps
    t = min(1.0, (step - warmup_steps) / max(1, total - warmup_steps))
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * t))


def wd_schedule(step, wd, total_steps):
    """wd = (start, final). Cosine ramp start -> final over total_steps, then hold.
    V-JEPA INCREASES weight decay across training (0.04 -> 0.4) as an anti-collapse
    regularizer: heavier decay late fights the degenerate low-norm constant solution.
    No warmup (unlike LR); ramps from step 0."""
    start, final = wd
    t = min(1.0, step / max(1, total_steps))
    return final + 0.5 * (start - final) * (1.0 + math.cos(math.pi * t))


def save_checkpoint(model, path, optim=None, epoch=None):
    """Save the model (and optionally optimizer/epoch) to `path`.

    Creates parent dirs as needed. Reload with:
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model": model.state_dict()}
    if optim is not None:
        ckpt["optim"] = optim.state_dict()
    if epoch is not None:
        ckpt["epoch"] = epoch
    torch.save(ckpt, path)
    return str(path)


def effective_rank(embeddings, eps=1e-7):
    # embeddings: [N, D] -- RankMe (Garrido et al. 2023): exp-entropy of the
    # raw singular value spectrum, no centering.
    s = torch.linalg.svdvals(embeddings)
    p = s / (s.sum() + eps)
    entropy = -(p * (p + eps).log()).sum()
    return entropy.exp().item()  # ranges from 1 (collapsed) to D (full rank)


def batch_collapse_metrics(embeddings):
    # embeddings: [N, D], N should be decent size (>=64)
    emb = F.normalize(embeddings, dim=-1)
    sim_matrix = emb @ emb.T  # [N, N]
    N = sim_matrix.shape[0]
    off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=emb.device)
    mean_offdiag_sim = sim_matrix[off_diag_mask].mean()
    return mean_offdiag_sim.item()


# ---------------------------------------------------------------------------
# CAUSAL collapse diagnostics.
#
# The panel above (erank/cos/vstd/dead) DETECTS collapse; the helpers below try
# to attribute its CAUSE. This model's only defenses against *directional*
# collapse (erank->1, cos->1) are the BYOL-style EMA + predictor dynamics --
# out_norm is RMSNorm and fixes scale only, so it is blind to the exact failure
# we keep hitting. These metrics fingerprint the fragile parts:
#
#   module_grad_norms / grad_norm  -- pre-clip grad norm per module. A SUDDEN
#       collapse usually opens with a grad spike in one module (often pred.mask
#       or enc.embed). Watch which module spikes first.
#   prenorm_rms (via attach_prenorm_probes) -- RMS of activations feeding each
#       out_norm. out_norm rescales to 1 and HIDES scale death from the loss; if
#       this dives toward 0 the real signal is gone and out_norm is amplifying
#       noise into a single direction. This is the blind spot -- watch it.
#   predictor_context_sensitivity -- does the predictor's output actually depend
#       on the encoder context, or has it become a constant map? ->0 means the
#       predictor collapsed to the mean (classic BYOL precursor), which then
#       drags the target down.
#   ema_param_divergence -- ||theta_online - theta_target|| / ||theta_online||.
#       If this collapses to ~0 the EMA asymmetry (the ONLY thing breaking the
#       trivial-solution symmetry) is gone -> nothing prevents co-collapse.
#   covariance_metrics -- redundancy (off-diag/diag) and top-eigenvalue share.
#       Smoother, earlier leading indicators than the erank scalar; redundancy
#       rising or top_share -> 1 is dimensional collapse forming.
# ---------------------------------------------------------------------------


@torch.no_grad()
def grad_norm(params):
    """L2 norm over a param iterable's .grad (pre-clip). None grads skipped.
    Accumulates on-device and syncs ONCE (per-param .item() would force a GPU
    sync per parameter every step -- expensive when called each step)."""
    sq = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
    if not sq:
        return 0.0
    return torch.sqrt(torch.stack(sq).sum()).item()


@torch.no_grad()
def module_grad_norms(model):
    """Per-module pre-clip grad norms. mask_token and the object-embedding MLP
    are broken out separately because they are the usual first movers in a
    sudden collapse."""
    enc = model.encoder
    groups = {
        "enc.embed": list(enc.embedding.parameters()),
        "enc.attn": [p for b in enc.attention for p in b.parameters()],
        "enc.ffn": [p for b in enc.ffn for p in b.parameters()],
        "pred.mask": [model.predictor.mask_token],
        "pred.rest": [p for n, p in model.predictor.named_parameters() if "mask_token" not in n],
    }
    return {k: grad_norm(v) for k, v in groups.items()}


def attach_prenorm_probes(model):
    """Register forward-pre-hooks on the three out_norm modules so we can read
    the activation RMS *before* normalization. Returns (store, handles); read
    store['enc'|'pred'|'tgt'] after any forward pass. Call handle.remove() to
    detach. This is the only window behind out_norm's scale masking."""
    store = {}

    def mk(name):
        def hook(_mod, inp):
            store[name] = inp[0].detach().float().pow(2).mean().sqrt().item()
        return hook

    handles = [
        model.encoder.out_norm.register_forward_pre_hook(mk("enc")),
        model.predictor.out_norm.register_forward_pre_hook(mk("pred")),
        model.target_encoder.out_norm.register_forward_pre_hook(mk("tgt")),
    ]
    return store, handles


@torch.no_grad()
def predictor_context_sensitivity(model, window):
    """Does the predictor use the encoder context, or is it a constant map?

    Runs the predictor on (a) the real context and (b) the same contexts shuffled
    across the batch, keeping each query's masked_idx fixed. A healthy predictor
    changes a lot when fed the WRONG sample's context; a collapsed one (predicting
    the mean regardless of input) barely moves.

    Returns (rel_change, cos):
      rel_change  mean ||z_real - z_shuffled|| / ||z_real||. ->0 == constant map.
      cos         mean cosine(z_real, z_shuffled). ->1 == constant map.
    """
    from jepa import build_mask
    state = model.encoder.pos(model.encoder.embedding.build(window))
    masked_idx, non_masked = build_mask(state, model.mask_probs, model.objects)
    ctx = model.encoder(non_masked)
    z_real = model.predictor(ctx, masked_idx).float()
    perm = torch.randperm(ctx.size(0), device=ctx.device)
    z_shuf = model.predictor(ctx[perm], masked_idx).float()
    rel = ((z_real - z_shuf).norm(dim=-1) / (z_real.norm(dim=-1) + 1e-8)).mean().item()
    cos = F.cosine_similarity(z_real, z_shuf, dim=-1).mean().item()
    return rel, cos


@torch.no_grad()
def ema_param_divergence(model):
    """Relative distance between the online encoder and its EMA target,
    ||theta_on - theta_tgt|| / ||theta_on||. This asymmetry is the ONLY symmetry
    breaker preventing the trivial solution; if it collapses toward 0 the target
    is just tracking the online net and there is nothing to stop co-collapse."""
    num, den = 0.0, 0.0
    for on, tg in zip(model.encoder.parameters(), model.target_encoder.parameters()):
        num += (on.detach() - tg.detach()).float().pow(2).sum().item()
        den += on.detach().float().pow(2).sum().item()
    return (num ** 0.5) / (den ** 0.5 + 1e-8)


@torch.no_grad()
def covariance_metrics(rep):
    """Two leading indicators on rep [N, D], one per collapse mode. Read together
    they say WHICH kind of collapse is forming, earlier/smoother than the erank
    scalar:

      top_share   lambda_1 / sum(lambda) of the UNCENTERED second moment
                  (rep.T @ rep). ->1 == all samples collapsing onto one direction
                  / one point (the mean dominates). THIS is the mode we keep
                  hitting (cos->1, erank->1); watch it climb.
      redundancy  mean|off-diag| / mean(diag) of the CENTERED covariance, i.e.
                  correlation among dims of the *variation about the mean*. Rising
                  == the spread that survives is piling into a subspace. High
                  redundancy while top_share stays low is the OTHER mode
                  (dimensional collapse of the residual), a different cause.
    """
    x = rep.float()
    n = x.size(0)

    # uncentered: catches point/directional collapse (mean domination)
    m2 = (x.T @ x) / max(1, n)
    ev = torch.linalg.eigvalsh(m2).clamp_min(0)
    top_share = (ev[-1] / (ev.sum() + 1e-8)).item()

    # centered: catches low-rank structure in the variation about the mean
    xc = x - x.mean(0, keepdim=True)
    cov = (xc.T @ xc) / max(1, n - 1)
    diag = torch.diagonal(cov)
    off = cov - torch.diag(diag)
    redundancy = off.abs().mean().item() / (diag.mean().item() + 1e-8)

    return redundancy, top_share


# ---------------------------------------------------------------------------
# COLLAPSE ATTRIBUTION
#
# The metrics above are ingredients. This is the part that answers the actual
# question -- "given a collapse, what caused it?" -- instead of leaving you to
# eyeball stdout. The idea: every signal has a healthy baseline (the early run)
# and a known "bad direction". When collapse trips, we find, per signal, the
# FIRST step it departed from baseline in its bad direction. The signal that
# departed EARLIEST is the root-cause candidate; everything that moved after it
# is downstream. `explain_collapse` prints that as a ranked timeline with the
# lead time over the cos/erank collapse itself.
# ---------------------------------------------------------------------------

# (bad_direction, one-line cause if this is the leading signal). Grad signals use
# persist=1 (see _PERSIST) because a destabilizing spike can be a SINGLE step --
# requiring 2 consecutive samples would hide exactly the event we're hunting.
_SIGNAL_SPEC = {
    "gnorm":     ("up",   "TOTAL GRADIENT SPIKE -> optimizer instability lit the fuse"),
    "gm_mask":   ("up",   "predictor mask-token gradient exploded (localized instability; the mask token is a common first mover)"),
    "gm_embed":  ("up",   "object-embedding gradient exploded (localized instability at the input projection)"),
    "gm_attn":   ("up",   "encoder attention gradient exploded"),
    "gm_ffn":    ("up",   "encoder FFN gradient exploded"),
    "gm_predrest": ("up", "predictor (non-mask) gradient exploded"),
    "pn_enc":    ("down", "encoder activation scale collapsed behind out_norm (signal died; out_norm then amplifies noise into one direction)"),
    "pn_pred":   ("down", "predictor activation scale collapsed behind out_norm"),
    "pn_tgt":    ("down", "TARGET activation scale collapsed behind out_norm (the labels themselves went degenerate)"),
    "pred_rel":  ("down", "predictor became a CONSTANT MAP (output stopped depending on context) -> BYOL-style predictor collapse dragged the target down"),
    "pred_cos":  ("up",   "predictor output went invariant to its context"),
    "ema_div":   ("down", "EMA online<->target asymmetry vanished -> the only symmetry-breaker was gone, nothing prevented co-collapse"),
    "top_share": ("up",   "point/directional collapse: variance concentrated onto one axis"),
    "redun":     ("up",   "dimensional collapse: surviving variance became redundant across dims"),
    "erank":     ("down", "effective rank fell (collapse outcome)"),
    "cos":       ("up",   "pairwise cosine rose toward 1 (collapse outcome)"),
}

# spike-type signals break on a single sample; slow signals need 2 to reject noise
_PERSIST = {"gnorm": 1, "gm_mask": 1, "gm_embed": 1, "gm_attn": 1, "gm_ffn": 1, "gm_predrest": 1}


def _first_break(steps, values, direction, ref_frac=0.30, k=6.0, persist=2, eps=1e-8):
    """First step where `values` departs from its early-run baseline in the bad
    `direction` by > k robust-sigmas and stays there for `persist` samples.

    Baseline = median +/- k*1.4826*MAD over the first `ref_frac` of the series
    (the healthy part, since collapse is late & sudden). Returns (step, ref_med,
    val_at_break) or None if it never breaks."""
    n = len(values)
    if n < 6:
        return None
    r = max(3, int(n * ref_frac))
    ref = sorted(values[:r])
    med = ref[len(ref) // 2]
    mad = sorted(abs(v - med) for v in values[:r])[r // 2]
    sigma = 1.4826 * mad + eps + 0.01 * abs(med)   # floor so a flat baseline still needs real movement
    hi, lo = med + k * sigma, med - k * sigma
    hit = 0
    for i in range(r, n):
        broke = values[i] > hi if direction == "up" else values[i] < lo
        hit = hit + 1 if broke else 0
        if hit >= persist:
            j = i - persist + 1
            return steps[j], med, values[j]
    return None


def attribute_collapse(hist):
    """hist: list of dicts, one per logged step, each with 'step' + whatever
    signals were recorded (missing keys are skipped). Returns a list of
    (break_step, signal, ref_median, break_value, cause) sorted earliest-first."""
    if not hist:
        return []
    steps = [h["step"] for h in hist]
    out = []
    for sig, (direction, cause) in _SIGNAL_SPEC.items():
        series = [(h["step"], h[sig]) for h in hist if sig in h and h[sig] is not None]
        if len(series) < 6:
            continue
        s_steps = [s for s, _ in series]
        s_vals = [v for _, v in series]
        br = _first_break(s_steps, s_vals, direction, persist=_PERSIST.get(sig, 2))
        if br is not None:
            bstep, med, val = br
            out.append((bstep, sig, med, val, cause))
    out.sort(key=lambda t: t[0])
    return out


def explain_collapse(hist, log_path=None):
    """Print (and optionally JSONL-dump) a ranked causal timeline for a collapse.
    Call this when an alarm trips. The earliest-departing signal is flagged as
    the root-cause candidate; the collapse outcome signals (cos/erank) are shown
    with their lead time so you can see how far ahead the true trigger moved."""
    ranked = attribute_collapse(hist)
    if log_path is not None:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for h in hist:
                f.write(json.dumps(h) + "\n")

    if not ranked:
        print("  == COLLAPSE ATTRIBUTION: no signal broke its baseline "
              "(too little history, or collapse faster than the log cadence) ==")
        return ranked

    # lead time of the root cause over the first *outcome* signal (cos/erank)
    outcome_steps = [bs for bs, sig, *_ in ranked if sig in ("cos", "erank")]
    first_outcome = min(outcome_steps) if outcome_steps else None
    root_step, root_sig, _, _, root_cause = ranked[0]

    print("  == COLLAPSE ATTRIBUTION ==================================")
    lead = f" ({first_outcome - root_step} steps before the cos/erank collapse)" if first_outcome else ""
    print(f"  ROOT CAUSE  [{root_sig} @ step {root_step}]{lead}:")
    print(f"    {root_cause}")
    print("  timeline (earliest departure first):")
    for bstep, sig, med, val, cause in ranked:
        tag = "<-- ROOT" if (bstep, sig) == (root_step, root_sig) else \
              ("outcome" if sig in ("cos", "erank") else "")
        print(f"    step {bstep:>8}  {sig:<9} baseline~{med:.3f} -> {val:.3f}  {tag}")
    print("  =========================================================")
    return ranked


@torch.no_grad()
def collapse_metrics(rep):
    """Collapse diagnostics on per-sample pooled latents rep: [N, D] (one batch).

    Collapse = the encoder mapping distinct game states to the same/low-rank
    output. These four cover the ways it shows up; read them together, not alone:

      erank   RankMe effective rank of the batch. Dimensional collapse -> 1;
              healthy = a large fraction of D.
      cos     mean pairwise cosine similarity across samples. Complete collapse
              (all samples identical) -> 1; healthy is near 0. THE decisive one:
              if this climbs toward 1 the reps are genuinely converging.
      vstd    mean per-dim std of L2-normalized reps (VICReg variance term).
              Collapse -> 0; scale-invariant, so unlike raw latent_std it can't
              be fooled by the overall magnitude drifting. ~1/sqrt(D) is healthy.
      dead    fraction of dims whose across-batch std is < 1% of the mean dim
              std — i.e. axes carrying no information. Partial collapse -> rises.
    """
    rep = rep.float()
    n = rep.size(0)
    erank = effective_rank(rep)
    p = F.normalize(rep, dim=-1)
    sim = p @ p.T
    cos = sim[~torch.eye(n, dtype=torch.bool, device=rep.device)].mean().item()
    vstd = p.std(0).mean().item()
    d = rep.std(0)
    dead = (d < 0.01 * d.mean()).float().mean().item()
    return erank, cos, vstd, dead
