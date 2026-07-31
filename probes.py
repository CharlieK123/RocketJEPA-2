"""
probes.py — offline arbiter for the "is it collapsing or just growing a mean?" question.

For each checkpoint it computes, on IDENTICAL data:
  geometry   raw_cos (the number in your panel), centered_cos (after removing the
             batch-mean), mean_share (fraction of rep energy living in the shared
             mean vector), erank of the CENTERED reps
  probe      ridge regression: pooled rep of the ctx-state history  ->  ball obs
             vector at the final state (build_obs ball features, model-independent),
             reported as test R^2, next to a raw-feature persistence baseline
             (ridge from the last two visible ball obs vectors to the same target).

Interpretation:
  raw_cos up + centered_cos flat/low + probe R^2 up   -> mean growth, benign. Train on.
  centered_cos up in tandem + probe R^2 flat/down     -> genuine ratchet. Intervene.
  probe R^2 below the persistence baseline            -> rep not earning its keep yet.

Run on the vast box from the repo root (defaults avoid touching the training GPU):
  python probes.py --shards /workspace/data/shards_75k \
      --ckpts ../checkpoints/<RUN>/rjepa_ALARM_step8000_collapse.pt \
              ../checkpoints/<RUN>/rjepa_ALARM_step14000_collapse.pt \
              ../checkpoints/<RUN>/rjepa_ALARM_step20000_collapse.pt
Sanity-check the plumbing anywhere with:  python probes.py --selftest
"""
import argparse, math, json
from pathlib import Path

import torch
import torch.nn.functional as F

from jepa import JEPA
from models.entity_encoding import build_obs

OBJS = 5


# ---------------------------------------------------------------- geometry ---
@torch.no_grad()
def geometry(rep):
    """rep [N, D] float32 -> dict of raw/centered stats."""
    n = min(rep.size(0), 512)
    r = rep[:n]
    mu = rep.mean(0, keepdim=True)

    def mean_pair_cos(x):
        x = F.normalize(x, dim=-1)
        g = x @ x.T
        return ((g.sum() - g.diag().sum()) / (x.size(0) * (x.size(0) - 1))).item()

    cen = rep - mu
    s = torch.linalg.svdvals(cen)
    p = (s * s) / (s * s).sum().clamp_min(1e-12)
    erank = math.exp(-(p * (p + 1e-12).log()).sum().item())
    return dict(
        raw_cos=mean_pair_cos(r),
        cen_cos=mean_pair_cos(cen[:n]),
        mean_share=(mu.pow(2).sum() / rep.pow(2).sum(1).mean().clamp_min(1e-12)).item(),
        erank_cen=erank,
        top_share_cen=p[0].item(),
    )


# ------------------------------------------------------------------- probe ---
def ridge_r2(Xtr, Ytr, Xte, Yte, seed=0):
    """Standardized ridge with a small internal lambda sweep. Returns test R^2."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(Xtr.size(0), generator=g)
    k = int(0.8 * len(idx)); fit, val = idx[:k], idx[k:]

    mx, sx = Xtr[fit].mean(0), Xtr[fit].std(0).clamp_min(1e-6)
    my = Ytr[fit].mean(0)
    Z  = lambda X: (X - mx) / sx

    def solve(lam):
        Xf = Z(Xtr[fit]); Yf = Ytr[fit] - my
        A = Xf.T @ Xf + lam * Xf.size(0) * torch.eye(Xf.size(1))
        return torch.linalg.solve(A, Xf.T @ Yf)

    def r2(W, X, Y):
        P = Z(X) @ W + my
        ss = (Y - P).pow(2).sum(); tot = (Y - Y.mean(0)).pow(2).sum().clamp_min(1e-12)
        return (1 - ss / tot).item()

    best = max(((r2(solve(l), Xtr[val], Ytr[val]), l) for l in (1e-4, 1e-3, 1e-2, 1e-1)))
    return r2(solve(best[1]), Xte, Yte)


# -------------------------------------------------------------------- reps ---
@torch.no_grad()
def encode(model, win, ctx, device, chunk=256):
    """Returns (rep_full [B,D] over all 75 tokens — matches your panel,
                rep_hist [B,D] over the ctx-state prefix — the deployment rep)."""
    full, hist = [], []
    for i in range(0, win.size(0), chunk):
        w = win[i:i + chunk].to(device)
        tok = model.encoder.pos(model.encoder.embedding.build(w))
        full.append(model.encoder(tok).mean(1).float().cpu())
        hist.append(model.encoder(tok[:, : ctx * OBJS]).mean(1).float().cpu())
    return torch.cat(full), torch.cat(hist)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = (ckpt.get("meta") or {}).get("model")
    if cfg is None:
        from main import MODEL_CFG as cfg  # __main__-guarded, safe to import
    model = JEPA(**{**cfg, "mask_probs": torch.tensor(cfg["mask_probs"])})
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt.get("meta", {}).get("step", "?")


# -------------------------------------------------------------------- main ---
def run(ckpts, win_fit, win_te, ctx, device):
    # model-independent probe targets: ball obs vector at the final state,
    # baseline inputs: ball obs at the last two visible states
    yb_tr = build_obs(win_fit)[2]; yb_te = build_obs(win_te)[2]
    Ytr, Yte = yb_tr[:, -1], yb_te[:, -1]
    Btr = torch.cat([yb_tr[:, ctx - 1], yb_tr[:, ctx - 2]], -1)
    Bte = torch.cat([yb_te[:, ctx - 1], yb_te[:, ctx - 2]], -1)
    base = ridge_r2(Btr, Ytr, Bte, Yte)
    print(f"\npersistence baseline (raw ball obs @ last 2 visible states -> final ball obs): R^2 = {base:.3f}")
    hdr = f"{'ckpt':<38}{'raw_cos':>8}{'cen_cos':>8}{'mean%':>7}{'erank_c':>8}{'top%_c':>7}{'probeR2':>8}"
    print("\n[full-window rep — the panel's view]");  rows_f = [hdr]
    rows_h = [hdr]
    for p in ckpts:
        model, step = load_model(p); model.to(device)
        rf, rh = encode(model, win_fit, ctx, device)
        rf_te, rh_te = encode(model, win_te, ctx, device)
        gf, gh = geometry(torch.cat([rf, rf_te])), geometry(torch.cat([rh, rh_te]))
        r2 = ridge_r2(rh, Ytr, rh_te, Yte)
        name = Path(p).name[:37]
        rows_f.append(f"{name:<38}{gf['raw_cos']:>8.3f}{gf['cen_cos']:>8.3f}{100*gf['mean_share']:>6.1f}%"
                      f"{gf['erank_cen']:>8.1f}{100*gf['top_share_cen']:>6.1f}%{'':>8}")
        rows_h.append(f"{name:<38}{gh['raw_cos']:>8.3f}{gh['cen_cos']:>8.3f}{100*gh['mean_share']:>6.1f}%"
                      f"{gh['erank_cen']:>8.1f}{100*gh['top_share_cen']:>6.1f}%{r2:>8.3f}")
        model.to("cpu")
    print("\n".join(rows_f))
    print("\n[history-prefix rep — what a policy would consume, probe lives here]")
    print("\n".join(rows_h))
    print("\nverdict guide: raw_cos climbing while cen_cos stays flat/low and probeR2 climbs -> benign mean "
          "growth.\ncen_cos climbing in tandem or probeR2 stalling/falling below baseline -> real ratchet, intervene.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=[])
    ap.add_argument("--shards", default="/workspace/data/shards_75k")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=10)
    ap.add_argument("--device", default="cpu", help="cpu (default; won't disturb training) or cuda")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        torch.manual_seed(0)
        cfg = dict(latent_dim=32, encoder_blocks=2, encoder_hdim=64, encoder_attheads=4,
                   proj_blocks=2, proj_hdim=32, proj_attheads=4, momentum=(0.998, 1.0, 100),
                   obj_lengths=(19, 19, 9, 7, 170), emb_hdim=32,
                   mask_probs=[0.1, 0.35, 0.45, 0.05, 0.05])
        for i in range(2):
            m = JEPA(**{**cfg, "mask_probs": torch.tensor(cfg["mask_probs"])})
            torch.save({"model": m.state_dict(), "meta": {"model": cfg, "step": i}}, f"/tmp/vt{i}.pt")
        w = torch.randn(512, 15, 110).clamp(-1, 1)
        run([f"/tmp/vt{i}.pt" for i in range(2)], w[:256], w[256:], args.ctx, "cpu")
        print("\nSELFTEST OK"); return

    from training.loader import build_window_loader
    loader, _ = build_window_loader(args.shards, window=15, batch_size=args.batch, num_workers=0,
                                    pad_state=True, normalize="physical", mirror=True, gap=2,
                                    step=10, seed=1234)
    it = iter(loader)
    win_fit, win_te = next(it).float(), next(it).float()
    run(args.ckpts, win_fit, win_te, args.ctx, args.device)


if __name__ == "__main__":
    main()