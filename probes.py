"""
probes.py — the "is this training actually working" battery.

Frozen-encoder LINEAR probes (ridge, closed form) on identical data across
checkpoints. Linear-only is deliberate: it measures how ACCESSIBLE information
is in the rep, which is what a fresh PPO head sees. Everything a probe reads is
derived from the window tensor itself (build_obs / raw physics columns), so no
external labels are needed and every checkpoint is scored on the same exam.

WHAT GETS PROBED
  reps      hist_mean  mean-pooled history tokens (deployment rep)
            last_state concat of the 5 tokens at the last visible state
            pred_tok@k the PREDICTOR's predicted token for (future state k, obj)
                       -> decode that object's obs from it (world-model metric)
  targets   nowcast (k=0) and forecasts at k in {1, mid, max} future states:
            ball_rel_pos/vel, opp_rel_pos, self_pos, self_boost
  events    (AUC) touch-in-future, ball-airborne@final, ball-near-agent@final
  baselines persistence (last 2 visible ball obs), raw last-state obs vector,
            and a RANDOM-INIT encoder of the same architecture (arch prior floor)

USAGE (box, repo root; cpu default so training is undisturbed):
  python probes.py --shards /workspace/data/shards_75k \
      --ckpts ../checkpoints/<RUN>/rjepa_step5k.pt ../checkpoints/<RUN>/rjepa_step20k.pt \
              ../checkpoints/<RUN>/rjepa_step40k.pt ../checkpoints/<RUN>/rjepa_latest.pt \
      --json probe_results.json
  python probes.py --selftest        # plumbing check anywhere
"""
import argparse, json, math
from pathlib import Path

import torch
import torch.nn.functional as F

from jepa import JEPA
from models.entity_encoding import build_obs, POS_DIV, POS_SCALE

OBJS = 5
# (name, build_obs tuple index, col slice)  — verified against entity_encoding.py
REG_TARGETS = [
    ("ball_rel_pos", 2, slice(0, 3)),
    ("ball_rel_vel", 2, slice(3, 6)),
    ("opp_rel_pos",  1, slice(0, 3)),
    ("self_pos",     0, slice(0, 3)),
    ("self_boost",   0, slice(15, 16)),
]
OBJ_OF_TARGET = {"ball": 2, "opp": 1, "self": 0}   # token slot per object


# ---------------------------------------------------------------- ridge/auc --
def ridge_fit_eval(Xtr, Ytr, Xte, Yte, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(Xtr.size(0), generator=g)
    k = int(0.8 * len(idx)); fit, val = idx[:k], idx[k:]
    mx, sx = Xtr[fit].mean(0), Xtr[fit].std(0).clamp_min(1e-6)
    my = Ytr[fit].mean(0)
    Z = lambda X: (X - mx) / sx

    def solve(lam):
        Xf, Yf = Z(Xtr[fit]), Ytr[fit] - my
        A = Xf.T @ Xf + lam * Xf.size(0) * torch.eye(Xf.size(1))
        return torch.linalg.solve(A, Xf.T @ Yf)

    def r2(W, X, Y):
        P = Z(X) @ W + my
        return (1 - (Y - P).pow(2).sum() / (Y - Y.mean(0)).pow(2).sum().clamp_min(1e-12)).item()

    best = max(((r2(solve(l), Xtr[val], Ytr[val]), l) for l in (1e-4, 1e-3, 1e-2, 1e-1)))
    W = solve(best[1])
    return r2(W, Xte, Yte), (Z(Xte) @ W + my)


def auc(scores, labels):
    """Rank AUC. scores [N], labels bool [N]."""
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float); ranks[order] = torch.arange(len(order), dtype=torch.float)
    pos = labels.bool(); n1, n0 = pos.sum().item(), (~pos).sum().item()
    if n1 == 0 or n0 == 0: return float("nan")
    return ((ranks[pos].sum().item() - n1 * (n1 - 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------- encode --
@torch.no_grad()
def reps_and_preds(model, win, ctx, device, chunk=256):
    """Returns hist_mean [B,D], last_state [B,5D], pred toks [B,F,5,D]."""
    hm, ls, pt = [], [], []
    T = win.size(1) * OBJS
    midx = torch.arange(ctx * OBJS, T, device=device)
    for i in range(0, win.size(0), chunk):
        w = win[i:i + chunk].to(device)
        tok = model.encoder.pos(model.encoder.embedding.build(w))
        hist = tok[:, : ctx * OBJS]
        ctx_lat = model.encoder(hist)
        hm.append(ctx_lat.mean(1).float().cpu())
        ls.append(ctx_lat[:, -OBJS:].reshape(ctx_lat.size(0), -1).float().cpu())
        z = model.predictor(ctx_lat, midx.unsqueeze(0).expand(w.size(0), -1))
        Fh = z.size(1) // OBJS
        pt.append(z.view(z.size(0), Fh, OBJS, -1).float().cpu())
    return torch.cat(hm), torch.cat(ls), torch.cat(pt)


def _find_cfg(d):
    """Recursively locate a model-config dict anywhere in checkpoint meta."""
    if isinstance(d, dict):
        if "latent_dim" in d:
            return d
        for v in d.values():
            r = _find_cfg(v)
            if r is not None:
                return r
    return None


def load_model(path, rand_init=False):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = _find_cfg(ckpt.get("meta"))
    if cfg is None:                      # fall back to main.py under common names
        import main as _m
        for name in ("MODEL_CFG", "model_cfg", "CFG", "cfg"):
            cfg = getattr(_m, name, None)
            if isinstance(cfg, dict) and "latent_dim" in cfg:
                break
            cfg = None
    if cfg is None:
        raise SystemExit(
            f"\nNo model config found in {Path(path).name} meta and none importable from main.py.\n"
            "Fix: add to main.py (module level) a dict matching THIS run's architecture, e.g.\n"
            "  MODEL_CFG = dict(latent_dim=..., encoder_blocks=..., encoder_hdim=..., encoder_attheads=...,\n"
            "                   proj_blocks=..., proj_hdim=..., proj_attheads=..., momentum=(0.998, 1.0, 100_000),\n"
            "                   obj_lengths=(19, 19, 9, 7, 170), emb_hdim=..., mask_probs=[.1, .35, .45, .05, .05])\n"
            "A wrong value fails loudly at load_state_dict (shape mismatch) — that guard is your friend.")
    torch.manual_seed(123)
    kw = dict(cfg)
    if "mask_probs" in kw:
        kw["mask_probs"] = torch.tensor(kw["mask_probs"])
    model = JEPA(**kw)
    if not rand_init:
        model.load_state_dict(ckpt["model"])
    return model.eval()


# --------------------------------------------------------------------- main --
def run(names_models, win_fit, win_te, ctx, device, json_out):
    S = win_fit.size(1)
    obs_tr, obs_te = build_obs(win_fit.float()), build_obs(win_te.float())
    Fh = S - ctx
    hors = sorted({1, (Fh + 1) // 2, Fh})
    results = {}

    def tgt(obs, name_i_sl, s):
        _, i, sl = name_i_sl
        return obs[i][:, s, sl]

    # ---- baselines (model-free) ----
    print(f"\nctx={ctx} states of history, horizons k={hors} (x0.2s). test N={win_te.size(0)}")
    print("== model-free baselines (R^2) ==")
    ball = ("ball", 2, slice(0, 9))
    for k in hors:
        s = ctx - 1 + k
        Btr = torch.cat([obs_tr[2][:, ctx - 1], obs_tr[2][:, ctx - 2]], -1)
        Bte = torch.cat([obs_te[2][:, ctx - 1], obs_te[2][:, ctx - 2]], -1)
        r_p, _ = ridge_fit_eval(Btr, tgt(obs_tr, ball, s)[:, :3], Bte, tgt(obs_te, ball, s)[:, :3])
        Rtr = torch.cat([o[:, ctx - 1] for o in obs_tr], -1)
        Rte = torch.cat([o[:, ctx - 1] for o in obs_te], -1)
        r_r, _ = ridge_fit_eval(Rtr, tgt(obs_tr, ball, s)[:, :3], Rte, tgt(obs_te, ball, s)[:, :3])
        results[f"baseline/k{k}"] = {"persistence": r_p, "raw_obs": r_r}
        print(f"  k={k}: ball_rel_pos  persistence={r_p:+.3f}  raw_last_obs={r_r:+.3f}")

    # ---- event labels (from raw physics cols; thresholds in game units) ----
    z_air = 300.0 / POS_DIV[2]
    near = 1000.0 / POS_SCALE
    def events(win, obs):
        fut = slice(ctx, S)
        return {
            "touch_in_future":   win[:, fut, 71].amax(1) > 0.5,
            "ball_air_final":    win[:, -1, 2] > z_air,
            "ball_near_final":   obs[2][:, -1, 0:3].norm(dim=-1) < near,
        }
    ev_tr, ev_te = events(win_fit, obs_tr), events(win_te, obs_te)

    # ---- per checkpoint ----
    for name, model in names_models:
        model.to(device)
        hm_tr, ls_tr, pt_tr = reps_and_preds(model, win_fit, ctx, device)
        hm_te, ls_te, pt_te = reps_and_preds(model, win_te, ctx, device)
        model.to("cpu")
        res = {}
        print(f"\n== {name} ==")
        for tname, i, sl in REG_TARGETS:
            row = {}
            for k in [0] + hors:
                s = ctx - 1 + k
                Ytr, Yte = tgt(obs_tr, (tname, i, sl), s), tgt(obs_te, (tname, i, sl), s)
                r_hm, _ = ridge_fit_eval(hm_tr, Ytr, hm_te, Yte)
                row[f"hist_mean/k{k}"] = r_hm
                if k == 0:
                    r_ls, _ = ridge_fit_eval(ls_tr, Ytr, ls_te, Yte)
                    row["last_state/k0"] = r_ls
                elif tname.split("_")[0] in OBJ_OF_TARGET:
                    o = OBJ_OF_TARGET[tname.split("_")[0]]
                    r_pt, _ = ridge_fit_eval(pt_tr[:, k - 1, o], Ytr, pt_te[:, k - 1, o], Yte)
                    row[f"pred_tok/k{k}"] = r_pt
            res[tname] = row
            ks = " ".join(f"k{k}:{row.get(f'hist_mean/k{k}', float('nan')):+.3f}" for k in [0] + hors)
            pts = " ".join(f"k{k}:{row[f'pred_tok/k{k}']:+.3f}" for k in hors if f"pred_tok/k{k}" in row)
            extra = f"  | last_state k0:{row['last_state/k0']:+.3f}" if "last_state/k0" in row else ""
            print(f"  {tname:<13} hist_mean {ks}{extra}" + (f"  | pred_tok {pts}" if pts else ""))
        arow = {}
        for ename in ev_tr:
            _, sc = ridge_fit_eval(hm_tr, ev_tr[ename].float().unsqueeze(1) * 2 - 1,
                                   hm_te, ev_te[ename].float().unsqueeze(1) * 2 - 1)
            arow[ename] = auc(sc.squeeze(1), ev_te[ename])
        res["events_auc"] = arow
        print("  events (AUC, hist_mean): " + "  ".join(f"{k}={v:.3f}" for k, v in arow.items()))
        results[name] = res

    if json_out:
        Path(json_out).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {json_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=[])
    ap.add_argument("--shards", default="/workspace/data/shards_75k")
    ap.add_argument("--ctx", type=int, default=9)
    ap.add_argument("--batches", type=int, default=2, help="loader batches each for fit/test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default="probe_results.json")
    ap.add_argument("--no-random-baseline", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        torch.manual_seed(0)
        cfg = dict(latent_dim=32, encoder_blocks=2, encoder_hdim=64, encoder_attheads=4,
                   proj_blocks=2, proj_hdim=32, proj_attheads=4, momentum=(0.998, 1.0, 100),
                   obj_lengths=(19, 19, 9, 7, 170), emb_hdim=32,
                   mask_probs=[0.1, 0.35, 0.45, 0.05, 0.05])
        m = JEPA(**{**cfg, "mask_probs": torch.tensor(cfg["mask_probs"])})
        torch.save({"model": m.state_dict(), "meta": {"model": cfg}}, "/tmp/pb.pt")
        w = torch.randn(1024, 15, 110).clamp(-1, 1)
        mods = [("ckpt", load_model("/tmp/pb.pt")), ("random_init", load_model("/tmp/pb.pt", rand_init=True))]
        run(mods, w[:512], w[512:], args.ctx, "cpu", None)
        print("\nSELFTEST OK"); return

    from training.loader import build_window_loader
    loader, _ = build_window_loader(args.shards, window=15, batch_size=2048, num_workers=0,
                                    pad_state=True, normalize="physical", mirror=True,
                                    gap=2, step=10, seed=4321)
    it = iter(loader)
    fit = torch.cat([next(it).float() for _ in range(args.batches)])
    te = torch.cat([next(it).float() for _ in range(args.batches)])
    mods = [(Path(p).name, load_model(p)) for p in args.ckpts]
    if not args.no_random_baseline:
        mods.insert(0, ("random_init", load_model(args.ckpts[0], rand_init=True)))
    run(mods, fit, te, args.ctx, args.device, args.json)


if __name__ == "__main__":
    main()