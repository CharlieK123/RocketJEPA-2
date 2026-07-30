"""probes.py — frozen-encoder probing suite for R-JEPA 2.

Loads the saved checkpoint, freezes the (EMA target) encoder, and trains a
ladder of increasingly hard probes on top of cached latents to measure what
the latent space actually learned:

  L1  identity        decode an object's own state from its own token (sanity)
  L2  cross-object    decode ball position from CAR tokens only
  L3  masked recovery re-encode with ball/opp hidden (states 1-9, like training)
                      and decode the hidden object from the visible pool
  L4  future          pooled latent -> ball world pos +0.5s / +1s past the window
  L5  intent          pooled latent -> opp pos +1s, opp-jumps-soon, opp-uses-boost

Every probe is also run against two controls:
  rand   same pool from a randomly-initialized frozen encoder (architecture prior)
  raw    linear probe straight off the flattened input window (task triviality)

Latents are cached to the scratch dir so re-runs skip extraction.
Run from the repo root:  python probing/probes.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa import JEPA
from models.entity_encoding import build_obs
from training.loader import WindowDataset

FEATURE_VERSION = "v6pre"   # bump to invalidate caches when the feature set changes
SHARDS      = r"C:\Users\charl\R-JEPA2\data\shards_150k"
# checkpoint to probe; override with PROBE_CKPT=<path> to compare across steps
CKPT        = Path(os.environ.get("PROBE_CKPT",
                                  r"/checkpoints/rjepa_latest.pt"))
CACHE_DIR   = Path(r"C:\Users\charl\AppData\Local\Temp\claude\C--Users-charl-R-JEPA2\c9d4684e-48d2-4f5d-bb0d-b614744a6d3c\scratchpad\probe_cache")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_SHARD = "shard_00310.zst"   # probe-train and probe-val come from different
VAL_SHARD   = "shard_00312.zst"   # shards -> different replays, no leakage
N_TRAIN     = 40_000
N_VAL       = 8_000
EXT_WINDOW  = 20                  # 10 visible frames + 10 future frames (1s each)
VISIBLE     = 10
ENC_BATCH   = 1024

# model config — must match main.py / the checkpoint
MODEL_CFG = dict(latent_dim=256, encoder_blocks=6, encoder_hdim=1024,
                 encoder_attheads=4, proj_blocks=3, proj_hdim=128,
                 proj_attheads=4, momentum=(0.995, 1.0, 210_000),
                 obj_lengths=(19, 19, 9, 7, 170), emb_hdim=128,
                 mask_probs=torch.tensor([0.10, 0.35, 0.45, 0.05, 0.05]))

OBJS = 5                          # token order per state: self, opp, ball, env, boost
SELF_O, OPP_O, BALL_O = 0, 1, 2


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def collect_windows(shard_name, n_max, step):
    """[N, EXT_WINDOW, 110] float32 normalized windows from one shard."""
    ds = WindowDataset(SHARDS, window=EXT_WINDOW, gap=2, step=step,
                       normalize="physical", shuffle=False, drop_noise=True,
                       pad_state=True)
    ds.files = [str(Path(SHARDS) / shard_name)]
    out = []
    for i, w in enumerate(ds):
        if i >= n_max:
            break
        out.append(w)
    return torch.stack(out)


def load_encoder(trained):
    torch.manual_seed(0)
    model = JEPA(**MODEL_CFG)
    if trained:
        ck = torch.load(CKPT, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
    enc = model.target_encoder            # EMA encoder, standard for JEPA probing
    enc.requires_grad_(False)
    enc.eval()
    return enc.to(DEVICE)


def token_ids(exclude_obj=None, keep_anchor=True):
    """Flattened token indices (state-major, 5 objs/state), optionally hiding one
    object in states 1-9 exactly like build_mask does during training."""
    ids = []
    for s in range(VISIBLE):
        for o in range(OBJS):
            if exclude_obj is not None and o == exclude_obj and (s > 0 or not keep_anchor):
                continue
            ids.append(s * OBJS + o)
    return ids


@torch.no_grad()
def encode_pools(enc, windows):
    """Run the frozen encoder over the visible half of every window and return
    the pooled views each probe reads from. All [N, 384] float32 numpy."""
    car_ids = [s * OBJS + o for s in range(VISIBLE) for o in (SELF_O, OPP_O)]
    vis_ball = token_ids(exclude_obj=BALL_O)     # 41 tokens, ball hidden 1-9
    vis_opp = token_ids(exclude_obj=OPP_O)       # 41 tokens, opp hidden 1-9

    base = ("ball9", "self9", "cars", "all", "s9cat", "maskball", "maskopp")
    pools = {k: [] for k in base}
    pools.update({k + "_pre": [] for k in base})   # same views, BEFORE out_norm

    # capture the input to out_norm (= the pre-normalization token tensor). Tests
    # whether out_norm strips linearly-decodable info, or training already lost it.
    cap = {}
    h = enc.out_norm.register_forward_pre_hook(lambda _m, inp: cap.__setitem__("x", inp[0]))

    def add(tag, full, hid_ball, hid_opp):
        pools["ball9" + tag].append(full[:, 9 * OBJS + BALL_O].float().cpu())
        pools["self9" + tag].append(full[:, 9 * OBJS + SELF_O].float().cpu())
        pools["cars" + tag].append(full[:, car_ids].mean(1).float().cpu())
        pools["all" + tag].append(full.mean(1).float().cpu())
        pools["s9cat" + tag].append(full[:, 9 * OBJS:10 * OBJS].flatten(1).float().cpu())
        pools["maskball" + tag].append(hid_ball.mean(1).float().cpu())
        pools["maskopp" + tag].append(hid_opp.mean(1).float().cpu())

    for i in range(0, len(windows), ENC_BATCH):
        w = windows[i:i + ENC_BATCH, :VISIBLE].to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            tok = enc.pos(enc.embedding.build(w))          # [B, 50, D]
            full = enc(tok);              full_pre = cap["x"]
            hid_ball = enc(tok[:, vis_ball]); hidball_pre = cap["x"]   # encode like training
            hid_opp = enc(tok[:, vis_opp]);   hidopp_pre = cap["x"]
        add("", full, hid_ball, hid_opp)                   # post-norm (unchanged)
        add("_pre", full_pre, hidball_pre, hidopp_pre)     # pre-norm
    h.remove()
    return {k: torch.cat(v).numpy() for k, v in pools.items()}


@torch.no_grad()
def make_targets(windows):
    """Probe targets from the raw normalized windows. Everything lives in the
    agent's state-9 LOCAL frame (build_obs's normalize-then-rotate convention).
    World-frame readouts are deliberately NOT probed: the feature frame is
    agent-centric, so world pose is irrelevant and expected to be discarded."""
    vis = windows[:, :VISIBLE]
    self_vec, opp_vec, ball_vec, _, _ = build_obs(vis)
    fwd, right, up = vis[:, 9, 18:21], vis[:, 9, 21:24], vis[:, 9, 24:27]

    def local9(v):                       # rotate a world vector into the s9 frame
        return torch.stack([(fwd * v).sum(-1), (right * v).sum(-1),
                            (up * v).sum(-1)], dim=-1)

    pos9 = vis[:, 9, 9:12]
    fut = lambda k, lo, hi: windows[:, VISIBLE + k, lo:hi]
    t = {
        # now (state 9, exactly the model's own token frame)
        "ball_rel_now":    ball_vec[:, 9, 0:3],
        "ball_relvel_now": ball_vec[:, 9, 3:6],
        "opp_rel_now":     opp_vec[:, 9, 0:3],
        "opp_relvel_now":  opp_vec[:, 9, 3:6],
        "self_boost":      self_vec[:, 9, 15:16],
        # future, in the s9 local frame
        "ball_rel_05s":    local9(fut(4, 0, 3) - pos9),
        "ball_rel_1s":     local9(fut(9, 0, 3) - pos9),
        "ballvel_rel_1s":  local9(fut(9, 3, 6)),      # direction flips on bounces
        "opp_rel_05s":     local9(fut(4, 39, 42) - pos9),
        "opp_rel_1s":      local9(fut(9, 39, 42) - pos9),
        "self_disp_1s":    local9(fut(9, 9, 12) - pos9),   # ego-motion
        # binaries over the next second (z-height is invariant to the frame's
        # rotation about z; opp_jump was dropped: sticky flag, base rate 0.99)
        "ball_ground_1s": (windows[:, VISIBLE:, 2] < 0.06).any(1, keepdim=True).float(),
        "opp_boost_1s":  ((windows[:, VISIBLE + 9, 57] - windows[:, VISIBLE - 1, 57])
                          < -0.02).unsqueeze(1).float(),
    }
    return {k: v.numpy() for k, v in t.items()}


def make_raw_inputs(windows):
    """Flattened raw baselines. For the L3 probes the hidden object's columns are
    zeroed in states 1-9 so the baseline sees exactly what the encoder saw."""
    vis = windows[:, :VISIBLE].clone()
    full = vis.reshape(len(vis), -1).numpy()
    vb = vis.clone(); vb[:, 1:, 0:9] = 0                    # ball cols hidden
    vo = vis.clone(); vo[:, 1:, 39:61] = 0                  # opp cols hidden
    return {"raw_full": full,
            "raw_noball": vb.reshape(len(vb), -1).numpy(),
            "raw_noopp": vo.reshape(len(vo), -1).numpy()}


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def standardize(Xtr, Xva):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xva - mu) / sd


def r2(pred, y):
    """Mean per-dim R^2 on the val set."""
    num = ((pred - y) ** 2).mean(0)
    den = y.var(0) + 1e-12
    return float((1.0 - num / den).mean())


def ridge_probe(Xtr, Ytr, Xva, Yva):
    """Closed-form ridge with a small lambda sweep, best val R^2."""
    Xtr, Xva = standardize(Xtr, Xva)
    Xtr = torch.from_numpy(Xtr).to(DEVICE)
    Xva = torch.from_numpy(Xva).to(DEVICE)
    Ytr = torch.from_numpy(Ytr).to(DEVICE)
    ym = Ytr.mean(0, keepdim=True)
    G = Xtr.T @ Xtr
    b = Xtr.T @ (Ytr - ym)
    eye = torch.eye(G.size(0), device=DEVICE)
    best = -1e9
    for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0):
        W = torch.linalg.solve(G + lam * len(Xtr) / 1000 * eye, b)
        pred = (Xva @ W + ym).cpu().numpy()
        best = max(best, r2(pred, Yva))
    return best


def mlp_probe(Xtr, Ytr, Xva, Yva, binary=False, epochs=40):
    """2-layer MLP probe, Adam, best val score (R^2 or AUC)."""
    Xtr, Xva = standardize(Xtr, Xva)
    Xtr = torch.from_numpy(Xtr).to(DEVICE)
    Xva = torch.from_numpy(Xva).to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr).to(DEVICE)
    net = nn.Sequential(nn.Linear(Xtr.size(1), 512), nn.GELU(),
                        nn.Linear(512, Ytr.shape[1])).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best, bs = -1e9, 8192
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            out = net(Xtr[idx])
            loss = (F.binary_cross_entropy_with_logits(out, Ytr_t[idx]) if binary
                    else F.mse_loss(out, Ytr_t[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = net(Xva).cpu().numpy()
        best = max(best, auc(pred, Yva) if binary else r2(pred, Yva))
    return best


def logreg_probe(Xtr, Ytr, Xva, Yva):
    """Linear logistic probe, full-batch Adam, val AUC."""
    Xtr, Xva = standardize(Xtr, Xva)
    Xtr = torch.from_numpy(Xtr).to(DEVICE)
    Xva = torch.from_numpy(Xva).to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr).to(DEVICE)
    lin = nn.Linear(Xtr.size(1), 1).to(DEVICE)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(300):
        loss = F.binary_cross_entropy_with_logits(lin(Xtr), Ytr_t)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return auc(lin(Xva).cpu().numpy(), Yva)


def auc(scores, y):
    """Rank-based ROC AUC (no sklearn needed)."""
    s, y = scores.ravel(), y.ravel()
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------- #
def get_features():
    """Extract (or load cached) windows, latent pools, raw baselines, targets."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # key the cache to the checkpoint identity + mtime so features from a DIFFERENT
    # (or updated) model can never be silently reused.
    ck_tag = f"{CKPT.stem}_{int(CKPT.stat().st_mtime)}_{FEATURE_VERSION}"
    cache = CACHE_DIR / f"features_{ck_tag}.npz"
    if cache.exists():
        print(f"loading cached features from {cache}")
        d = dict(np.load(cache))
        return d

    d = {}
    for split, shard, n, step in (("tr", TRAIN_SHARD, N_TRAIN, 35),
                                  ("va", VAL_SHARD, N_VAL, 150)):
        print(f"[{split}] collecting up to {n} windows from {shard} ...")
        w = collect_windows(shard, n, step)
        print(f"[{split}] {len(w)} windows")
        for name, trained in (("jepa", True), ("rand", False)):
            enc = load_encoder(trained)
            pools = encode_pools(enc, w)
            for k, v in pools.items():
                d[f"{split}_{name}_{k}"] = v
            del enc
            torch.cuda.empty_cache()
        for k, v in make_raw_inputs(w).items():
            d[f"{split}_{k}"] = v
        for k, v in make_targets(w).items():
            d[f"{split}_y_{k}"] = v
    np.savez_compressed(cache, **d)
    print(f"cached -> {cache}")
    return d


PROBES = [
    # (tier, name, pool, raw baseline, target, task type)
    ("L1", "ball token -> ball rel pos (now)",   "ball9",    "raw_full",   "ball_rel_now", "reg-lin"),
    ("L1", "self token -> own boost (now)",      "self9",    "raw_full",   "self_boost",   "reg-lin"),
    ("L2", "car tokens -> ball rel pos (now)",   "cars",     "raw_full",   "ball_rel_now", "reg"),
    ("L3", "ball MASKED -> ball rel pos (now)",  "maskball", "raw_noball", "ball_rel_now",    "reg"),
    ("L3", "ball MASKED -> ball rel vel (now)",  "maskball", "raw_noball", "ball_relvel_now", "reg"),
    ("L3", "opp MASKED  -> opp rel pos (now)",   "maskopp",  "raw_noopp",  "opp_rel_now",     "reg"),
    ("L3", "opp MASKED  -> opp rel vel (now)",   "maskopp",  "raw_noopp",  "opp_relvel_now",  "reg"),
    ("L4", "s9 tokens -> ball rel pos +0.5s",    "s9cat",    "raw_full",   "ball_rel_05s",    "reg"),
    ("L4", "s9 tokens -> ball rel pos +1s",      "s9cat",    "raw_full",   "ball_rel_1s",     "reg"),
    ("L4", "s9 tokens -> ball local vel +1s",    "s9cat",    "raw_full",   "ballvel_rel_1s",  "reg"),
    ("L5", "s9 tokens -> opp rel pos +0.5s",     "s9cat",    "raw_full",   "opp_rel_05s",     "reg"),
    ("L5", "s9 tokens -> opp rel pos +1s",       "s9cat",    "raw_full",   "opp_rel_1s",      "reg"),
    ("L5", "s9 tokens -> own displacement +1s",  "s9cat",    "raw_full",   "self_disp_1s",    "reg"),
    ("L5", "all tokens -> ball grounded in 1s",  "all",      "raw_full",   "ball_ground_1s",  "cls"),
    ("L5", "all tokens -> opp burns boost 1s",   "all",      "raw_full",   "opp_boost_1s",    "cls"),
]


def main():
    d = get_features()
    ytr = lambda k: d[f"tr_y_{k}"]
    yva = lambda k: d[f"va_y_{k}"]

    results = []
    for tier, name, pool, rawkey, target, kind in PROBES:
        Ytr, Yva = ytr(target), yva(target)
        row = {"tier": tier, "probe": name, "target": target}
        # jepapre = pre-out_norm features (linear only: tests linear-decodability)
        if kind == "cls":
            rate = float(Yva.mean())
            row["base_rate"] = round(rate, 3)
            for tag, key in (("jepa", f"jepa_{pool}"), ("jepapre", f"jepa_{pool}_pre"),
                             ("rand", f"rand_{pool}"), ("raw", rawkey)):
                Xtr, Xva = d[f"tr_{key}"], d[f"va_{key}"]
                row[f"{tag}_lin"] = round(logreg_probe(Xtr, Ytr, Xva, Yva), 3)
                if tag in ("jepa", "rand"):
                    row[f"{tag}_mlp"] = round(
                        mlp_probe(Xtr, Ytr, Xva, Yva, binary=True), 3)
        else:
            for tag, key in (("jepa", f"jepa_{pool}"), ("jepapre", f"jepa_{pool}_pre"),
                             ("rand", f"rand_{pool}"), ("raw", rawkey)):
                Xtr, Xva = d[f"tr_{key}"], d[f"va_{key}"]
                row[f"{tag}_lin"] = round(ridge_probe(Xtr, Ytr, Xva, Yva), 3)
                if kind == "reg" and tag in ("jepa", "rand"):
                    row[f"{tag}_mlp"] = round(mlp_probe(Xtr, Ytr, Xva, Yva), 3)
        results.append(row)
        print(row)

    out = Path(__file__).parent / f"results_{CKPT.stem}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out}")

    # compact table (R^2 for reg, AUC for cls; higher is better).
    # jepaPRE = pre-out_norm linear probe -> if it beats jepa lin, out_norm is
    # stripping linearly-decodable info that training otherwise kept.
    hdr = (f"{'tier':<4} {'probe':<38} {'jepa lin':>8} {'jepaPRE':>8} {'jepa mlp':>8} "
           f"{'rand lin':>8} {'rand mlp':>8} {'raw lin':>8}")
    print(f"\nCKPT: {CKPT.name}\n" + hdr + "\n" + "-" * len(hdr))
    for r in results:
        print(f"{r['tier']:<4} {r['probe']:<38} "
              f"{r.get('jepa_lin', float('nan')):>8.3f} {r.get('jepapre_lin', float('nan')):>8.3f} "
              f"{r.get('jepa_mlp', float('nan')):>8.3f} "
              f"{r.get('rand_lin', float('nan')):>8.3f} {r.get('rand_mlp', float('nan')):>8.3f} "
              f"{r.get('raw_lin', float('nan')):>8.3f}"
              + (f"   (base rate {r['base_rate']})" if "base_rate" in r else ""))


if __name__ == "__main__":
    main()
