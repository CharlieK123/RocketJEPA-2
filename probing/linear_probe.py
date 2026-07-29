"""linear_probe.py — frozen linear-probe evaluation of RocketJEPA-2 checkpoints.

Arbitrates run 20260729-051441's loss rise past step ~17.6k:
  (A) rising weight decay is degrading the model  vs
  (B) the EMA targets got harder and the loss number is misleading.
If probe R^2 keeps improving past 17.6k it's (B); if it peaks ~15-20k and
declines it's (A). Also tests whether `erank` is a valid health metric, online
vs target encoder, pooled [384] vs last-state tokens [1920], and the
never-measured within-sample token rank.

Run ON THE VAST BOX (source there matches the checkpoints; the local models/
copy is stale — these ckpts have split pos-encoding buffers + predictor.out_proj):

    cd /workspace/RocketJEPA-2 && python probing/linear_probe.py            # everything
    python probing/linear_probe.py --device cpu                            # keep off the training GPU
    python probing/linear_probe.py --smoke --models rand                   # harness sanity check

The training run owns the GPU: encoder forwards run in small batches
(--enc-batch, default 256) and ridge Gram matrices are accumulated streaming,
so peak GPU memory is a few hundred MB. Use --device cpu if in doubt.

Anti-leakage: shards are split into disjoint train/val/test pools BEFORE any
window is cut (windows at step=1 overlap 14/15 states, so a random window split
would leak almost perfectly); mirror pairs come from the same shard and so land
on the same side. Windows+targets are cached once and every checkpoint sees the
identical tensors. Ridge alphas are tuned once (on --ref, validation split) and
reused for every checkpoint.

Outputs (in --out, default <script_dir>/linear_probe_out):
  results.csv         rows = (model, feature set), cols = task R^2 + diagnostics
  results.json        everything incl. NRMSE, alphas, shard splits, config
  probe_r2_<fs>.png   R^2 vs step per task group, baselines as hlines,
                      loss-min step marked; diagnostics panel with loss + erank
  verdict_support.md  auto-computed answers to the five report questions
"""

import argparse
import json
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# paths — auto-detect vast vs the local Windows box
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent


def _first_existing(*cands):
    for c in cands:
        if c is not None and Path(c).exists():
            return Path(c)
    return None


REPO = _first_existing(
    SCRIPT_DIR if (SCRIPT_DIR / "jepa.py").exists() else None,
    SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "jepa.py").exists() else None,
    "/workspace/RocketJEPA-2",
    SCRIPT_DIR.parent / "RocketJEPA-2",
)
if REPO is None:
    sys.exit("cannot locate the RocketJEPA-2 repo (jepa.py) relative to this script")
sys.path.insert(0, str(REPO))

from jepa import JEPA                                    # noqa: E402
from training.loader import WindowDataset, load_shard    # noqa: E402
from training.functions import (                          # noqa: E402
    collapse_metrics, covariance_metrics, effective_rank,
)

DEF_SHARDS = _first_existing("/workspace/data/shards_75k", REPO.parent / "data" / "shards_75k")
DEF_CKPTS = _first_existing("/workspace/checkpoints/20260729-051441", REPO.parent / "rjepa_ckpts")

# --------------------------------------------------------------------------- #
# probe geometry
# --------------------------------------------------------------------------- #
EXT, VISIBLE, GAP = 20, 15, 2          # 20 states @ gap 2; model sees the first 15
FUT = {"p02": 15, "p06": 17, "p10": 19}  # ext index -> +0.2s / +0.6s / +1.0s past s14
OBJS = 5
HORIZON_LABEL = {"p02": "+0.2s", "p06": "+0.6s", "p10": "+1.0s"}

# task -> (ext state index, raw column slice). Loader-normalized ("physical")
# space, world frame; slices from models/entity_encoding.py build_obs layout.
TASKS = {
    "t0_ball_pos": (14, (0, 3)),
    "t0_ball_vel": (14, (3, 6)),
    "t0_opp_pos":  (14, (39, 42)),
    "t0_opp_vel":  (14, (42, 45)),
    "t5_boost":    (14, (27, 28)),
}
for tag, k in FUT.items():
    TASKS[f"t1_ballpos_{tag}"] = (k, (0, 3))
    TASKS[f"t2_ballvel_{tag}"] = (k, (3, 6))
    TASKS[f"t3_opppos_{tag}"] = (k, (39, 42))
    TASKS[f"t4_selfpos_{tag}"] = (k, (9, 12))
TASK_ORDER = list(TASKS)

FEATSETS = ("pool", "tok", "tgt")      # online pooled / last-state tokens / target pooled
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)

DIAG_KEYS = ["eval_loss", "loss_self", "loss_opp", "loss_ball", "loss_env", "loss_boost",
             "erank", "cos", "vstd", "dead", "redun", "top_share",
             "tgt_erank", "tgt_cos", "tok_erank", "tok_cos",
             "tgt_tok_erank", "tgt_tok_cos", "prenorm_enc", "prenorm_tgt"]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def split_shards(shards_dir, n_train, n_val, n_test):
    """Disjoint shard pools by sorted index (i%6: 4->val pool, 5->test pool,
    else train), then evenly-spaced picks within each pool for corpus coverage."""
    files = sorted(Path(shards_dir).glob("shard_*.zst"))
    if not files:
        sys.exit(f"no shard_*.zst in {shards_dir}")
    pools = {"train": [f for i, f in enumerate(files) if i % 6 < 4],
             "val":   [f for i, f in enumerate(files) if i % 6 == 4],
             "test":  [f for i, f in enumerate(files) if i % 6 == 5]}
    want = {"train": n_train, "val": n_val, "test": n_test}
    out = {}
    for split, pool in pools.items():
        n = min(want[split], len(pool))
        idx = np.linspace(0, len(pool) - 1, n).round().astype(int)
        out[split] = [pool[i] for i in sorted(set(idx.tolist()))]
    got = {k: [f.name for f in v] for k, v in out.items()}
    assert not (set(got["train"]) & set(got["val"]) | set(got["train"]) & set(got["test"])
                | set(got["val"]) & set(got["test"])), "shard splits overlap"
    return out


def collect_shard(shards_dir, shard, quota):
    """Up to `quota` live [EXT, F] windows spanning the whole shard (native +
    mirrored views; both stay in this shard => this split)."""
    frames = json.loads(Path(shard).with_suffix(".json").read_text())["shape"][0]
    span = (EXT - 1) * GAP + 1
    # stride so the full shard yields ~1.4x quota (mirror doubles, dead time ~-20%)
    stride = max(1, int(frames * 2 * 0.8 * 1.0 / (quota * 1.4)) or 1)
    ds = WindowDataset(str(shards_dir), window=EXT, gap=GAP, step=stride,
                       normalize="physical", shuffle=False, seed=0,
                       drop_noise=True, pad_state=True, mirror=True)
    ds.files = [str(shard)]
    out = list(islice(iter(ds), quota * 4))
    if len(out) > quota:
        idx = np.linspace(0, len(out) - 1, quota).round().astype(int)
        out = [out[i] for i in idx]
    return torch.stack(out) if out else torch.zeros(0, EXT, ds.feat_dim)


def get_windows(shards_dir, cache_dir, splits, quotas, refresh=False):
    """{split: fp32 tensor [N, EXT, F]} — cached once, identical for every ckpt."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    wins = {}
    for split, files in splits.items():
        tag = f"{split}_s{len(files)}_n{quotas[split]}_g{GAP}e{EXT}"
        cache = cache_dir / f"win_{tag}.pt"
        if cache.exists() and not refresh:
            wins[split] = torch.load(cache, weights_only=True).float()
        else:
            t0 = time.time()
            per = max(1, quotas[split] // len(files))
            parts = [collect_shard(shards_dir, f, per) for f in files]
            w = torch.cat([p for p in parts if len(p)]).half()
            torch.save(w, cache)
            print(f"[{split}] {len(w)} windows from {len(files)} shards "
                  f"in {time.time() - t0:.0f}s")
            wins[split] = w.float()   # same fp16 roundtrip as the cached path
        print(f"[{split}] windows: {tuple(wins[split].shape)}")
    return wins


def check_schema(shards_dir):
    ds = WindowDataset(str(shards_dir), window=EXT, gap=GAP, normalize="physical",
                       pad_state=True)
    n = ds.feature_names
    assert n[0] == "ball.pos_x" and n[3] == "ball.vel_x", n[:6]
    assert n[9] == "player.pos_x" and n[12] == "player.vel_x", n[9:15]
    assert n[27] == "player.boost", n[27]
    assert n[39] == "opponent.pos_x" and n[42] == "opponent.vel_x", n[39:45]
    return n


def make_targets(win):
    return {name: win[:, k, a:b].numpy().astype(np.float32)
            for name, (k, (a, b)) in TASKS.items()}


def make_raw(win, feature_names):
    """Raw-feature baselines (action columns excluded — the model never sees them).
    raw_last = state 14 alone (THE key baseline); raw_win = all 15 states flat."""
    nonact = [i for i, n in enumerate(feature_names) if ".act." not in n]
    vis = win[:, :VISIBLE, nonact]
    return {"raw_last": vis[:, -1].numpy().astype(np.float16),
            "raw_win": vis.reshape(len(vis), -1).numpy().astype(np.float16)}


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def ckpt_cfg(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hp = (ck.get("meta") or {}).get("hparams") or {}
    cfg = dict(hp.get("model") or {})
    if not cfg:
        cj = Path(path).parent / "config.json"
        if cj.exists():
            cfg = dict(json.loads(cj.read_text()).get("model") or {})
    if not cfg:
        sys.exit(f"{path}: no model config in ckpt meta or config.json")
    cfg["mask_probs"] = torch.tensor(cfg["mask_probs"])
    cfg["momentum"] = tuple(cfg["momentum"])
    cfg["obj_lengths"] = tuple(cfg["obj_lengths"])
    cfg["STATES"] = int(hp.get("window", VISIBLE))
    return cfg, ck


def load_model(name, ckpts, device):
    if name == "rand":
        cfg, _ = ckpt_cfg(next(iter(ckpts.values())))
        torch.manual_seed(0)
        model = JEPA(**cfg)
    else:
        cfg, ck = ckpt_cfg(ckpts[name])
        model = JEPA(**cfg)
        try:
            model.load_state_dict(ck["model"], strict=True)
        except RuntimeError as e:
            sys.exit(f"{name}: STRICT LOAD FAILED — the repo source does not match "
                     f"the source that trained this run. Run this script on the box "
                     f"whose source produced the checkpoints (vast).\n{e}")
    assert model.encoder.pos.states == VISIBLE, \
        f"model expects {model.encoder.pos.states} states, probe feeds {VISIBLE}"
    return model.eval().requires_grad_(False).to(device)


@torch.no_grad()
def extract(model, win, device, bs):
    """Training-eval path exactly (training_loop.py eval block): full unmasked
    window, fp32, no autocast. Returns the three probe feature sets (fp16)."""
    enc, tgt = model.encoder, model.target_encoder
    pools = {k: [] for k in FEATSETS}
    for i in range(0, len(win), bs):
        w = win[i:i + bs, :VISIBLE].to(device)
        tok_c = enc(enc.pos(enc.embedding.build(w)))          # [B, 75, D]
        pools["pool"].append(tok_c.mean(1).cpu().half())
        pools["tok"].append(tok_c[:, -OBJS:].flatten(1).cpu().half())
        tok_t = tgt(tgt.pos(tgt.embedding.build(w)))
        pools["tgt"].append(tok_t.mean(1).cpu().half())
    return {k: torch.cat(v).numpy() for k, v in pools.items()}


def get_feats(name, ckpts, wins, cache_dir, device, bs, refresh=False):
    if name == "rand":
        tag = "rand"
    else:
        tag = f"{name}_{int(Path(ckpts[name]).stat().st_mtime)}"
    cache = cache_dir / f"feats_{tag}_n{'_'.join(str(len(wins[s])) for s in wins)}.npz"
    if cache.exists() and not refresh:
        return dict(np.load(cache))
    model = load_model(name, ckpts, device)
    t0 = time.time()
    d = {}
    for split, w in wins.items():
        for k, v in extract(model, w, device, bs).items():
            d[f"{split}_{k}"] = v
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    np.savez(cache, **d)
    print(f"  [{name}] features extracted in {time.time() - t0:.0f}s")
    return d


# --------------------------------------------------------------------------- #
# diagnostics — one fixed held-out batch, identical for every checkpoint
# --------------------------------------------------------------------------- #
@torch.no_grad()
def diagnostics(name, ckpts, feats, win_test, device, bs):
    out = {}
    # pooled-rep panel from the cached test features (first 4096 rows)
    rep = torch.from_numpy(feats["test_pool"][:4096].astype(np.float32))
    out["erank"], out["cos"], out["vstd"], out["dead"] = collapse_metrics(rep)
    out["redun"], out["top_share"] = covariance_metrics(rep)
    trep = torch.from_numpy(feats["test_tgt"][:4096].astype(np.float32))
    te, tc, _, _ = collapse_metrics(trep)
    out["tgt_erank"], out["tgt_cos"] = te, tc  # tgt_cos: never logged in training

    model = load_model(name, ckpts, device)
    enc, tgt = model.encoder, model.target_encoder

    # within-sample token rank — the blind spot: every training metric pools the
    # 75 tokens before measuring, so token-space homogenization is invisible.
    w = win_test[:64, :VISIBLE].to(device)
    for pref, m in (("", enc), ("tgt_", tgt)):
        toks = m(m.pos(m.embedding.build(w))).float()          # [64, 75, D]
        ers = [effective_rank(toks[i]) for i in range(len(toks))]
        tn = F.normalize(toks, dim=-1)
        sim = tn @ tn.transpose(1, 2)                          # [64, 75, 75]
        off = ~torch.eye(sim.size(1), dtype=torch.bool, device=device)
        out[f"{pref}tok_erank"] = float(np.mean(ers))
        out[f"{pref}tok_cos"] = sim[:, off].mean().item()

    # pre-out_norm activation RMS (the wd-degradation signature from the run logs)
    cap = {}
    h = [enc.out_norm.register_forward_pre_hook(
            lambda m, i: cap.__setitem__("enc", i[0].float().pow(2).mean().sqrt().item())),
         tgt.out_norm.register_forward_pre_hook(
            lambda m, i: cap.__setitem__("tgt", i[0].float().pow(2).mean().sqrt().item()))]

    # JEPA loss recomputed on the identical batch with identical (seeded) masks —
    # unlike the training log's loss these numbers are comparable across ckpts.
    if name == "rand":
        torch.manual_seed(0)   # keep rand's weights-draw isolated from mask draws
    tot, per_obj, nb = 0.0, np.zeros(OBJS), 0
    counts = np.zeros(OBJS)
    torch.manual_seed(123)
    for i in range(0, min(len(win_test), 4096), bs):
        wb = win_test[i:i + bs, :VISIBLE].to(device)
        res = model(wb)
        z_hat, z = res[0], res[1]
        tot += F.smooth_l1_loss(z_hat, z).item()
        nb += 1
        if len(res) > 2:
            obj = (res[2][:, 0] % OBJS).cpu().numpy()
            per = F.smooth_l1_loss(z_hat.float(), z.float(),
                                   reduction="none").mean(dim=(1, 2)).cpu().numpy()
            for o in range(OBJS):
                sel = obj == o
                per_obj[o] += per[sel].sum()
                counts[o] += sel.sum()
    for hh in h:
        hh.remove()
    out["eval_loss"] = tot / max(nb, 1)
    out["prenorm_enc"], out["prenorm_tgt"] = cap.get("enc"), cap.get("tgt")
    for o, nm in enumerate(("self", "opp", "ball", "env", "boost")):
        out[f"loss_{nm}"] = float(per_obj[o] / counts[o]) if counts[o] else None
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# ridge — streamed Gram so GPU memory stays tiny while the training run owns it
# --------------------------------------------------------------------------- #
class Ridge:
    def __init__(self, Xtr, device, chunk=8192):
        X32 = Xtr.astype(np.float32)
        self.mu = X32.mean(0, keepdims=True)
        self.sd = X32.std(0, keepdims=True) + 1e-6
        self.device, self.chunk = device, chunk
        D = X32.shape[1]
        self.n = len(X32)
        self.G = torch.zeros(D, D, device=device, dtype=torch.float32)
        self._Xtr = X32

    def _std_chunks(self, X):
        for i in range(0, len(X), self.chunk):
            yield torch.from_numpy(
                ((X[i:i + self.chunk].astype(np.float32) - self.mu) / self.sd)
            ).to(self.device)

    def solve(self, Ytr, alphas):
        """W per alpha for the full concatenated target matrix."""
        Y = torch.from_numpy(Ytr).to(self.device)
        self.ym = Y.mean(0, keepdim=True)
        Yc = Y - self.ym
        B = torch.zeros(self.G.size(0), Y.size(1), device=self.device)
        self.G.zero_()
        r = 0
        for xc in self._std_chunks(self._Xtr):
            self.G += xc.T @ xc
            B += xc.T @ Yc[r:r + len(xc)]
            r += len(xc)
        eye = torch.eye(self.G.size(0), device=self.device)
        return {a: torch.linalg.solve(self.G + a * self.n / 1000 * eye, B)
                for a in alphas}

    def predict(self, X, W):
        return torch.cat([xc @ W for xc in self._std_chunks(X)]) + self.ym


def scores(pred, y):
    """(mean R^2, mean NRMSE) over target dims."""
    mse = ((pred - y) ** 2).mean(0)
    var = y.var(0) + 1e-12
    return float((1.0 - mse / var).mean()), float((mse / var).sqrt().mean())


def run_probes(Xd, targets, alphas_store, fs_key, tune, device):
    """Fit ridge for every task from one feature set. tune=True picks alpha per
    task on val and stores it; otherwise the stored alpha is used verbatim."""
    Ytr = np.concatenate([targets["train"][t] for t in TASK_ORDER], 1)
    cols, c = {}, 0
    for t in TASK_ORDER:
        d = targets["train"][t].shape[1]
        cols[t] = slice(c, c + d)
        c += d
    solver = Ridge(Xd["train"], device)
    need = ALPHAS if tune else tuple(sorted({alphas_store[fs_key][t] for t in TASK_ORDER}))
    Ws = solver.solve(Ytr, need)
    preds = {sp: {a: solver.predict(Xd[sp], W) for a, W in Ws.items()}
             for sp in ("val", "test")}
    out = {}
    for t in TASK_ORDER:
        if tune:
            yv = torch.from_numpy(targets["val"][t]).to(device)
            best = max(need, key=lambda a: scores(preds["val"][a][:, cols[t]], yv)[0])
            alphas_store.setdefault(fs_key, {})[t] = best
        a = alphas_store[fs_key][t]
        yt = torch.from_numpy(targets["test"][t]).to(device)
        r2, nrmse = scores(preds["test"][a][:, cols[t]], yt)
        out[t] = {"r2": round(r2, 4), "nrmse": round(nrmse, 4), "alpha": a}
    return out


def const_baseline(targets):
    out = {}
    for t in TASK_ORDER:
        mu = torch.from_numpy(targets["train"][t].mean(0, keepdims=True))
        yt = torch.from_numpy(targets["test"][t])
        r2, nrmse = scores(mu.expand_as(yt), yt)
        out[t] = {"r2": round(r2, 4), "nrmse": round(nrmse, 4), "alpha": None}
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def write_csv(results, out_path):
    diag = {m: r.get("diagnostics", {}) for m, r in results.items()}
    lines = ["model,step,features," + ",".join(TASK_ORDER) + "," + ",".join(DIAG_KEYS)]
    for m, r in results.items():
        step = r.get("step", "")
        for fs, probes in r["probes"].items():
            row = [m, str(step), fs]
            row += [f"{probes[t]['r2']:.4f}" if t in probes else "" for t in TASK_ORDER]
            row += [(f"{diag[m][k]:.4f}" if isinstance(diag[m].get(k), float)
                     else "") for k in DIAG_KEYS]
            lines.append(",".join(row))
    out_path.write_text("\n".join(lines))


TASK_GROUPS = [
    ("T1 future ball position", [f"t1_ballpos_{h}" for h in FUT]),
    ("T2 future ball velocity", [f"t2_ballvel_{h}" for h in FUT]),
    ("T3 future opp position", [f"t3_opppos_{h}" for h in FUT]),
    ("T4 future self position", [f"t4_selfpos_{h}" for h in FUT]),
    ("T0 present readout", ["t0_ball_pos", "t0_ball_vel", "t0_opp_pos", "t0_opp_vel"]),
    ("T5 own boost", ["t5_boost"]),
]
HORIZON_BLUES = {"p02": "#9ecae1", "p06": "#4292c6", "p10": "#08519c"}  # light->dark
T0_COLORS = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0"]


def make_plots(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — skipping plots")
        return
    ck = sorted([m for m in results if m.startswith("step")],
                key=lambda m: results[m]["step"])
    steps = [results[m]["step"] / 1000 for m in ck]
    dg = {m: results[m]["diagnostics"] for m in ck}
    losses = [dg[m].get("eval_loss") for m in ck]
    loss_min_step = steps[int(np.nanargmin([l if l is not None else np.nan
                                            for l in losses]))] if any(losses) else None

    for fs in FEATSETS:
        fig, axes = plt.subplots(2, 4, figsize=(22, 9))
        fig.suptitle(f"Linear probe R² vs training step — features: {fs} "
                     f"(test split, shard-disjoint)", fontsize=13)
        for ax, (title, tasks) in zip(axes.flat, TASK_GROUPS):
            for j, t in enumerate(tasks):
                col = (HORIZON_BLUES[t.rsplit("_", 1)[1]]
                       if t.rsplit("_", 1)[1] in HORIZON_BLUES else T0_COLORS[j % 4])
                lab = (HORIZON_LABEL.get(t.rsplit("_", 1)[1], t.split("_", 1)[1]))
                ys = [results[m]["probes"][fs][t]["r2"] for m in ck]
                ax.plot(steps, ys, "o-", color=col, lw=2, ms=5, label=lab)
                for base, style in (("raw_last", ":"), ("rand", "--")):
                    if base in results and fs in results[base]["probes"]:
                        ax.axhline(results[base]["probes"][fs][t]["r2"],
                                   color=col, ls=style, lw=1, alpha=0.55)
                    elif base == "raw_last" and base in results:
                        ax.axhline(results[base]["probes"]["raw_last"][t]["r2"],
                                   color=col, ls=style, lw=1, alpha=0.55)
            if loss_min_step is not None:
                ax.axvline(loss_min_step, color="#999999", ls="-", lw=1, alpha=0.6)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("step (k)")
            ax.set_ylabel("R²")
            ax.grid(alpha=0.25, lw=0.5)
            ax.legend(fontsize=7, frameon=False,
                      title="dotted=raw  dashed=rand", title_fontsize=7)
        # diagnostics panels
        ax = axes.flat[6]
        ax.plot(steps, losses, "o-", color="#555555", lw=2, ms=5, label="eval loss (fixed batch)")
        ax.set_ylabel("smooth-L1 loss")
        ax.set_xlabel("step (k)")
        ax2 = ax.twinx()
        ax2.plot(steps, [dg[m]["erank"] for m in ck], "s--", color="#4269d0",
                 lw=1.5, ms=4, label="erank (online pooled)")
        ax2.set_ylabel("erank")
        ax.set_title("recomputed loss + erank (identical batch)", fontsize=10)
        ln = ax.get_lines() + ax2.get_lines()
        ax.legend(ln, [l.get_label() for l in ln], fontsize=7, frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
        ax = axes.flat[7]
        for key, col, mk in (("tok_erank", "#4269d0", "o"), ("tgt_tok_erank", "#ff725c", "s"),
                             ("tgt_erank", "#6cc5b0", "^")):
            ax.plot(steps, [dg[m][key] for m in ck], mk + "-", color=col, lw=1.5,
                    ms=4, label=key)
        ax.set_title("token-level rank (max 75) & target erank", fontsize=10)
        ax.set_xlabel("step (k)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        p = out_dir / f"probe_r2_{fs}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"wrote {p}")


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def verdict_support(results, out_dir):
    ck = sorted([m for m in results if m.startswith("step")],
                key=lambda m: results[m]["step"])
    steps = [results[m]["step"] for m in ck]
    dg = {m: results[m]["diagnostics"] for m in ck}
    L = ["# Verdict support (auto-computed)", "",
         f"checkpoints: {', '.join(ck)}", ""]

    L.append("## 1. Does probe quality peak near 17.6k or keep improving? (headline)")
    for fs in FEATSETS:
        L.append(f"\n### features: {fs}")
        L.append("| task | " + " | ".join(f"{s//1000}k" for s in steps) + " | peak |")
        L.append("|" + "---|" * (len(steps) + 2))
        for t in TASK_ORDER:
            ys = [results[m]["probes"][fs][t]["r2"] for m in ck]
            pk = steps[int(np.argmax(ys))]
            L.append(f"| {t} | " + " | ".join(f"{y:.3f}" for y in ys) + f" | **{pk//1000}k** |")
    fut = [t for t in TASK_ORDER if t[1] in "1234"]
    for fs in FEATSETS:
        pks = [steps[int(np.argmax([results[m]['probes'][fs][t]['r2'] for m in ck]))]
               for t in fut]
        L.append(f"\n[{fs}] median peak step over future tasks: "
                 f"{int(np.median(pks)) // 1000}k  (peaks: {sorted(p // 1000 for p in pks)})")

    L.append("\n## 2. Does probe R2 track erank?")
    er = [dg[m]["erank"] for m in ck]
    for fs in FEATSETS:
        rs = [pearson([results[m]["probes"][fs][t]["r2"] for m in ck], er) for t in fut]
        L.append(f"[{fs}] Pearson r(R2, erank) over future tasks: "
                 f"median {np.nanmedian(rs):.2f}, range [{np.nanmin(rs):.2f}, {np.nanmax(rs):.2f}]")

    L.append("\n## 3. Online vs target encoder (pooled)")
    for m in ck:
        d = [results[m]["probes"]["tgt"][t]["r2"] - results[m]["probes"]["pool"][t]["r2"]
             for t in TASK_ORDER]
        L.append(f"{m}: mean R2(tgt - online) = {np.mean(d):+.4f}")

    L.append("\n## 4. Pooled [384] vs last-state tokens [1920]")
    for m in ck:
        d = [results[m]["probes"]["tok"][t]["r2"] - results[m]["probes"]["pool"][t]["r2"]
             for t in TASK_ORDER]
        L.append(f"{m}: mean R2(tok - pool) = {np.mean(d):+.4f}")

    L.append("\n## 5. Is tok_erank falling?")
    L.append("| step | tok_erank | tgt_tok_erank | tok_cos | erank(pooled) | tgt_erank | eval_loss |")
    L.append("|---|---|---|---|---|---|---|")
    for m in ck:
        d = dg[m]
        L.append(f"| {results[m]['step']//1000}k | {d['tok_erank']:.1f} | "
                 f"{d['tgt_tok_erank']:.1f} | {d['tok_cos']:.3f} | {d['erank']:.1f} | "
                 f"{d['tgt_erank']:.1f} | {d['eval_loss']:.5f} |")

    p = out_dir / "verdict_support.md"
    p.write_text("\n".join(L))
    print(f"wrote {p}")
    print("\n".join(L[:6]))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shards", default=DEF_SHARDS)
    ap.add_argument("--ckpt-dir", default=DEF_CKPTS)
    ap.add_argument("--out", default=SCRIPT_DIR / "linear_probe_out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--enc-batch", type=int, default=256,
                    help="encoder forward batch (keep small: training owns the GPU)")
    ap.add_argument("--models", default="all",
                    help='"all", or comma list e.g. "step5k,step30k,rand"')
    ap.add_argument("--ref", default=None,
                    help="checkpoint whose val split tunes the ridge alphas "
                         "(default: middle checkpoint)")
    ap.add_argument("--train-shards", type=int, default=9)
    ap.add_argument("--val-shards", type=int, default=3)
    ap.add_argument("--test-shards", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=48_000)
    ap.add_argument("--n-val", type=int, default=12_000)
    ap.add_argument("--n-test", type=int, default=16_000)
    ap.add_argument("--refresh", action="store_true", help="rebuild caches")
    ap.add_argument("--smoke", action="store_true", help="tiny data, harness check")
    args = ap.parse_args()
    if args.smoke:
        args.train_shards, args.val_shards, args.test_shards = 2, 1, 1
        args.n_train, args.n_val, args.n_test = 3000, 1000, 1200

    if args.shards is None or args.ckpt_dir is None:
        sys.exit("could not auto-locate shards/checkpoints; pass --shards/--ckpt-dir")
    out_dir = Path(args.out)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = {p.stem.replace("rjepa_", ""): p
             for p in sorted(Path(args.ckpt_dir).glob("rjepa_step*k.pt"),
                             key=lambda p: int(p.stem.split("step")[1][:-1]))}
    print(f"repo={REPO}\nshards={args.shards}\nckpts={list(ckpts)}\ndevice={args.device}")
    names = (list(ckpts) + ["rand"] if args.models == "all"
             else args.models.split(","))
    ref = args.ref or (sorted([n for n in names if n.startswith("step")],
                              key=lambda n: int(n[4:-1]))
                       or ["rand"])[len([n for n in names if n.startswith("step")]) // 2]
    print(f"alpha-tuning reference: {ref}")

    quotas = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    feature_names = check_schema(args.shards)
    splits = split_shards(args.shards, args.train_shards, args.val_shards,
                          args.test_shards)
    wins = get_windows(args.shards, cache_dir, splits, quotas, args.refresh)
    targets = {sp: make_targets(w) for sp, w in wins.items()}

    alphas_path = out_dir / "alphas.json"
    alphas_store = json.loads(alphas_path.read_text()) if alphas_path.exists() else {}
    results = {}
    t00 = time.time()

    # reference first so its tuned alphas exist for everyone else
    order = ([ref] if ref in names else []) + [n for n in names if n != ref]
    for name in order:
        print(f"\n=== {name} ===")
        feats = get_feats(name, ckpts, wins, cache_dir, args.device,
                          args.enc_batch, args.refresh)
        step = (torch.load(ckpts[name], map_location="cpu", weights_only=False)
                ["meta"]["step"] if name in ckpts else None)
        res = {"step": step, "probes": {}, "diagnostics": {}}
        for fs in FEATSETS:
            tune = (name == ref) and (fs not in alphas_store)
            Xd = {sp: feats[f"{sp}_{fs}"] for sp in ("train", "val", "test")}
            res["probes"][fs] = run_probes(Xd, targets, alphas_store, fs,
                                           tune, args.device)
            print(f"  [{fs}] " + " ".join(
                f"{t.split('_', 1)[1]}={res['probes'][fs][t]['r2']:.3f}"
                for t in ("t0_ball_pos", "t1_ballpos_p10", "t3_opppos_p10")))
        res["diagnostics"] = diagnostics(name, ckpts, feats, wins["test"],
                                         args.device, args.enc_batch)
        d = res["diagnostics"]
        print(f"  diag: loss={d['eval_loss']:.5f} erank={d['erank']:.1f} "
              f"cos={d['cos']:.3f} tgt_erank={d['tgt_erank']:.1f} "
              f"tgt_cos={d['tgt_cos']:.3f} tok_erank={d['tok_erank']:.1f}")
        results[name] = res
        alphas_path.write_text(json.dumps(alphas_store, indent=1))

    # raw-feature + constant baselines (feature sets of their own)
    print("\n=== raw / const baselines ===")
    raw = {sp: make_raw(w, feature_names) for sp, w in wins.items()}
    res = {"step": None, "probes": {}, "diagnostics": {}}
    for rk in ("raw_last", "raw_win"):
        Xd = {sp: raw[sp][rk] for sp in ("train", "val", "test")}
        tune = rk not in alphas_store
        res["probes"][rk] = run_probes(Xd, targets, alphas_store, rk, tune, args.device)
        print(f"  [{rk}] " + " ".join(
            f"{t.split('_', 1)[1]}={res['probes'][rk][t]['r2']:.3f}"
            for t in ("t0_ball_pos", "t1_ballpos_p10", "t3_opppos_p10")))
    results["raw_last"] = {"step": None, "diagnostics": {},
                           "probes": {"raw_last": res["probes"]["raw_last"]}}
    results["raw_win"] = {"step": None, "diagnostics": {},
                          "probes": {"raw_win": res["probes"]["raw_win"]}}
    results["const"] = {"step": None, "diagnostics": {},
                        "probes": {"const": const_baseline(targets)}}
    alphas_path.write_text(json.dumps(alphas_store, indent=1))

    meta = {"repo": str(REPO), "shards": str(args.shards),
            "ckpt_dir": str(args.ckpt_dir),
            "splits": {k: [f.name for f in v] for k, v in splits.items()},
            "n_windows": {k: len(v) for k, v in wins.items()},
            "quotas": quotas, "gap": GAP, "ext": EXT, "visible": VISIBLE,
            "ref": ref, "alphas": alphas_store,
            "note": ("shard-disjoint splits; 'unseen by training' NOT guaranteed "
                     "(loader shuffles within a 1-epoch pass) — treat probe R2 as "
                     "a RELATIVE comparison across checkpoints, which is the "
                     "question being asked."),
            }
    (out_dir / "results.json").write_text(json.dumps(
        {"meta": meta, "results": results}, indent=1))
    write_csv(results, out_dir / "results.csv")
    print(f"\nwrote {out_dir / 'results.csv'} and results.json")

    if any(m.startswith("step") for m in results):
        make_plots(results, out_dir)
        verdict_support(results, out_dir)
    print(f"done in {(time.time() - t00) / 60:.1f} min")


if __name__ == "__main__":
    main()
