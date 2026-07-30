"""suite.py — objective-grounded linear + non-linear probe battery for R-JEPA 2.

WHAT THE TRAINING OBJECTIVE ACTUALLY OPTIMIZES (derived from current source):
  Per sample: 10 states (gap=2 -> 0.2s apart, 1.8s span), 5 object tokens/state.
  ONE object o ~ (self .10, opp .35, ball .45, env .05, boost .05) is hidden in
  states 1-9 (state-0 anchor stays visible). The CONTEXT encoder embeds the 41
  visible tokens; the predictor gets those latents + 9 positional mask queries
  and must output the EMA TARGET encoder's (LayerNorm'd) latents at the hidden
  slots. Smooth-L1, loss on masked positions only.

  => DIRECTLY optimized ("direct"):
     - visible-object latents must carry the hidden object's 1.8s trajectory
       (cross-object routing), mostly for ball (45%) and opp (35%);
     - the predictor's ability to reconstruct that trajectory ON THE TARGET
       LATENT MANIFOLD, given everyone else's FULL window (this is state
       completion / inpainting-with-context, NOT forecasting).
  => NOT optimized ("transfer" — failing these is not a training bug, but they
     are what downstream PPO consumes):
     - forecasting past the window (features strictly from states 0..9,
       targets at +0.4/+1/+2s);
     - game events (touch/impulse, grounded, boost usage, who-is-closer);
     - actions: controller inputs are NOT model inputs, so self-action decode
       = behaviour-cloning readout, opp-action decode = intent prior;
     - world-frame pose (agent-centric features), metric calibration, goals
       (drop_noise removes them from every window).

PROBE FAMILIES (the linear/non-linear separability ladder), per target:
  L ridge / logistic (linear)     K kNN (k=32, cosine — "is the info there at all")
  R random-Fourier RBF kernel ridge (smooth non-linear)   M 1-hidden-layer MLP (512)

CONTROLS: rand = same-architecture random-init encoder; raw = flattened input
window WITHOUT action columns (the model never sees actions), with the hidden
object's columns zeroed for masked probes.

Substrates: context encoder (gets gradients; receives masked views in training),
EMA target encoder (produces targets; the usual JEPA probe substrate), and the
predictor's outputs decoded to physical units.

Run:  .venv/Scripts/python.exe probing/suite.py --models quick
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "RocketJEPA-2"
sys.path.insert(0, str(SRC))

from jepa import JEPA                                   # noqa: E402
from models.entity_encoding import build_obs, POS_DIV, POS_SCALE   # noqa: E402
from training.loader import WindowDataset               # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHARDS = ROOT / "data" / "shards_150k"
SCRATCH = Path(r"C:\Users\charl\AppData\Local\Temp\claude\C--Users-charl-R-JEPA2"
               r"\608955f3-eeda-4ffe-9ee5-bf8fb838fe3c\scratchpad") / "suite_cache"
OUT = ROOT / "probing" / "suite_results"

TRAIN_SHARD, VAL_SHARD = "shard_00310.zst", "shard_00312.zst"
N_TRAIN, N_VAL = 24_000, 8_000
TRAIN_STEP, VAL_STEP = 35, 100     # window-start stride (raw frames)
MLP_EPOCHS, LOGREG_ITERS = 25, 300
GAP, VISIBLE, EXT = 2, 10, 20      # 20 states x gap 2 = 39 raw frames = 3.9s span
FUT04, FUT1, FUT2 = 11, 14, 19     # ext-state index: +0.4s / +1.0s / +2.0s past s9
ENC_BATCH = 512

OBJS = 5
SELF_O, OPP_O, BALL_O, ENV_O, BOOST_O = range(5)
MASKABLE = (BALL_O, OPP_O, SELF_O)
OBJ_NAME = {BALL_O: "ball", OPP_O: "opp", SELF_O: "self"}
# raw feature columns per object (for zeroing in raw baselines)
OBJ_RAWCOLS = {BALL_O: (0, 9), SELF_O: (9, 31), OPP_O: (39, 61)}
NONACT = [i for i in range(110) if not (31 <= i < 39 or 61 <= i < 69)]

RUN_DIR = ROOT / "checkpoints" / "20260728-153023"
CKPTS = {
    "step5k":  RUN_DIR / "rjepa_step5k.pt",
    "step8k":  RUN_DIR / "rjepa_step8k.pt",
    "step10k": RUN_DIR / "rjepa_step10k.pt",
    "step12k": RUN_DIR / "rjepa_step12k.pt",
    "step15k": RUN_DIR / "rjepa_step15k.pt",
    "step20k": RUN_DIR / "rjepa_step20k.pt",
    "step25k": RUN_DIR / "rjepa_step25k.pt",
    "step30k": RUN_DIR / "rjepa_step30k.pt",
    "alarm15800": RUN_DIR / "rjepa_ALARM_step15800_collapse.pt",
}
QUICK = ["step5k", "step15k", "step30k", "alarm15800"]

# STATES pinned to 10: this run trained with window=10, but the source's default
# is now 15 (the cloud run) — without it the pos-encoding buffers mismatch.
MODEL_CFG = dict(latent_dim=256, encoder_blocks=6, encoder_hdim=1024,
                 encoder_attheads=4, proj_blocks=3, proj_hdim=128,
                 proj_attheads=4, momentum=(0.998, 1.0, 210_000),
                 obj_lengths=(19, 19, 9, 7, 170), emb_hdim=128,
                 mask_probs=torch.tensor([0.10, 0.35, 0.45, 0.05, 0.05]),
                 STATES=10)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def collect_windows(shard, n_max, step):
    """All live [EXT, 110] windows from one shard (native view, no mirror),
    uniformly subsampled to n_max for whole-shard replay coverage."""
    ds = WindowDataset(str(SHARDS), window=EXT, gap=GAP, step=step,
                       normalize="physical", shuffle=False, drop_noise=True,
                       pad_state=True)
    ds.files = [str(SHARDS / shard)]
    names = ds.feature_names
    assert names[31] == "player.act.throttle" and names[32] == "player.act.steer", names[31:33]
    assert names[36] == "player.act.jump" and names[37] == "player.act.boost"
    assert names[61] == "opponent.act.throttle" and names[67] == "opponent.act.boost"
    assert names[57] == "opponent.boost" and names[69] == "env.seconds_remaining"
    out = list(islice(iter(ds), 60_000))
    if len(out) > n_max:
        idx = np.linspace(0, len(out) - 1, n_max).round().astype(int)
        out = [out[i] for i in idx]
    return torch.stack(out)


def get_windows(refresh=False):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    wins = {}
    for split, shard, n, step in (("tr", TRAIN_SHARD, N_TRAIN, TRAIN_STEP),
                                  ("va", VAL_SHARD, N_VAL, VAL_STEP)):
        cache = SCRATCH / f"win_{split}_{Path(shard).stem}_n{n}_s{step}_g{GAP}e{EXT}.pt"
        if cache.exists() and not refresh:
            wins[split] = torch.load(cache, weights_only=True).float()
        else:
            t0 = time.time()
            w = collect_windows(shard, n, step)
            torch.save(w.half(), cache)
            print(f"[{split}] {len(w)} windows from {shard} in {time.time()-t0:.0f}s")
            wins[split] = w.float()
        print(f"[{split}] windows: {tuple(wins[split].shape)}")
    return wins["tr"], wins["va"]


@torch.no_grad()
def make_targets(win):
    """Probe targets. 'Now' targets live in the model's own token frame (current
    build_obs); future positions are formed in RAW units then rotated into the
    s9 agent frame and isotropically scaled (matches entity_encoding's fix)."""
    vis = win[:, :VISIBLE]
    self_vec, opp_vec, ball_vec, _, _ = build_obs(vis)
    fwd, right, up = vis[:, 9, 18:21], vis[:, 9, 21:24], vis[:, 9, 24:27]

    def loc9(v):                                    # world -> s9 agent frame
        return torch.stack([(fwd * v).sum(-1), (right * v).sum(-1),
                            (up * v).sum(-1)], dim=-1)

    pos_div = torch.tensor(POS_DIV)
    p9_raw = vis[:, 9, 9:12] * pos_div              # self position, raw units

    def rel9(p_norm):                               # future pos -> rel, s9 frame
        return loc9(p_norm * pos_div - p9_raw) / POS_SCALE

    def rdist(a_norm, b_norm):                      # raw-unit distance
        return ((a_norm * pos_div - b_norm * pos_div) ** 2).sum(-1).sqrt()

    f = lambda k: win[:, k]
    # ball impulse: sharp velocity change (touch OR bounce) within +1s.
    dv = win[:, VISIBLE - 1:FUT1 + 1, 3:6].diff(dim=1).norm(dim=-1)   # steps s9->s14
    t = {
        # ---- now (state 9, model token frame) ----
        "ball_rel_now":    ball_vec[:, 9, 0:3],
        "ball_relvel_now": ball_vec[:, 9, 3:6],
        "opp_rel_now":     opp_vec[:, 9, 0:3],
        "opp_relvel_now":  opp_vec[:, 9, 3:6],
        "self_boost":      self_vec[:, 9, 15:16],
        "self_locvel_now": self_vec[:, 9, 3:6],
        "env_secs":        vis[:, 9, 69:70],
        "pads3":           vis[:, 9, [76, 93, 109]],
        # trajectory profile (for predictor state-k decode + target-content ceilings)
        "ball_rel_s1":     ball_vec[:, 1, 0:3],
        "ball_rel_s5":     ball_vec[:, 5, 0:3],
        "opp_rel_s1":      opp_vec[:, 1, 0:3],
        "opp_rel_s5":      opp_vec[:, 5, 0:3],
        # ---- future, s9 local frame (TRANSFER: never optimized) ----
        "ball_rel_04":     rel9(f(FUT04)[:, 0:3]),
        "ball_rel_1s":     rel9(f(FUT1)[:, 0:3]),
        "ball_rel_2s":     rel9(f(FUT2)[:, 0:3]),
        "ballvel_loc_1s":  loc9(f(FUT1)[:, 3:6]),   # ball vel is isotropic-normalized
        "opp_rel_1s":      rel9(f(FUT1)[:, 39:42]),
        "self_disp_1s":    rel9(f(FUT1)[:, 9:12]),
        # ---- events (TRANSFER) ----
        "ball_ground_1s":  (win[:, VISIBLE:FUT1 + 1, 2] < 0.06).any(1, keepdim=True).float(),
        "ball_impulse_1s": (dv > 0.02).any(1, keepdim=True).float(),
        "opp_burn_1s":     ((f(FUT1)[:, 57] - vis[:, 9, 57]) < -0.02).unsqueeze(1).float(),
        "opp_closer_1s":   (rdist(f(FUT1)[:, 39:42], f(FUT1)[:, 0:3]) <
                            rdist(f(FUT1)[:, 9:12], f(FUT1)[:, 0:3])).unsqueeze(1).float(),
        "opp_air_1s":      (f(FUT1)[:, 41] > 300.0 / 2044.0).unsqueeze(1).float(),
        # ---- actions at s9 (TRANSFER: not model inputs) ----
        "self_thrsteer":   vis[:, 9, [31, 32]],
        "self_jump":       vis[:, 9, 36:37],
        "self_boostbtn":   vis[:, 9, 37:38],
        "opp_thrsteer":    vis[:, 9, [61, 62]],
        "opp_boostbtn":    vis[:, 9, 67:68],
    }
    return {k: v.numpy().astype(np.float32) for k, v in t.items()}


def make_raw(win):
    """Raw-input baselines, ACTION COLUMNS EXCLUDED (model never sees them).
    Masked variants zero the hidden object's columns in states 1-9."""
    vis = win[:, :VISIBLE]
    d = {"raw_full": vis[:, :, NONACT].reshape(len(vis), -1).numpy().astype(np.float16)}
    for o, (lo, hi) in OBJ_RAWCOLS.items():
        v = vis.clone()
        v[:, 1:, lo:hi] = 0.0
        d[f"raw_no{OBJ_NAME[o]}"] = v[:, :, NONACT].reshape(len(v), -1).numpy().astype(np.float16)
    return d


# --------------------------------------------------------------------------- #
# models + extraction
# --------------------------------------------------------------------------- #
def infer_cfg(sd):
    """Reconstruct the JEPA config from a state dict (att heads NOT recoverable
    from shapes — assumed 4, the value every run has used)."""
    n_blocks = lambda p: len({k.split(".")[2] for k in sd if k.startswith(p)})
    cfg = dict(MODEL_CFG)
    cfg.update(
        latent_dim=sd["encoder.norm.0.weight"].shape[0],
        STATES=sd["encoder.pos.state_pe"].shape[0],
        encoder_blocks=n_blocks("encoder.attention."),
        encoder_hdim=sd["encoder.ffn.0.0.weight"].shape[0],
        proj_blocks=n_blocks("predictor.attention."),
        proj_hdim=sd["predictor.ffn.0.0.weight"].shape[0],
        emb_hdim=sd["encoder.embedding.object_projections.0.0.weight"].shape[0],
        obj_lengths=tuple(sd[f"encoder.embedding.object_projections.{i}.0.weight"].shape[1]
                          for i in range(OBJS)),
        mask_probs=sd["mask_probs"].clone(),
    )
    return cfg


def load_model(name):
    torch.manual_seed(0)
    if name == "rand":
        model = JEPA(**MODEL_CFG)
    else:
        ck = torch.load(CKPTS[name], map_location="cpu", weights_only=True)
        sd = ck.get("model", ck)
        model = JEPA(**infer_cfg(sd))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if k.split(".")[0] in ("encoder", "predictor", "target_encoder")]
        if bad:
            raise RuntimeError(f"{name}: missing param keys {bad[:8]}")
        if missing or unexpected:
            print(f"  [{name}] non-fatal key diffs: missing={missing} unexpected={unexpected}")
    model.eval().requires_grad_(False)
    return model.to(DEVICE)


def token_ids(exclude_obj):
    """Visible flat token ids with `exclude_obj` hidden in states 1-9 (anchor kept)."""
    return [s * OBJS + o for s in range(VISIBLE) for o in range(OBJS)
            if not (o == exclude_obj and s > 0)]


@torch.no_grad()
def extract(model, win):
    """Frozen forward passes -> feature pools (numpy f16) + extraction scalars."""
    enc, tgt, pred = model.encoder, model.target_encoder, model.predictor
    vis_ids = {o: torch.tensor(token_ids(o), device=DEVICE) for o in MASKABLE}
    midx = {o: torch.tensor([o + s * OBJS for s in range(1, VISIBLE)], device=DEVICE)
            for o in MASKABLE}
    pools = defaultdict(list)
    prenorm = {"ctx": 0.0, "tgt": 0.0, "n": 0}
    cap = {}
    h1 = enc.out_norm.register_forward_pre_hook(lambda m, i: cap.__setitem__("ctx", i[0]))
    h2 = tgt.out_norm.register_forward_pre_hook(lambda m, i: cap.__setitem__("tgt", i[0]))
    latcos = {o: torch.zeros(VISIBLE - 1) for o in MASKABLE}
    latsl1 = {o: 0.0 for o in MASKABLE}
    # latent-space objective calibration: trivial baselines + context control.
    # copy  = cos(target s0 anchor latent, target latent at slot k)  ("just copy the anchor")
    # mean  = cos(batch-mean target latent at slot k, target latent) ("predict the average")
    # shuf  = predictor fed ANOTHER sample's context (batch roll)    ("unconditional prior?")
    copycos = {o: torch.zeros(VISIBLE - 1) for o in MASKABLE}
    meancos = {o: torch.zeros(VISIBLE - 1) for o in MASKABLE}
    shufcos = {o: torch.zeros(VISIBLE - 1) for o in MASKABLE}
    nb = 0

    car_ids = [s * OBJS + o for s in range(VISIBLE) for o in (SELF_O, OPP_O)]
    for i in range(0, len(win), ENC_BATCH):
        w = win[i:i + ENC_BATCH, :VISIBLE].to(DEVICE)
        B = w.size(0)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            tok_c = enc.pos(enc.embedding.build(w))
            full_c = enc(tok_c)
            prenorm["ctx"] += cap["ctx"].float().pow(2).mean().sqrt().item()
            tok_t = tgt.pos(tgt.embedding.build(w))
            full_t = tgt(tok_t)
            prenorm["tgt"] += cap["tgt"].float().pow(2).mean().sqrt().item()
            prenorm["n"] += 1

            def keep(key, x):
                pools[key].append(x.float().cpu().half())

            keep("ctx_all", full_c.mean(1))
            keep("tgt_all", full_t.mean(1))
            keep("tgt_s9cat", full_t[:, 9 * OBJS:].flatten(1))
            keep("tgt_cars", full_t[:, car_ids].mean(1))
            for o, nm in ((BALL_O, "ball"), (OPP_O, "opp"), (SELF_O, "self"),
                          (ENV_O, "env"), (BOOST_O, "boost")):
                keep(f"tgt_{nm}9", full_t[:, 9 * OBJS + o])
            for o in MASKABLE:   # target tokens at s1/s5 = per-state content ceilings
                keep(f"tgt_{OBJ_NAME[o]}1", full_t[:, 1 * OBJS + o])
                keep(f"tgt_{OBJ_NAME[o]}5", full_t[:, 5 * OBJS + o])

            for o in MASKABLE:
                nm = OBJ_NAME[o]
                lat = enc(tok_c[:, vis_ids[o]])          # [B, 41, D] like training
                p = pred(lat, midx[o].unsqueeze(0).expand(B, -1))   # [B, 9, D]
                keep(f"mask{nm}_pool", lat.mean(1))
                keep(f"mask{nm}_cat",
                     torch.cat([lat[:, o], lat[:, -(OBJS - 1):].flatten(1)], dim=1))
                for tag, k in (("s1", 0), ("s5", 4), ("s9", 8)):
                    keep(f"pred{nm}_{tag}", p[:, k])
                tlat = full_t[:, midx[o]]
                latcos[o] += F.cosine_similarity(p.float(), tlat.float(), dim=-1).mean(0).cpu()
                latsl1[o] += F.smooth_l1_loss(p.float(), tlat.float()).item()
                anchor = full_t[:, o:o + 1]        # object o's state-0 token
                copycos[o] += F.cosine_similarity(anchor.float(), tlat.float(), dim=-1).mean(0).cpu()
                mu = tlat.float().mean(0, keepdim=True)
                meancos[o] += F.cosine_similarity(mu, tlat.float(), dim=-1).mean(0).cpu()
                p_shuf = pred(lat.roll(1, 0), midx[o].unsqueeze(0).expand(B, -1))
                shufcos[o] += F.cosine_similarity(p_shuf.float(), tlat.float(), dim=-1).mean(0).cpu()
        nb += 1
    h1.remove(); h2.remove()
    out = {k: torch.cat(v).numpy() for k, v in pools.items()}
    scalars = {
        "prenorm_ctx": prenorm["ctx"] / prenorm["n"],
        "prenorm_tgt": prenorm["tgt"] / prenorm["n"],
        "pred_latcos": {OBJ_NAME[o]: (latcos[o] / nb).tolist() for o in MASKABLE},
        "pred_sl1":    {OBJ_NAME[o]: latsl1[o] / nb for o in MASKABLE},
        "copy_latcos": {OBJ_NAME[o]: (copycos[o] / nb).tolist() for o in MASKABLE},
        "mean_latcos": {OBJ_NAME[o]: (meancos[o] / nb).tolist() for o in MASKABLE},
        "shuf_latcos": {OBJ_NAME[o]: (shufcos[o] / nb).tolist() for o in MASKABLE},
    }
    return out, scalars


def get_pools(name, win_tr, win_va, refresh=False):
    tag = name if name == "rand" else f"{name}_{int(CKPTS[name].stat().st_mtime)}"
    cache = SCRATCH / f"pools_v2_{tag}_n{len(win_tr)}v{len(win_va)}.npz"
    scache = SCRATCH / f"scalars_v2_{tag}_n{len(win_tr)}v{len(win_va)}.json"
    if cache.exists() and scache.exists() and not refresh:
        d = dict(np.load(cache))
        return d, json.loads(scache.read_text())
    t0 = time.time()
    model = load_model(name)
    d = {}
    tr, sc_tr = extract(model, win_tr)
    va, sc_va = extract(model, win_va)
    for k, v in tr.items():
        d[f"tr_{k}"] = v
    for k, v in va.items():
        d[f"va_{k}"] = v
    del model
    torch.cuda.empty_cache()
    np.savez(cache, **d)
    scache.write_text(json.dumps({"tr": sc_tr, "va": sc_va}))
    print(f"  [{name}] extracted in {time.time()-t0:.0f}s")
    return d, {"tr": sc_tr, "va": sc_va}


# --------------------------------------------------------------------------- #
# probe families
# --------------------------------------------------------------------------- #
def _std(Xtr, Xva):
    Xtr = Xtr.astype(np.float32); Xva = Xva.astype(np.float32)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xva - mu) / sd


def r2(pred, y):
    num = ((pred - y) ** 2).mean(0)
    den = y.var(0) + 1e-12
    return float((1.0 - num / den).mean())


def auc(scores, y):
    s, y = scores.ravel(), y.ravel()
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ridge(Xtr, Ytr, Xva, Yva):
    Xtr, Xva = _std(Xtr, Xva)
    Xtr = torch.from_numpy(Xtr).to(DEVICE); Xva = torch.from_numpy(Xva).to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr).to(DEVICE)
    ym = Ytr_t.mean(0, keepdim=True)
    G = Xtr.T @ Xtr
    b = Xtr.T @ (Ytr_t - ym)
    eye = torch.eye(G.size(0), device=DEVICE)
    best = -1e9
    for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0):
        W = torch.linalg.solve(G + lam * len(Xtr) / 1000 * eye, b)
        best = max(best, r2((Xva @ W + ym).cpu().numpy(), Yva))
    return best


def logreg(Xtr, Ytr, Xva, Yva):
    torch.manual_seed(0)
    Xtr, Xva = _std(Xtr, Xva)
    Xtr = torch.from_numpy(Xtr).to(DEVICE); Xva = torch.from_numpy(Xva).to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr).to(DEVICE)
    lin = nn.Linear(Xtr.size(1), 1).to(DEVICE)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(LOGREG_ITERS):
        loss = F.binary_cross_entropy_with_logits(lin(Xtr), Ytr_t)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return auc(lin(Xva).cpu().numpy(), Yva)


def knn(Xtr, Ytr, Xva, Yva, kind, k=32):
    Xt = F.normalize(torch.from_numpy(Xtr.astype(np.float32)), dim=1).half().to(DEVICE)
    Xv = F.normalize(torch.from_numpy(Xva.astype(np.float32)), dim=1).half().to(DEVICE)
    Yt = torch.from_numpy(Ytr).to(DEVICE)
    preds = []
    for i in range(0, len(Xv), 1024):
        sims = Xv[i:i + 1024] @ Xt.T
        idx = sims.topk(k, dim=1).indices
        preds.append(Yt[idx].float().mean(1).cpu())
    pred = torch.cat(preds).numpy()
    return auc(pred, Yva) if kind == "cls" else r2(pred, Yva)


def rff(Xtr, Ytr, Xva, Yva, D=2048):
    """RBF kernel ridge via random Fourier features (regression only)."""
    torch.manual_seed(0)
    Xtr, Xva = _std(Xtr, Xva)
    Xt = torch.from_numpy(Xtr).to(DEVICE); Xv = torch.from_numpy(Xva).to(DEVICE)
    sub = Xt[torch.randperm(len(Xt), device=DEVICE)[:2048]]
    d2 = torch.cdist(sub, sub).pow(2)
    gamma = 1.0 / (d2[d2 > 0].median().item() + 1e-9)
    Wf = torch.randn(Xt.size(1), D, device=DEVICE) * math.sqrt(2 * gamma)
    bf = torch.rand(D, device=DEVICE) * 2 * math.pi
    zt = math.sqrt(2.0 / D) * torch.cos(Xt @ Wf + bf)
    zv = math.sqrt(2.0 / D) * torch.cos(Xv @ Wf + bf)
    Ytr_t = torch.from_numpy(Ytr).to(DEVICE)
    ym = Ytr_t.mean(0, keepdim=True)
    G = zt.T @ zt
    b = zt.T @ (Ytr_t - ym)
    eye = torch.eye(D, device=DEVICE)
    best = -1e9
    for lam in (1e-3, 1e-2, 1e-1, 1.0):
        Wr = torch.linalg.solve(G + lam * len(zt) * eye / 1000, b)
        best = max(best, r2((zv @ Wr + ym).cpu().numpy(), Yva))
    return best


def mlp(Xtr, Ytr, Xva, Yva, kind, epochs=None, hidden=512):
    epochs = epochs or MLP_EPOCHS
    torch.manual_seed(0)
    Xtr, Xva = _std(Xtr, Xva)
    Xt = torch.from_numpy(Xtr).to(DEVICE); Xv = torch.from_numpy(Xva).to(DEVICE)
    Yt = torch.from_numpy(Ytr).to(DEVICE)
    net = nn.Sequential(nn.Linear(Xt.size(1), hidden), nn.GELU(),
                        nn.Linear(hidden, Ytr.shape[1])).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best, bs = -1e9, 8192
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            out = net(Xt[idx])
            loss = (F.binary_cross_entropy_with_logits(out, Yt[idx]) if kind == "cls"
                    else F.mse_loss(out, Yt[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = net(Xv).cpu().numpy()
        best = max(best, auc(pred, Yva) if kind == "cls" else r2(pred, Yva))
    return best


FAM_FN = {
    "lin": lambda X, Y, Xv, Yv, kind: logreg(X, Y, Xv, Yv) if kind == "cls" else ridge(X, Y, Xv, Yv),
    "knn": lambda X, Y, Xv, Yv, kind: knn(X, Y, Xv, Yv, kind),
    "rff": lambda X, Y, Xv, Yv, kind: float("nan") if kind == "cls" else rff(X, Y, Xv, Yv),
    "mlp": lambda X, Y, Xv, Yv, kind: mlp(X, Y, Xv, Yv, kind),
}
FAM_CODE = {"L": "lin", "K": "knn", "R": "rff", "M": "mlp"}


# --------------------------------------------------------------------------- #
# battery
# --------------------------------------------------------------------------- #
# (tier, name, feature pool, target, kind, families, optimized-status)
PROBES = [
    ("D", "masked ball  -> ball rel pos @s9",   "maskball_pool", "ball_rel_now",    "reg", "LKRM", "direct"),
    ("D", "masked ball  -> ball rel vel @s9",   "maskball_pool", "ball_relvel_now", "reg", "LKRM", "direct"),
    ("D", "masked ball CAT -> ball rel pos",    "maskball_cat",  "ball_rel_now",    "reg", "LM",   "direct"),
    ("D", "masked opp   -> opp rel pos @s9",    "maskopp_pool",  "opp_rel_now",     "reg", "LKRM", "direct"),
    ("D", "masked opp   -> opp rel vel @s9",    "maskopp_pool",  "opp_relvel_now",  "reg", "LKRM", "direct"),
    ("D", "masked opp CAT -> opp rel pos",      "maskopp_cat",   "opp_rel_now",     "reg", "LM",   "direct"),
    ("D", "masked self  -> own boost",          "maskself_pool", "self_boost",      "reg", "LM",   "direct"),
    ("D", "masked self CAT -> self loc vel",    "maskself_cat",  "self_locvel_now", "reg", "LM",   "direct"),
    ("P", "predictor(ball) s9 -> ball rel pos", "predball_s9",   "ball_rel_now",    "reg", "LM",   "direct"),
    ("P", "predictor(ball) s5 -> ball rel pos@s5", "predball_s5", "ball_rel_s5",    "reg", "L",    "direct"),
    ("P", "predictor(ball) s1 -> ball rel pos@s1", "predball_s1", "ball_rel_s1",    "reg", "L",    "direct"),
    ("P", "predictor(opp) s9 -> opp rel pos",   "predopp_s9",    "opp_rel_now",     "reg", "LM",   "direct"),
    ("P", "predictor(opp) s5 -> opp rel pos@s5", "predopp_s5",   "opp_rel_s5",      "reg", "L",    "direct"),
    ("P", "predictor(opp) s1 -> opp rel pos@s1", "predopp_s1",   "opp_rel_s1",      "reg", "L",    "direct"),
    ("P", "predictor(ball) s9 -> ball rel vel", "predball_s9",   "ball_relvel_now", "reg", "L",    "direct"),
    # C-tier: content CEILINGS — decode from the TARGET latent at the same masked
    # slot the predictor is scored on. R2(pred)/R2(target) = recovery fraction;
    # a falling ceiling with flat fraction = target erosion, not predictor failure.
    ("C", "TARGET ball s1 -> ball rel pos@s1",  "tgt_ball1",     "ball_rel_s1",     "reg", "L",    "ceiling"),
    ("C", "TARGET ball s5 -> ball rel pos@s5",  "tgt_ball5",     "ball_rel_s5",     "reg", "L",    "ceiling"),
    ("C", "TARGET opp s1  -> opp rel pos@s1",   "tgt_opp1",      "opp_rel_s1",      "reg", "L",    "ceiling"),
    ("C", "TARGET opp s5  -> opp rel pos@s5",   "tgt_opp5",      "opp_rel_s5",      "reg", "L",    "ceiling"),
    ("R", "car tokens -> ball rel pos (routing)", "tgt_cars",    "ball_rel_now",    "reg", "LKRM", "direct"),
    ("I", "ball token -> ball rel pos",         "tgt_ball9",     "ball_rel_now",    "reg", "L",    "sanity"),
    ("I", "ball token -> ball rel vel",         "tgt_ball9",     "ball_relvel_now", "reg", "L",    "sanity"),
    ("I", "opp token  -> opp rel pos",          "tgt_opp9",      "opp_rel_now",     "reg", "L",    "sanity"),
    ("I", "self token -> own boost",            "tgt_self9",     "self_boost",      "reg", "L",    "sanity"),
    ("I", "env token  -> secs remaining",       "tgt_env9",      "env_secs",        "reg", "L",    "sanity"),
    ("I", "boost token -> 3 pad timers",        "tgt_boost9",    "pads3",           "reg", "L",    "sanity"),
    ("F", "s9 tokens -> ball rel pos +0.4s",    "tgt_s9cat",     "ball_rel_04",     "reg", "LM",   "transfer"),
    ("F", "s9 tokens -> ball rel pos +1s",      "tgt_s9cat",     "ball_rel_1s",     "reg", "LKRM", "transfer"),
    ("F", "s9 tokens -> ball rel pos +2s",      "tgt_s9cat",     "ball_rel_2s",     "reg", "LM",   "transfer"),
    ("F", "s9 tokens -> ball loc vel +1s",      "tgt_s9cat",     "ballvel_loc_1s",  "reg", "LKRM", "transfer"),
    ("F", "s9 tokens -> opp rel pos +1s",       "tgt_s9cat",     "opp_rel_1s",      "reg", "LM",   "transfer"),
    ("F", "s9 tokens -> own displacement +1s",  "tgt_s9cat",     "self_disp_1s",    "reg", "LM",   "transfer"),
    ("F", "pooled all -> ball rel pos +1s",     "tgt_all",       "ball_rel_1s",     "reg", "L",    "transfer"),
    ("E", "s9 tokens -> ball grounded <=1s",    "tgt_s9cat",     "ball_ground_1s",  "cls", "LM",   "transfer"),
    ("E", "s9 tokens -> ball impulse <=1s",     "tgt_s9cat",     "ball_impulse_1s", "cls", "LM",   "transfer"),
    ("E", "s9 tokens -> opp burns boost <=1s",  "tgt_s9cat",     "opp_burn_1s",     "cls", "LM",   "transfer"),
    ("E", "s9 tokens -> opp closer @+1s",       "tgt_s9cat",     "opp_closer_1s",   "cls", "LM",   "transfer"),
    ("E", "s9 tokens -> opp airborne @+1s",     "tgt_s9cat",     "opp_air_1s",      "cls", "LM",   "transfer"),
    ("A", "s9 tokens -> SELF throttle+steer",   "tgt_s9cat",     "self_thrsteer",   "reg", "LM",   "transfer"),
    ("A", "s9 tokens -> SELF boost held",       "tgt_s9cat",     "self_boostbtn",   "cls", "LM",   "transfer"),
    ("A", "s9 tokens -> SELF jump held",        "tgt_s9cat",     "self_jump",       "cls", "LM",   "transfer"),
    ("A", "s9 tokens -> OPP throttle+steer",    "tgt_s9cat",     "opp_thrsteer",    "reg", "LRM",  "transfer"),
    ("A", "s9 tokens -> OPP boost held",        "tgt_s9cat",     "opp_boostbtn",    "cls", "LM",   "transfer"),
]

# raw-baseline feature for each pool (masked pools see the same hidden-object view)
def raw_key(pool):
    for o, nm in OBJ_NAME.items():
        if pool.startswith((f"mask{nm}", f"pred{nm}")):
            return f"raw_no{nm}"
    return "raw_full"


SAMP_EFF = [("maskball_pool", "ball_rel_now"), ("tgt_s9cat", "ball_rel_1s"),
            ("tgt_s9cat", "opp_thrsteer")]
SAMP_NS = (2000, 8000, None)          # None -> all train rows


def run_battery(feats, targets_tr, targets_va, model_name):
    rows = []
    for tier, name, pool, targ, kind, fams, status in PROBES:
        key_tr, key_va = f"tr_{pool}", f"va_{pool}"
        if key_tr not in feats:
            continue
        Xtr, Xva = feats[key_tr], feats[key_va]
        Ytr, Yva = targets_tr[targ], targets_va[targ]
        row = {"tier": tier, "probe": name, "pool": pool, "target": targ,
               "kind": kind, "status": status}
        if kind == "cls":
            row["base_rate"] = round(float(Yva.mean()), 3)
        for code in fams:
            fam = FAM_CODE[code]
            t0 = time.time()
            try:
                score = FAM_FN[fam](Xtr, Ytr, Xva, Yva, kind)
            except Exception as e:      # keep the battery running
                print(f"    !! {model_name} {name} [{fam}] failed: {e}")
                score = float("nan")
            row[fam] = round(score, 3) if score == score else None
            row[f"{fam}_sec"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"  [{model_name}] {tier} {name:<42}"
              + " ".join(f"{FAM_CODE[c]}={row.get(FAM_CODE[c])}" for c in fams))
    return rows


def run_sample_eff(feats, targets_tr, targets_va):
    out = []
    for pool, targ in SAMP_EFF:
        if f"tr_{pool}" not in feats:
            continue
        Xtr, Xva = feats[f"tr_{pool}"], feats[f"va_{pool}"]
        Ytr, Yva = targets_tr[targ], targets_va[targ]
        for n in SAMP_NS:
            n = n or len(Xtr)
            out.append({"pool": pool, "target": targ, "n": int(n),
                        "lin": round(ridge(Xtr[:n], Ytr[:n], Xva, Yva), 3)})
    return out


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def effective_rank(x):
    x = torch.from_numpy(x.astype(np.float32))
    x = x - x.mean(0, keepdim=True)
    cov = x.T @ x / len(x)
    ev = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
    p = ev / ev.sum()
    return float((-(p * p.log()).sum()).exp()), float((ev.max() / ev.sum()))


def geo_stats(x):
    er, top1 = effective_rank(x)
    xt = torch.from_numpy(x[:4096].astype(np.float32))
    xn = F.normalize(xt, dim=1)                    # plain cosine, like training panel
    sim = xn @ xn.T
    off = sim[~torch.eye(len(xn), dtype=torch.bool)]
    stds = xt.std(0)
    return {"erank": round(er, 1), "top1_share": round(top1, 3),
            "cos": round(float(off.mean()), 3),
            "dead_dims": round(float((stds < 0.01 * stds.mean()).float().mean()), 3)}


def geometry(feats, scalars):
    g = {}
    for key in ("ctx_all", "tgt_all", "tgt_s9cat", "maskball_pool",
                "predball_s9", "tgt_cars"):
        if f"va_{key}" in feats:
            g[key] = geo_stats(feats[f"va_{key}"])
    g["per_object_token_erank"] = {
        nm: round(effective_rank(feats[f"va_tgt_{nm}9"])[0], 1)
        for nm in ("self", "opp", "ball", "env", "boost") if f"va_tgt_{nm}9" in feats}
    g["prenorm_rms"] = {"ctx": round(scalars["va"]["prenorm_ctx"], 3),
                        "tgt": round(scalars["va"]["prenorm_tgt"], 3)}
    g["pred_latcos_per_state"] = {k: [round(v, 3) for v in vv]
                                  for k, vv in scalars["va"]["pred_latcos"].items()}
    g["pred_sl1"] = {k: round(v, 4) for k, v in scalars["va"]["pred_sl1"].items()}
    for key in ("copy_latcos", "mean_latcos", "shuf_latcos"):
        if key in scalars["va"]:
            g[key] = {k: [round(v, 3) for v in vv]
                      for k, vv in scalars["va"][key].items()}
    return g


def cka(x, y):
    x = torch.from_numpy(x.astype(np.float32)); y = torch.from_numpy(y.astype(np.float32))
    x = x - x.mean(0, keepdim=True); y = y - y.mean(0, keepdim=True)
    hsic = (x.T @ y).pow(2).sum()
    return float(hsic / ((x.T @ x).pow(2).sum().sqrt() * (y.T @ y).pow(2).sum().sqrt() + 1e-12))


# --------------------------------------------------------------------------- #
def main():
    global N_TRAIN, N_VAL, TRAIN_STEP, VAL_STEP, MLP_EPOCHS, LOGREG_ITERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="quick",
                    help='"quick", "all", or comma list of names (+"rand","raw")')
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny data + short fits, just to validate the harness")
    args = ap.parse_args()
    if args.smoke:
        N_TRAIN, N_VAL, TRAIN_STEP, VAL_STEP = 3000, 1200, 200, 400
        MLP_EPOCHS, LOGREG_ITERS = 4, 80
    names = (QUICK if args.models == "quick"
             else list(CKPTS) if args.models == "all"
             else args.models.split(","))
    names = list(dict.fromkeys(names + ["rand"]))

    OUT.mkdir(parents=True, exist_ok=True)
    t00 = time.time()
    win_tr, win_va = get_windows(args.refresh)
    targets_tr, targets_va = make_targets(win_tr), make_targets(win_va)
    rates = {k: round(float(v.mean()), 3) for k, v in targets_va.items() if v.shape[1] == 1
             and set(np.unique(v)) <= {0.0, 1.0}}
    print(f"targets ready; event base rates: {rates}")

    # raw-input control (battery over raw features, no encoder)
    raw_tr, raw_va = make_raw(win_tr), make_raw(win_va)
    raw_feats = {}
    for k, v in raw_tr.items():
        raw_feats[f"tr_{k}"] = v
    for k, v in raw_va.items():
        raw_feats[f"va_{k}"] = v

    all_results = {}
    val_pools_for_cka = {}
    for name in names:
        print(f"\n=== {name} ===")
        feats, scalars = get_pools(name, win_tr, win_va, args.refresh)
        res = {"model": name,
               "ckpt": str(CKPTS.get(name, "random-init")),
               "n_train": len(win_tr), "n_val": len(win_va),
               "geometry": geometry(feats, scalars),
               "probes": run_battery(feats, targets_tr, targets_va, name),
               "sample_eff": run_sample_eff(feats, targets_tr, targets_va)}
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
        all_results[name] = res
        val_pools_for_cka[name] = {k: feats[f"va_{k}"] for k in ("tgt_all", "tgt_s9cat")}
        print(f"  geometry: {json.dumps(res['geometry'], indent=None)[:400]}")

    # raw control battery (linear + mlp where the probe row asks for M)
    print("\n=== raw control ===")
    raw_rows = []
    for tier, name, pool, targ, kind, fams, status in PROBES:
        rk = raw_key(pool)
        Xtr, Xva = raw_feats[f"tr_{rk}"], raw_feats[f"va_{rk}"]
        Ytr, Yva = targets_tr[targ], targets_va[targ]
        row = {"tier": tier, "probe": name, "raw_feature": rk, "target": targ, "kind": kind}
        row["lin"] = round(FAM_FN["lin"](Xtr, Ytr, Xva, Yva, kind), 3)
        if "M" in fams:
            row["mlp"] = round(FAM_FN["mlp"](Xtr, Ytr, Xva, Yva, kind), 3)
        raw_rows.append(row)
        print(f"  [raw] {name:<42} lin={row['lin']} mlp={row.get('mlp')}")
    (OUT / "raw.json").write_text(json.dumps(
        {"model": "raw", "probes": raw_rows, "event_base_rates": rates}, indent=1))

    # cross-checkpoint CKA on identical val windows
    ck_names = list(val_pools_for_cka)
    cka_out = {}
    for key in ("tgt_all", "tgt_s9cat"):
        m = [[round(cka(val_pools_for_cka[a][key], val_pools_for_cka[b][key]), 3)
              for b in ck_names] for a in ck_names]
        cka_out[key] = {"names": ck_names, "matrix": m}
    (OUT / "cka.json").write_text(json.dumps(cka_out, indent=1))
    print(f"\nCKA(tgt_all):")
    for nm, row in zip(ck_names, cka_out["tgt_all"]["matrix"]):
        print(f"  {nm:<16}" + " ".join(f"{v:5.3f}" for v in row))
    print(f"\ndone in {(time.time()-t00)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
