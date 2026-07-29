"""demo_moment.py — one watchable moment: what the model predicts under each mask.

Picks a live 3.9s span from the VAL shard (shard_00312) containing a car->ball
touch in the middle of the prediction region, then runs a chosen checkpoint on
the 10-state window exactly like training (anchor visible at state 0, object
hidden states 1-9), decodes the predictor's latents to physical units, and
prints predicted vs actual per state.

Decoders (ridge, per object x per state) are fitted on TRAIN-shard (00310)
target-encoder latents -> build_obs features, so the demo replay is unseen.
"Ceiling" = the same decoder applied to the target encoder's latent of the
TRUE state: the best any perfect predictor could convey through this latent
code + decoder.

Run:  .venv/Scripts/python.exe probing/demo_moment.py [--ckpt step20k] [--rank N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite                                             # noqa: E402
from suite import (SHARDS, GAP, VISIBLE, EXT, OBJS, BALL_O, OPP_O, SELF_O,
                   OBJ_NAME, load_model, token_ids, get_windows, DEVICE)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RocketJEPA-2"))
from models.entity_encoding import build_obs, POS_DIV, POS_SCALE   # noqa: E402
from training.loader import (load_shard, build_physical_norm,       # noqa: E402
                             apply_physical_norm, live_play_mask)
from training.boost_pad_state import shard_pad_recharge             # noqa: E402

VAL_ZST = SHARDS / "shard_00312.zst"
SPAN = (EXT - 1) * GAP + 1          # 39 raw frames = 3.9 s
VEL_SCALE = 23000.0                  # entity_encoding relative-velocity scale
RAW_VEL_TO_UUS = 0.1                 # stored velocity is 10x uu/s


# --------------------------------------------------------------------------- #
def find_moments(arr, meta, names, top=5):
    """Live 39-frame spans with a car->ball touch mid-prediction-region.
    Returns list of (score, replay_dict, start_frame, touch_frame)."""
    col = {n: i for i, n in enumerate(names)}
    out = []
    for r in meta["replays"]:
        seg = arr[r["start"]: r["start"] + r["length"]].astype(np.float32)
        if len(seg) < SPAN + 2:
            continue
        live = live_play_mask(seg, names)
        ps = np.concatenate([[0], np.cumsum(~live)])
        bv = seg[:, [col["ball.vel_x"], col["ball.vel_y"], col["ball.vel_z"]]]
        dv = np.linalg.norm(np.diff(bv, axis=0), axis=1) * RAW_VEL_TO_UUS  # uu/s
        bp = seg[:, [col["ball.pos_x"], col["ball.pos_y"], col["ball.pos_z"]]]
        d_self = np.linalg.norm(bp - seg[:, [col["player.pos_x"], col["player.pos_y"], col["player.pos_z"]]], axis=1)
        d_opp = np.linalg.norm(bp - seg[:, [col["opponent.pos_x"], col["opponent.pos_y"], col["opponent.pos_z"]]], axis=1)
        near = np.minimum(d_self, d_opp)
        for st in range(0, len(seg) - SPAN, 5):
            if ps[st + SPAN] - ps[st] != 0:          # span must be fully live
                continue
            # touch inside states 3..7 of the prediction region (frames st+6..st+14)
            w = dv[st + 6: st + 14]
            k = int(np.argmax(w))
            f = st + 6 + k
            if w[k] < 500 or near[f] > 400:          # big impulse, car adjacent
                continue
            score = w[k] + 0.3 * bp[f, 2]            # prefer aerial-ish moments
            out.append((float(score), r, st, f))
    out.sort(key=lambda x: -x[0])
    return out[:top]


@torch.no_grad()
def fit_decoders(model, n_fit=12000):
    """Per (object, state) ridge decoders: target latent -> physical features.
    Returns {(obj, s): (W, b, mu, sd)} + val R^2 per object."""
    win_tr, win_va = get_windows()
    win_tr, win_va = win_tr[:n_fit], win_va[:4000]
    tgt = model.target_encoder
    feats = {}
    for split, win in (("tr", win_tr), ("va", win_va)):
        lat = []
        for i in range(0, len(win), 512):
            w = win[i:i + 512, :VISIBLE].to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                lat.append(tgt(tgt.pos(tgt.embedding.build(w))).float().cpu())
        feats[split] = torch.cat(lat)                          # [N, 50, 256]
        s, o, b, _, _ = build_obs(win[:, :VISIBLE])
        feats[split + "_y"] = {BALL_O: b[:, :, 0:6], OPP_O: torch.cat([o[:, :, 0:6], o[:, :, 15:16]], -1),
                               SELF_O: torch.cat([s[:, :, 3:6], s[:, :, 15:16]], -1)}
    dec, r2s = {}, {}
    for obj in (BALL_O, OPP_O, SELF_O):
        rs = []
        for st in range(1, VISIBLE):
            X = feats["tr"][:, st * OBJS + obj].to(DEVICE)
            Y = feats["tr_y"][obj][:, st].to(DEVICE)
            mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
            Xs = (X - mu) / sd
            ym = Y.mean(0, keepdim=True)
            G = Xs.T @ Xs + 0.1 * len(Xs) / 1000 * torch.eye(Xs.size(1), device=DEVICE)
            W = torch.linalg.solve(G, Xs.T @ (Y - ym))
            dec[(obj, st)] = (W, ym, mu, sd)
            Xv = (feats["va"][:, st * OBJS + obj].to(DEVICE) - mu) / sd
            Yv = feats["va_y"][obj][:, st].to(DEVICE)
            pred = Xv @ W + ym
            rs.append(float((1 - ((pred - Yv) ** 2).mean(0) / (Yv.var(0) + 1e-9)).mean()))
        r2s[obj] = rs
    return dec, r2s


def apply_dec(dec, obj, st, latent):
    W, ym, mu, sd = dec[(obj, st)]
    return ((latent.to(DEVICE) - mu) / sd) @ W + ym


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="step20k")
    ap.add_argument("--rank", type=int, default=0, help="which candidate moment (0=best)")
    args = ap.parse_args()

    arr, meta = load_shard(VAL_ZST)
    names = meta["feature_names"]
    cands = find_moments(arr, meta, names)
    print("candidate moments (score, replay, start-frame, touch-frame):")
    col = {n: i for i, n in enumerate(names)}
    for i, (sc, r, st, f) in enumerate(cands):
        secs = arr[r["start"] + st, col["env.seconds_remaining"]]
        print(f"  [{i}] score={sc:6.0f}  {r['id']}  {r['players']}  "
              f"frame {st} (touch @{f - st})  clock {int(secs) // 60}:{int(secs) % 60:02d}")
    sc, rep, st, tf = cands[args.rank]
    g = rep["start"]

    # normalized 110-col window, exactly like the loader
    scale, flag = build_physical_norm(names)
    pads = shard_pad_recharge(arr, meta)
    seg = apply_physical_norm(arr[g + st: g + st + SPAN].astype(np.float32), scale, flag)
    win = np.concatenate([seg, pads[g + st: g + st + SPAN]], axis=1)[::GAP]
    win = torch.from_numpy(win[:EXT]).float().unsqueeze(0)          # [1, 20, 110]
    vis = win[:, :VISIBLE]

    raw0 = arr[g + st].astype(np.float32)
    secs = raw0[col["env.seconds_remaining"]]
    print(f"\n=== MOMENT ===")
    print(f"shard_00312  replay {rep['id']}  players {rep['players']}  self={rep['players'][0]} ({rep['self_team']})")
    print(f"replay frame {st} (~{st / 10:.0f}s in)  in-game clock {int(secs) // 60}:{int(secs) % 60:02d}"
          f"{'  OT' if raw0[col['env.is_overtime']] > 0 else ''}")
    bp0 = raw0[[col["ball.pos_x"], col["ball.pos_y"], col["ball.pos_z"]]]
    print(f"ball world pos at t0: ({bp0[0]:.0f}, {bp0[1]:.0f}, {bp0[2]:.0f}) uu; "
          f"touch at t0+{(tf - st) / 10:.1f}s")
    print(f"window: 10 states, 0.2s apart (t0 .. t0+1.8s); model predicts states 1-9 of the masked object\n")

    model = load_model(args.ckpt)
    print(f"checkpoint: {args.ckpt}   fitting latent->physical decoders on shard_00310 ...")
    dec, r2s = fit_decoders(model)
    for o, nm in ((BALL_O, "ball(pos+vel)"), (OPP_O, "opp(pos+vel+boost)"), (SELF_O, "self(locvel+boost)")):
        print(f"  decoder val R^2 {nm:<20} s1..s9: " + " ".join(f"{v:.2f}" for v in r2s[o]))

    enc, prd, tgt = model.encoder, model.predictor, model.target_encoder
    truth = {}
    s_vec, o_vec, b_vec, _, _ = build_obs(vis)
    truth[BALL_O], truth[OPP_O] = b_vec[0, :, 0:6], torch.cat([o_vec[0, :, 0:6], o_vec[0, :, 15:16]], -1)
    truth[SELF_O] = torch.cat([s_vec[0, :, 3:6], s_vec[0, :, 15:16]], -1)

    # agent pose per state (for local->world conversion)
    fwd, right, up = vis[0, :, 18:21], vis[0, :, 21:24], vis[0, :, 24:27]
    rot = torch.stack([fwd, right, up], dim=-1)                    # local->world
    agent_pos_raw = vis[0, :, 9:12] * torch.tensor(POS_DIV)

    def to_world(rel, s):                                          # rel pos -> world uu
        return rot[s] @ (rel * POS_SCALE) + agent_pos_raw[s]

    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        w = vis.to(DEVICE)
        tok = enc.pos(enc.embedding.build(w))
        full_t = tgt(tgt.pos(tgt.embedding.build(w))).float()
        for obj in (BALL_O, OPP_O, SELF_O):
            nm = OBJ_NAME[obj]
            ids = torch.tensor(token_ids(obj), device=DEVICE)
            midx = torch.tensor([[obj + s * OBJS for s in range(1, VISIBLE)]], device=DEVICE)
            p = prd(enc(tok[:, ids]), midx).float()                # [1, 9, 256]
            print(f"\n--- mask = {nm.upper()} (hidden states 1-9; model saw {nm}@t0 + everything else) ---")
            if obj == BALL_O:
                hdr = f"{'t+s':>6} {'pred world pos (uu)':>24} {'true world pos':>20} {'err':>5} {'pred speed':>10} {'true':>5} {'ceil err':>8} {'latcos':>6}"
            elif obj == OPP_O:
                hdr = f"{'t+s':>6} {'pred world pos (uu)':>24} {'true world pos':>20} {'err':>5} {'pred boost':>10} {'true':>5} {'ceil err':>8} {'latcos':>6}"
            else:
                hdr = f"{'t+s':>6} {'pred local vel (uu/s)':>24} {'true local vel':>20} {'err':>5} {'pred boost':>10} {'true':>5} {'ceil err':>8} {'latcos':>6}"
            print(hdr)
            for k, s in enumerate(range(1, VISIBLE)):
                z_hat = p[:, k]
                z_tgt = full_t[:, s * OBJS + obj]
                latcos = float(F.cosine_similarity(z_hat.to(DEVICE), z_tgt.to(DEVICE)).mean())
                yh = apply_dec(dec, obj, s, z_hat)[0].float().cpu()
                yc = apply_dec(dec, obj, s, z_tgt)[0].float().cpu()
                yt = truth[obj][s]
                if obj in (BALL_O, OPP_O):
                    pw, cw, tw = to_world(yh[0:3], s), to_world(yc[0:3], s), to_world(yt[0:3], s)
                    err, cerr = float((pw - tw).norm()), float((cw - tw).norm())
                    if obj == BALL_O:
                        # relative vel -> world speed (rel to agent), report |v| uu/s
                        vh = (rot[s] @ (yh[3:6] * VEL_SCALE) + vis[0, s, 12:15] * VEL_SCALE) * RAW_VEL_TO_UUS
                        vt = (rot[s] @ (yt[3:6] * VEL_SCALE) + vis[0, s, 12:15] * VEL_SCALE) * RAW_VEL_TO_UUS
                        aux, auxt = f"{float(vh.norm()):.0f}", f"{float(vt.norm()):.0f}"
                    else:
                        aux, auxt = f"{float(yh[6]) * 100:.0f}%", f"{float(yt[6]) * 100:.0f}%"
                    print(f"+{s * 0.2:4.1f}s ({pw[0]:7.0f},{pw[1]:7.0f},{pw[2]:6.0f}) "
                          f"({tw[0]:6.0f},{tw[1]:7.0f},{tw[2]:5.0f}) {err:5.0f} {aux:>10} {auxt:>5} {cerr:8.0f} {latcos:6.3f}")
                else:
                    vh = yh[0:3] * VEL_SCALE * RAW_VEL_TO_UUS
                    vt = yt[0:3] * VEL_SCALE * RAW_VEL_TO_UUS
                    err = float((vh - vt).norm())
                    vc = yc[0:3] * VEL_SCALE * RAW_VEL_TO_UUS
                    cerr = float((vc - vt).norm())
                    print(f"+{s * 0.2:4.1f}s ({vh[0]:7.0f},{vh[1]:7.0f},{vh[2]:6.0f}) "
                          f"({vt[0]:6.0f},{vt[1]:7.0f},{vt[2]:5.0f}) {err:5.0f} "
                          f"{float(yh[3]) * 100:9.0f}% {float(yt[3]) * 100:4.0f}% {cerr:8.0f} {latcos:6.3f}")

    print("\nerr = distance predicted vs actual (uu; car ~118uu long, field 8192x10240)")
    print("ceil err = decoder applied to the TRUE state's target latent (decode floor)")
    print("latcos = cosine(predicted latent, target latent) at that slot")


if __name__ == "__main__":
    main()
