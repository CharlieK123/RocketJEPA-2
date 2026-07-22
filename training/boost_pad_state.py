"""boost_pad_state.py — reconstruct per-frame boost-pad availability from shards.

Rocket League replays do NOT store boost-pad state, but it's deterministic and
recoverable from car positions (already in the shards) + fixed game constants.
This is an OFFLINE pass over the existing shards (no re-decode).

Rule (the actual game rule): a pad is consumed whenever a car's path overlaps it
AND it is currently available, then it recharges for a fixed time. We therefore
trigger on POSITION (robust to full-boost pickups, which leave no boost delta),
interpolate the car path between the 10 fps samples (so fast fly-bys aren't
missed), and use the boost-amount delta only as an independent VALIDATOR.

Output per frame: pad_active[34] in {1.0 = available, 0.0 = recharging}. Optionally
also pad_recharge[34] in [0,1] = fraction of the respawn timer still remaining.

Run:  .venv-decode/bin/python boost_pad_state.py --shard data/shards/shard_00000.zst
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import zstandard as zstd

# --- authoritative pad table (RLBot wiki / RLGym common_values) ------------- #
# (x, y, is_big). z is ~70 (small) / ~73 (big); pads sit on the floor.
BOOST_PADS = np.array([
    [    0.0, -4240.0, 0], [-1792.0, -4184.0, 0], [ 1792.0, -4184.0, 0],
    [-3072.0, -4096.0, 1], [ 3072.0, -4096.0, 1], [ -940.0, -3308.0, 0],
    [  940.0, -3308.0, 0], [    0.0, -2816.0, 0], [-3584.0, -2484.0, 0],
    [ 3584.0, -2484.0, 0], [-1788.0, -2300.0, 0], [ 1788.0, -2300.0, 0],
    [-2048.0, -1036.0, 0], [    0.0, -1024.0, 0], [ 2048.0, -1036.0, 0],
    [-3584.0,     0.0, 1], [-1024.0,     0.0, 0], [ 1024.0,     0.0, 0],
    [ 3584.0,     0.0, 1], [-2048.0,  1036.0, 0], [    0.0,  1024.0, 0],
    [ 2048.0,  1036.0, 0], [-1788.0,  2300.0, 0], [ 1788.0,  2300.0, 0],
    [-3584.0,  2484.0, 0], [ 3584.0,  2484.0, 0], [    0.0,  2816.0, 0],
    [ -940.0,  3308.0, 0], [  940.0,  3308.0, 0], [-3072.0,  4096.0, 1],
    [ 3072.0,  4096.0, 1], [-1792.0,  4184.0, 0], [ 1792.0,  4184.0, 0],
    [    0.0,  4240.0, 0],
], dtype=np.float64)
PAD_XY   = BOOST_PADS[:, :2]                       # [34, 2]
IS_BIG   = BOOST_PADS[:, 2].astype(bool)           # [34]
PAD_R    = np.where(IS_BIG, 208.0, 144.0)          # pickup radius (uu)
RESPAWN  = np.where(IS_BIG, 10.0, 4.0)             # seconds
N_PADS   = len(BOOST_PADS)
# margin added to the pad pickup radius for the car hitbox. Swept empirically
# (2026-07-20): recall vs boost-delta plateaus at ~margin 20-30 (~92.6%), while
# false positives climb monotonically past it AND recall REVERSES beyond ~40
# (spurious pickups trip a pad's cooldown and mask real ones). 30 sits at the knee.
CAR_MARGIN_DEFAULT = 30.0
Z_MAX_DEFAULT = 200.0        # ignore aerial/wall frames (pads are on the floor)


def load_shard(zst_path):
    zst_path = Path(zst_path)
    meta = json.loads(zst_path.with_suffix(".json").read_text())
    raw = zstd.ZstdDecompressor().decompress(zst_path.read_bytes())
    arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
    return arr, meta


def _seg_pad_dist(xy):
    """Point-to-segment distance from every pad to each consecutive car segment.
    xy: [n, 2] car positions. Returns dist[n-1, 34] (segment t = xy[t]->xy[t+1])."""
    a = xy[:-1]                                    # [n-1, 2] segment start
    d = xy[1:] - a                                 # [n-1, 2] segment vector
    dd = (d * d).sum(1)                            # [n-1] |d|^2
    ap = PAD_XY[None] - a[:, None]                 # [n-1, 34, 2]
    t = (ap * d[:, None]).sum(2) / np.maximum(dd[:, None], 1e-9)   # [n-1, 34]
    t = np.clip(t, 0.0, 1.0)
    proj = a[:, None] + t[..., None] * d[:, None]  # [n-1, 34, 2] closest point on seg
    return np.linalg.norm(PAD_XY[None] - proj, axis=2)            # [n-1, 34]


def reconstruct_replay(seg, fnames, fps=10, car_margin=CAR_MARGIN_DEFAULT,
                       z_max=Z_MAX_DEFAULT, with_recharge=False):
    """seg: [n, feat] one replay's rows. Returns pad_active[n, 34] float32
    (1=available, 0=recharging); optionally also pad_recharge[n, 34] in [0,1]."""
    ix = {n: i for i, n in enumerate(fnames)}
    n = len(seg)
    dt = 1.0 / fps
    eff_r2 = (PAD_R + car_margin) ** 2             # squared effective radius per pad

    # both cars' xy + z (float32: fp16 positions are fine, but be safe)
    def car(prefix):
        return (np.stack([seg[:, ix[f"{prefix}.pos_x"]],
                          seg[:, ix[f"{prefix}.pos_y"]]], axis=1).astype(np.float64),
                seg[:, ix[f"{prefix}.pos_z"]].astype(np.float64))

    self_xy, self_z = car("player")
    opp_xy,  opp_z  = car("opponent")

    # per-segment overlap (path-interpolated) for each car, gated to on-ground frames
    overlap = np.zeros((n, N_PADS), dtype=bool)    # overlap[t] uses segment (t-1 -> t)
    for xy, z in ((self_xy, self_z), (opp_xy, opp_z)):
        dist = _seg_pad_dist(xy)                   # [n-1, 34]
        on_ground = (np.minimum(z[:-1], z[1:]) < z_max)[:, None]
        overlap[1:] |= (dist * dist < eff_r2) & on_ground

    # walk overlaps in time, enforcing respawn cooldown per pad -> pickup frames
    respawn_frames = np.ceil(RESPAWN * fps).astype(int)   # [34]
    available_at = np.zeros(N_PADS, dtype=int)     # earliest frame each pad is available
    active = np.ones((n, N_PADS), dtype=np.float32)
    recharge = np.zeros((n, N_PADS), dtype=np.float32) if with_recharge else None

    for p in range(N_PADS):
        rf = int(respawn_frames[p])
        hit_frames = np.nonzero(overlap[:, p])[0]
        picks = []
        for f in hit_frames:
            if f >= available_at[p]:               # pad was available -> collected
                picks.append(f)
                available_at[p] = f + rf
        for f in picks:
            end = min(n, f + rf)
            active[f:end, p] = 0.0
            if with_recharge:
                # remaining fraction: 1 at pickup -> 0 when it respawns
                k = np.arange(f, end)
                recharge[f:end, p] = 1.0 - (k - f) / rf
    return (active, recharge) if with_recharge else active


# names for the 34 appended env columns (recharge fraction: 0=available, ->1 just picked)
PAD_FEATURE_NAMES = [f"env.pad_recharge_{i:02d}" for i in range(N_PADS)]


def replay_pad_recharge(seg, fnames, fps=10, car_margin=CAR_MARGIN_DEFAULT,
                        z_max=Z_MAX_DEFAULT):
    """[n, 34] float32 recharge fraction per pad for ONE replay segment.
    0.0 = available; 1.0 = just picked up; linearly decays to 0 over the pad's
    respawn (4 s small / 10 s big). Accurate to ~1 frame (<=2.5%) where the pickup
    is detected (~93% recall vs the boost signal; see module docstring)."""
    _, recharge = reconstruct_replay(seg, fnames, fps=fps, car_margin=car_margin,
                                     z_max=z_max, with_recharge=True)
    return recharge


def shard_pad_recharge(arr, meta, car_margin=CAR_MARGIN_DEFAULT, z_max=Z_MAX_DEFAULT):
    """[total, 34] pad-recharge columns for a whole shard, reconstructed PER REPLAY
    (row ranges never cross a replay). Compute this from RAW positions (before any
    normalization)."""
    fnames = meta["feature_names"]
    fps = meta.get("fps", 10)
    a = arr.astype(np.float32)
    pads = np.zeros((len(a), N_PADS), dtype=np.float32)
    for r in meta["replays"]:
        lo, L = r["start"], r["start"] + r["length"]
        pads[lo:L] = replay_pad_recharge(a[lo:L], fnames, fps=fps,
                                         car_margin=car_margin, z_max=z_max)
    return pads


def expand_with_pad_recharge(arr, meta, car_margin=CAR_MARGIN_DEFAULT,
                             z_max=Z_MAX_DEFAULT):
    """Append the 34 pad-recharge columns to every frame of a whole shard array.
    Returns (arr_out[total, feat+34] float32, feature_names+34)."""
    pads = shard_pad_recharge(arr, meta, car_margin=car_margin, z_max=z_max)
    out = np.concatenate([arr.astype(np.float32), pads], axis=1)
    return out, list(meta["feature_names"]) + PAD_FEATURE_NAMES


def _boost_increase_events(seg, fnames, fps):
    """Frames where either car's boost jumped up during live play — an independent
    pickup signal to VALIDATE positional detection against. Excludes kickoff/reset
    (boost -> ~33) frames. Returns (n_events, self_evt_frames, opp_evt_frames)."""
    ix = {n: i for i, n in enumerate(fnames)}
    live = (seg[:, ix["env.ball_has_been_hit"]] >= 0.5) & (seg[:, ix["env.kickoff"]] < 0.5)
    evts = {}
    for prefix in ("player", "opponent"):
        b = seg[:, ix[f"{prefix}.boost"]].astype(np.float64)   # 0-255
        d = np.diff(b, prepend=b[:1])
        # a real pad delta is >= ~+30 (small pad = +12/100 = +30.6 in 0-255 units)
        up = (d > 15.0) & live
        evts[prefix] = np.nonzero(up)[0]
    return evts["player"], evts["opponent"]


def validate(seg, fnames, active, fps, tol=3):
    """How well do POSITIONAL pickups line up with BOOST-increase events?
    Reports recall = fraction of boost-up events that have a positional pickup
    within +/-tol frames (positional should catch every not-already-full pickup)."""
    self_evt, opp_evt = _boost_increase_events(seg, fnames, fps)
    # positional pickup frames = where any pad flips available->recharging
    drops = np.nonzero((active[:-1] > 0.5) & (active[1:] < 0.5))[0] + 1
    dropset = drops
    def matched(evts):
        if len(evts) == 0:
            return 0
        return sum(np.any(np.abs(dropset - e) <= tol) for e in evts)
    n_evt = len(self_evt) + len(opp_evt)
    n_match = matched(self_evt) + matched(opp_evt)
    return n_evt, n_match, len(drops)


def main():
    ap = argparse.ArgumentParser(description="Reconstruct boost-pad state from a shard")
    ap.add_argument("--shard", default=None, help="path to a shard_*.zst (default: first in data/shards)")
    ap.add_argument("--fps", type=int, default=None, help="override fps (default: from shard meta)")
    ap.add_argument("--car-margin", type=float, default=CAR_MARGIN_DEFAULT)
    ap.add_argument("--max-replays", type=int, default=25, help="replays to process for the report")
    args = ap.parse_args()

    shard = args.shard or sorted(glob.glob("data/shards/shard_*.zst"))[0]
    arr, meta = load_shard(shard)
    fnames = meta["feature_names"]
    fps = args.fps or meta.get("fps", 10)
    print(f"shard {Path(shard).name}  frames={arr.shape[0]:,}  fps={fps}  pads={N_PADS} "
          f"(big={IS_BIG.sum()}, small={(~IS_BIG).sum()})\n")

    tot_evt = tot_match = tot_drop = tot_frames = 0
    for r in meta["replays"][:args.max_replays]:
        seg = arr[r["start"]: r["start"] + r["length"]].astype(np.float32)
        active = reconstruct_replay(seg, fnames, fps=fps, car_margin=args.car_margin)
        n_evt, n_match, n_drop = validate(seg, fnames, active, fps)
        tot_evt += n_evt; tot_match += n_match; tot_drop += n_drop
        tot_frames += len(seg)

    rec = (tot_match / tot_evt) if tot_evt else float("nan")
    print(f"processed {args.max_replays} replays, {tot_frames:,} frames")
    print(f"positional pickups detected : {tot_drop:,}")
    print(f"boost-increase events (live): {tot_evt:,}")
    print(f"  matched by a positional pickup (+/-3 frames): {tot_match:,}"
          f"  -> recall {rec:.1%}")
    print("\n(recall well below 100% => radius/margin too small or fly-bys missed;")
    print(" note full-boost pickups have NO boost event, so they can't lower recall.)")

    # show mean availability per pad (sanity: big pads used less often -> lower)
    r0 = meta["replays"][0]
    seg0 = arr[r0["start"]: r0["start"] + r0["length"]].astype(np.float32)
    act0 = reconstruct_replay(seg0, fnames, fps=fps, car_margin=args.car_margin)
    frac = act0.mean(0)
    print(f"\nreplay 0: mean pad availability (1=always up), big pads marked *")
    for p in range(N_PADS):
        tag = "*BIG" if IS_BIG[p] else ""
        print(f"  pad {p:2d} ({PAD_XY[p,0]:7.0f},{PAD_XY[p,1]:7.0f}) avail {frac[p]:.2f} {tag}")


if __name__ == "__main__":
    main()