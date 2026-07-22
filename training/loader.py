"""loader.py — read decoded zstd shards for training (runs in the training .venv).

Each shard row is one timestep: at index t you get state_t and the action taken
at t, aligned. `state` = all non-action columns (ball + player physical state +
opponent + env); `action` = the 8 `player.act.*` columns (throttle, steer, pitch,
yaw, roll, jump, boost, handbrake). Pairs/windows are NOT stored on disk — form
them here from the per-replay sequence (row ranges never cross replay boundaries).

Shard format (written by decode_replays.py):
  shard_NNNNN.zst   zstd-compressed float16 bytes, reshape to [total_frames, feat_dim]
  shard_NNNNN.json  {dtype, shape, fps, obj_lengths, feat_dim, feature_names, replays[]}
                    replays[i] = {id, start, length, players, self_team}
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd
from torch.utils.data import IterableDataset, get_worker_info

ACTION_PREFIX = "player.act."


def load_shard(zst_path):
    """Return (arr[frames, feat_dim] float array, meta dict)."""
    zst_path = Path(zst_path)
    meta = json.loads(zst_path.with_suffix(".json").read_text())
    raw = zstd.ZstdDecompressor().decompress(zst_path.read_bytes())
    arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
    return arr, meta


def split_indices(feature_names):
    """(state_idx, action_idx) column indices. action = player.act.* ; state = rest."""
    action_idx = [i for i, n in enumerate(feature_names) if n.startswith(ACTION_PREFIX)]
    aset = set(action_idx)
    state_idx = [i for i in range(len(feature_names)) if i not in aset]
    return state_idx, action_idx


def iter_replays(shards_dir):
    """Yield (replay_id, state[n, state_dim] float32, action[n, action_dim] float32)
    for every replay across all shards. state[t]/action[t] are aligned at timestep t."""
    for zst in sorted(glob.glob(str(Path(shards_dir) / "shard_*.zst"))):
        arr, meta = load_shard(zst)
        s_idx, a_idx = split_indices(meta["feature_names"])
        for r in meta["replays"]:
            seg = arr[r["start"]: r["start"] + r["length"]]
            yield r["id"], seg[:, s_idx].astype(np.float32), seg[:, a_idx].astype(np.float32)


class TimestepDataset(IterableDataset):
    """Streams (state_t, action_t) timesteps across all shards.

    IterableDataset (not map-style) because the full corpus is billions of
    timesteps — building a global row index would blow up memory. Shards are
    decompressed one at a time; set shuffle=True to shuffle shard order and rows
    within each shard (buffered), which is enough decorrelation for SGD.
    """

    def __init__(self, shards_dir, shuffle=True, seed=0):
        self.files = sorted(glob.glob(str(Path(shards_dir) / "shard_*.zst")))
        if not self.files:
            raise FileNotFoundError(f"no shard_*.zst in {shards_dir}")
        self.shuffle = shuffle
        self.seed = seed
        # dims from the first shard's schema
        _, meta = load_shard(self.files[0])
        self.state_idx, self.action_idx = split_indices(meta["feature_names"])
        self.state_dim = len(self.state_idx)
        self.action_dim = len(self.action_idx)
        self.feature_names = meta["feature_names"]

    def __iter__(self):
        info = get_worker_info()
        files = self.files
        if info is not None:  # shard the shard-list across DataLoader workers
            files = files[info.id:: info.num_workers]
        rng = np.random.default_rng(self.seed + (info.id if info else 0))
        order = rng.permutation(len(files)) if self.shuffle else range(len(files))
        for fi in order:
            arr, meta = load_shard(files[fi])
            states = arr[:, self.state_idx].astype(np.float32)
            actions = arr[:, self.action_idx].astype(np.float32)
            rows = rng.permutation(len(arr)) if self.shuffle else range(len(arr))
            for t in rows:
                yield torch.from_numpy(states[t]), torch.from_numpy(actions[t])


def build_loader(shards_dir, batch_size=256, shuffle=True, num_workers=0, seed=0):
    """DataLoader yielding (state[B, state_dim], action[B, action_dim]) batches."""
    ds = TimestepDataset(shards_dir, shuffle=shuffle, seed=seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers), ds


# --------------------------------------------------------------------------- #
# Normalization (empirical z-score) — makes the weird raw scales irrelevant
# --------------------------------------------------------------------------- #
def compute_norm_stats(shards_dir, max_frames=1_000_000, save=True):
    """Streaming per-feature mean/std over up to `max_frames` frames. Constant
    columns get std=1 (no scaling). Saved to shards/norm_stats.npz if save=True."""
    files = sorted(glob.glob(str(Path(shards_dir) / "shard_*.zst")))
    if not files:
        raise FileNotFoundError(f"no shard_*.zst in {shards_dir}")
    s = ss = None
    n = 0
    names = None
    for f in files:
        arr, meta = load_shard(f)
        a = arr.astype(np.float64)
        names = meta["feature_names"]
        s = a.sum(0) if s is None else s + a.sum(0)
        ss = (a * a).sum(0) if ss is None else ss + (a * a).sum(0)
        n += len(a)
        if n >= max_frames:
            break
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12))
    std = np.where(std < 1e-6, 1.0, std)          # constant cols -> leave as-is
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    if save:
        np.savez(Path(shards_dir) / "norm_stats.npz",
                 mean=mean, std=std, feature_names=np.array(names))
    return mean, std


def _load_norm(shards_dir, normalize):
    """Resolve the `normalize` arg -> (mean, std) or None."""
    if normalize is False or normalize is None:
        return None
    if isinstance(normalize, (tuple, list)):
        return np.asarray(normalize[0], np.float32), np.asarray(normalize[1], np.float32)
    p = Path(shards_dir) / "norm_stats.npz"
    if not p.exists():
        raise FileNotFoundError("normalize=True but norm_stats.npz missing — "
                                "run compute_norm_stats(shards_dir) first")
    d = np.load(p, allow_pickle=True)
    return d["mean"], d["std"]


# --------------------------------------------------------------------------- #
# Physical-constant normalization — divide each feature by its KNOWN game bound
# so inputs land in ~[-1, 1] (or [0, 1]), then clip. Deterministic (no dataset
# stats to persist), train/deploy-consistent, and matches RLGym-style obs — the
# right default when you'll fine-tune in an RL env later.
#
# Raw-unit reminders (see the encoding audit): velocity is x10 uu/s, angular
# velocity is x1000 rad/s, boost is a 0-255 byte, rotations are euler radians,
# and *_active are 0-12 STATE CODES (thresholded to a bool, not scaled).
# --------------------------------------------------------------------------- #
_PI = float(np.pi)


def _phys_divisor(name):
    """(divisor, is_flag) for one feature. is_flag -> threshold >0 to {0,1}
    instead of scaling. divisor 1.0 -> leave as-is (already normalized)."""
    if ".act." in name:                      # controller inputs already in [-1,1]/{0,1}
        return 1.0, False
    if name.endswith(("jump_active", "dodge_active", "double_jump_active")):
        return 1.0, True                     # 0-12 state code -> boolean
    if name.startswith("ball."):
        if "ang_vel" in name: return 6000.0, False       # 6 rad/s x1000
        if "vel" in name:     return 60000.0, False      # 6000 uu/s x10
        if name.endswith("pos_x"): return 4096.0, False
        if name.endswith("pos_y"): return 5120.0, False
        if name.endswith("pos_z"): return 2044.0, False
        if "rot" in name: return _PI, False
    if name.startswith("player.") or name.startswith("opponent."):
        if "ang_vel" in name: return 5500.0, False       # 5.5 rad/s x1000
        if "vel" in name:     return 23000.0, False      # 2300 uu/s x10
        if name.endswith("pos_x"): return 4096.0, False
        if name.endswith("pos_y"): return 5120.0, False
        if name.endswith("pos_z"): return 2044.0, False
        if "rot" in name: return _PI, False
        if name.endswith(".boost"): return 255.0, False
    if name.startswith("env."):
        if name.endswith("seconds_remaining"): return 300.0, False
        if name.endswith(("blue_score", "orange_score", "score_diff")): return 10.0, False
        # is_overtime / ball_has_been_hit / kickoff / pad_recharge_* -> already 0-1
    return 1.0, False


def build_physical_norm(feature_names):
    """-> (scale[F] float32 multiplier = 1/bound, flag_mask[F] bool)."""
    scale = np.ones(len(feature_names), np.float32)
    flag = np.zeros(len(feature_names), dtype=bool)
    for i, n in enumerate(feature_names):
        div, is_flag = _phys_divisor(n)
        scale[i] = 1.0 / div
        flag[i] = is_flag
    return scale, flag


def apply_physical_norm(a, scale, flag):
    """a[N,F] raw -> physically normalized, clipped to [-1,1]. Flag columns become
    (x>0) booleans; already-[0,1] columns (booleans, pads) pass through unclipped."""
    out = a * scale
    if flag.any():
        out[:, flag] = (a[:, flag] > 0.0).astype(np.float32)
    np.clip(out, -1.0, 1.0, out=out)
    return out


# --------------------------------------------------------------------------- #
# Symmetric (physics-only) layout + team-perspective mirror (RLGym-sim invert).
#
# For 1v1 self-play the two teams are the same policy viewed from opposite ends,
# so RLGym canonicalizes orange into blue's frame. Their invert (confirmed from
# rlgym_sim DefaultObs: inverted_car_data / inverted_ball / inverted_boost_pads)
# is a 180 deg rotation about z: every vector * [-1,-1,1] (flip x & y -> yaw+=pi),
# swap the two cars (orange becomes "self"), and reverse the 34 boost pads (the
# pad list is antipodal, so reverse == the (x,y)->(-x,-y) permutation). It's a
# proper rotation (no handedness flip), so roll and steer keep their sign.
#
# This needs the two cars to be the SAME schema, so we also drop the 8 self-only
# action columns -> both cars are identical 16-dim physical objects (a shared
# per-car encoder + PPO-transferable obs). Actions stay in the shards; they just
# don't feed this encoder. Enable via symmetric=True; mirror=True additionally
# emits the inverted view (2x data).
# --------------------------------------------------------------------------- #
def symmetric_keep(feature_names):
    """(keep_idx, kept_names): OPTIONAL physics-only mode — drop BOTH cars' 8 action
    cols so self & opponent are 16-dim physical objects. Not needed now that shards
    store both cars' actions (both cars are already symmetric 24-dim); kept for a
    physics-only variant. ball(12)+self(16)+opp(16)+env(7) = 51."""
    keep = [i for i, n in enumerate(feature_names)
            if not (n.startswith("player.act.") or n.startswith("opponent.act."))]
    return keep, [feature_names[i] for i in keep]


def build_invert_plan(feature_names):
    """Precompute the team invert on a symmetric frame -> (neg[F], yaw_cols, perm[F]).
    neg: *-1 on x/y vector components and score_diff. yaw_cols: rot_y (+pi, wrapped).
    perm: swap player<->opponent columns and blue<->orange score."""
    F = len(feature_names)
    neg = np.ones(F, np.float32)
    # 180deg-about-z invert: every WORLD vector's x,y components flip. That's exactly
    # the "_x"/"_y" columns of pos/vel/ang_vel and the forward/right/up orientation
    # vectors (z stays). No non-vector feature ends in _x/_y (actions, flags, env,
    # pads don't), so this selects precisely the components to negate. score_diff too.
    yaw_cols = []                                   # orientation is fwd/right/up now, no euler yaw
    idx = {n: i for i, n in enumerate(feature_names)}
    for i, n in enumerate(feature_names):
        base = n.rsplit(".", 1)[-1]
        if base.endswith("_x") or base.endswith("_y") or base == "score_diff":
            neg[i] = -1.0
    perm = list(range(F))
    for i, n in enumerate(feature_names):           # swap the two cars (incl their
        if n.startswith("player."):                 # actions — invariant under the
            j = idx.get("opponent." + n.split(".", 1)[1])   # rotation, just travel
            if j is not None:                        # with their car)
                perm[i], perm[j] = j, i
    if "env.blue_score" in idx and "env.orange_score" in idx:
        b, o = idx["env.blue_score"], idx["env.orange_score"]
        perm[b], perm[o] = o, b
    return neg, np.array(yaw_cols, dtype=int), np.array(perm, dtype=int)


def apply_invert(a, plan):
    """Invert a symmetric physics-only frame [N,F] (RAW units) to the opposite team
    perspective. Apply BEFORE normalization."""
    neg, yaw_cols, perm = plan
    out = a * neg
    if len(yaw_cols):
        out[:, yaw_cols] = (out[:, yaw_cols] + np.pi + np.pi) % (2 * np.pi) - np.pi
    return out[:, perm]


# --------------------------------------------------------------------------- #
# Windowed dataset — for the masked-history JEPA. Experiment with the horizon
# via `window` (#frames) and `gap` (frame spacing; window spans (window-1)*gap+1
# real frames, so gap dilates the temporal horizon without more tokens).
# --------------------------------------------------------------------------- #
def live_play_mask(seg, feature_names, resume="go", freeze_speed=1000.0):
    """Per-frame boolean over ONE replay's rows: True = live play, False = noise.

    Drops the two dead-time noise sources that carry no useful dynamics:
      * the post-goal explosion / cars-flung / celebration + reset window (from
        the frame a goal is scored, detected as a `blue_score`/`orange_score`
        increment);
      * the frozen kickoff countdown, where the game forces the cars stationary.

    `resume` controls where each dead span ENDS (where live play resumes):
      * "go"  (default) — resume the frame the car is unfrozen and free to move,
        i.e. right after the last frozen frame in the span. The kickoff
        drive-to-ball is KEPT. "Frozen" = self-car speed < `freeze_speed` (raw
        stored units; frozen countdown reads ~0-72, driving jumps to >2400, so
        the 1000 default separates them cleanly).
      * "first_touch" — resume only when the ball is next hit
        (`ball_has_been_hit` 0->1). The whole pre-touch kickoff is dropped too.

    Must be called per replay: scores are cumulative within a replay, so diffs
    across a replay boundary would be meaningless.
    """
    def col(name):
        return seg[:, feature_names.index(name)]

    bhh = col("env.ball_has_been_hit") >= 0.5
    total_score = col("env.blue_score") + col("env.orange_score")
    dead = ~bhh                                  # freeze + pre-first-touch
    goal_frames = np.where(np.diff(total_score) > 0)[0] + 1
    for g in goal_frames:                        # goals are few per game
        j = g
        while j < len(seg) and bhh[j]:           # explosion: goal -> reset
            dead[j] = True
            j += 1

    if resume == "go":
        # keep the drive-to-ball: within each dead span, un-drop everything
        # after the last frozen frame (the countdown ends -> car may move).
        vel = np.stack([col("player.vel_x"), col("player.vel_y"),
                        col("player.vel_z")], axis=1).astype(np.float32)
        speed = np.sqrt((vel ** 2).sum(axis=1))   # float32: fp16 vels overflow squared
        frozen = speed < freeze_speed
        d = dead.astype(np.int8)
        edges = np.where(np.diff(np.concatenate([[0], d, [0]])) != 0)[0]
        for k in range(0, len(edges), 2):        # each dead run [a, b)
            a, b = edges[k], edges[k + 1]
            fz = np.where(frozen[a:b])[0]
            if len(fz):                          # resume after last frozen frame
                dead[a + fz[-1] + 1: b] = False

    # demos: a demolished car's position comes through as NaN -> nan_to_num -> (0,0,0)
    # for the ~3s it's dead; real grounded cars sit at z~=17, never all-zero. Mark
    # those frames dead (applied AFTER the resume un-drop so demos stay dead) — this
    # removes both the origin-parked frames and, via the span check, the respawn
    # teleport at the demo's edges. Catches brief NaN glitches too.
    for who in ("player", "opponent"):
        dead |= ((col(f"{who}.pos_x") == 0) & (col(f"{who}.pos_y") == 0)
                 & (col(f"{who}.pos_z") == 0))
    return ~dead


class WindowDataset(IterableDataset):
    """Yields full-frame windows [window, feat_dim] (float32), never crossing a
    replay boundary. `gap` spaces frames within a window; `step` strides between
    successive windows. `normalize`: True (load shards/norm_stats.npz), a
    (mean,std) tuple, or False.

    `drop_noise=True` (default) discards windows whose time-span overlaps a
    dead region (kickoff freeze / post-goal explosion) via `live_play_mask`. The
    check is on the whole window SPAN, not just its sampled frames, so a window
    can never straddle a goal and contain a discontinuity — critical for a
    dynamics model. Set False to keep every window (old behaviour). `resume`
    ("go"/"first_touch") is forwarded to `live_play_mask` — "go" keeps the
    kickoff drive-to-ball, "first_touch" drops the whole pre-touch kickoff."""

    def __init__(self, shards_dir, window=5, gap=1, step=1,
                 normalize=False, shuffle=True, seed=0, drop_noise=True,
                 resume="go", pad_state=False, symmetric=False, mirror=False):
        self.files = sorted(glob.glob(str(Path(shards_dir) / "shard_*.zst")))
        if not self.files:
            raise FileNotFoundError(f"no shard_*.zst in {shards_dir}")
        self.window, self.gap, self.step = window, gap, step
        self.shuffle, self.seed = shuffle, seed
        self.drop_noise = drop_noise
        self.resume = resume
        # pad_state: append the 34 boost-pad recharge cols (env 7 -> 41).
        self.pad_state = pad_state
        # Shards now store BOTH cars' actions, so self & opponent are already
        # symmetric 24-dim objects and mirror works directly (no dropping needed).
        # symmetric=True is an OPTIONAL physics-only mode that drops both cars'
        # actions (67 -> 51). mirror=True emits the team-inverted view (2x data).
        self.symmetric = symmetric
        self.mirror = mirror
        _, meta = load_shard(self.files[0])
        base_names = meta["feature_names"]                       # 59
        if self.symmetric:
            self.keep_idx, base_names = symmetric_keep(base_names)   # -> 51
            self.keep_idx = np.array(self.keep_idx)
        else:
            self.keep_idx = None
        self.feature_names = list(base_names)
        self.feat_dim = len(base_names)
        if pad_state:
            from boost_pad_state import PAD_FEATURE_NAMES
            self.feature_names = self.feature_names + PAD_FEATURE_NAMES
            self.feat_dim += len(PAD_FEATURE_NAMES)
        # obj_lengths derived from the ACTUAL feature set (ball / player / opponent /
        # env), so the model auto-matches whatever schema the shards carry — old
        # 59-dim (12,24,16,7[+34]) or new (9,30,30,7[+34]) — no hardcoding.
        self.obj_lengths = tuple(
            sum(1 for n in self.feature_names if n.startswith(p))
            for p in ("ball.", "player.", "opponent.", "env."))
        # normalization is built on the (possibly symmetric) base feature set;
        # pad/boolean cols already 0-1 pass through. "physical" -> fixed game-bound
        # scaling; True/(mean,std) -> z-score via norm_stats.npz; False -> raw.
        self.physical = isinstance(normalize, str) and normalize == "physical"
        if self.physical:
            self.phys_scale, self.phys_flag = build_physical_norm(base_names)
            self.mean = self.std = None
        else:
            norm = _load_norm(shards_dir, normalize)
            self.mean, self.std = (norm if norm is not None else (None, None))
        # team-invert plan (built on the symmetric base) for the mirror view
        self.invert_plan = build_invert_plan(base_names) if self.mirror else None

    def __iter__(self):
        info = get_worker_info()
        files = self.files[info.id:: info.num_workers] if info else self.files
        rng = np.random.default_rng(self.seed + (info.id if info else 0))
        order = rng.permutation(len(files)) if self.shuffle else range(len(files))
        span = (self.window - 1) * self.gap + 1
        offs = np.arange(self.window) * self.gap

        def finalize(x, pads):                        # normalize base + append pads
            if self.physical:
                x = apply_physical_norm(x, self.phys_scale, self.phys_flag)
            elif self.mean is not None:
                x = (x - self.mean) / self.std
            if pads is not None:
                x = np.concatenate([x, pads], axis=1)
            return x

        for fi in order:
            arr, meta = load_shard(files[fi])
            # live mask on the RAW full frame (uses hit flag, scores, player vel)
            base_names = meta["feature_names"]
            live = None
            if self.drop_noise:
                live = np.ones(len(arr), dtype=bool)
                for r in meta["replays"]:
                    lo, L = r["start"], r["length"]
                    live[lo:lo + L] = live_play_mask(arr[lo:lo + L], base_names,
                                                     resume=self.resume)
            # boost-pad recharge fractions from RAW positions (per replay)
            pads = None
            if self.pad_state:
                from boost_pad_state import shard_pad_recharge
                pads = shard_pad_recharge(arr, meta)          # [total, 34] raw [0,1]
            # symmetric physics-only base (drop actions), RAW
            a = arr.astype(np.float32)
            if self.keep_idx is not None:
                a = a[:, self.keep_idx]
            # native view + optional team-inverted view (both normalized)
            variants = [finalize(a, pads)]
            if self.mirror:
                a_inv = apply_invert(a, self.invert_plan)     # RAW invert -> opp team
                pads_inv = pads[:, ::-1].copy() if pads is not None else None
                variants.append(finalize(a_inv, pads_inv))
            # window starts (identical for every variant — same temporal structure)
            dead_ps = (np.concatenate([[0], np.cumsum(~live)])
                       if live is not None else None)
            starts = []
            for r in meta["replays"]:               # windows stay within a replay
                lo, L = r["start"], r["length"]
                if L < span:
                    continue
                cand = lo + np.arange(0, L - span + 1, self.step)
                if dead_ps is not None:
                    # reject any window whose SPAN [st, st+span) touches dead time
                    keep = (dead_ps[cand + span] - dead_ps[cand]) == 0
                    cand = cand[keep]
                starts.extend(cand.tolist())
            if self.shuffle:
                rng.shuffle(starts)
            for st in starts:
                for v in variants:                  # native (+ mirror) -> 2x windows
                    yield torch.from_numpy(v[st + offs])   # [window, feat_dim]


def build_window_loader(shards_dir, window=5, gap=1, step=1, batch_size=64,
                        normalize=False, num_workers=0, shuffle=True, seed=0,
                        drop_noise=True, resume="go", pad_state=False,
                        symmetric=False, mirror=False):
    """DataLoader yielding windows [B, window, feat_dim] for the masked-history model.
    `drop_noise=True` filters kickoff-freeze/post-goal windows; `resume` sets where
    each dead span ends. `pad_state=True` appends 34 boost-pad recharge cols.
    `normalize`: "physical" (fixed game-bound scaling -> [-1,1]), True/(mean,std)
    (z-score via norm_stats.npz), or False (raw).
    `symmetric=True` drops the 8 action cols so self/opponent are identical 16-dim
    physical objects (frame 59->51). `mirror=True` (implies symmetric) also emits the
    RLGym-style team-inverted view -> 2x data, all in one canonical frame."""
    ds = WindowDataset(shards_dir, window, gap, step, normalize, shuffle, seed,
                       drop_noise, resume, pad_state, symmetric, mirror)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers), ds


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sanity-check the shard loader")
    p.add_argument("--shards", default="shards")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    dl, ds = build_loader(args.shards, batch_size=args.batch_size)
    print(f"state_dim={ds.state_dim} action_dim={ds.action_dim} shards={len(ds.files)}")
    s, a = next(iter(dl))
    print("batch state:", tuple(s.shape), "action:", tuple(a.shape))
