# R-JEPA 2 — Handoff Context Pack

**Purpose of this document:** This is a handoff from a Claude Code session (2026-07-23) in which the R-JEPA 2 codebase was fully audited, smoke-tested end-to-end, and cross-checked against Meta's official V-JEPA implementation (facebookresearch/jepa). It contains everything needed to continue the discussion in a fresh chat: project summary, the audit findings, the current discussion focus (**masking distribution design**), and the complete source code of all 8 files. Read this first; the source appendix is ground truth.

---

## 1. What the project is

R-JEPA 2 is a JEPA (Joint-Embedding Predictive Architecture, I-JEPA/V-JEPA style) pretraining system for **Rocket League 1v1 state sequences** (not video — engineered state vectors from replays).

- **Data**: zstd shards of decoded replays, 10 fps, float16, on a Windows training box at `C:\Users\charl\R-JEPA2\data\shards_150k`. Each frame = 76 base features (ball 9, player 22+8 actions, opponent 22+8 actions, env 7) + 34 boost-pad recharge columns reconstructed offline → 110 total.
- **Windowing**: 10 consecutive states per sample (1 second), never crossing replay boundaries; dead time (kickoff freeze, post-goal explosions, demos) filtered out (`drop_noise=True`).
- **Tokenization**: each state → 5 object tokens via per-object MLPs: `(self 19, opp 19, ball 9, env 7, boost 170)` feature dims → 256-d latents. 10 states × 5 objects = 50 tokens, with sinusoidal state+object positional encoding.
- **Masking**: per sample, ONE object is chosen (multinomial over `mask_probs`) and hidden in states 1–9; state 0 stays visible as an anchor. 41 visible tokens → context encoder (5 blocks, pre-norm, RMSNorm, MHA, FFN 1024). Predictor (2 blocks, FFN 128) gets context latents + 9 mask queries (`mask_token + pos[masked_idx]`), returns the 9 predictions.
- **Targets**: EMA target encoder (momentum 0.995) encodes the full 50 tokens; target latents are layer-normalized; smooth-L1 loss on the 9 masked positions only.
- **Training**: AdamW lr 1e-4, wd 1e-5, batch 2048, bf16 autocast, grad clip 1.0, 100 epochs, EMA update after each optimizer step.
- **Important feature-frame fact**: all car/ball/pad features are **agent-centric** (relative to the "self" car and rotated into its frame); the self token carries absolute position; boost-pad tokens carry relative positions to 34 pads at known constant world locations.

**Audit verdict:** operational. A synthetic-shard smoke test ran the real loader + real `train()` end-to-end (loss 0.97→0.06 over 26 steps); gradient isolation, EMA direction/order, mask/target index alignment all verified correct. Core mechanics match V-JEPA semantics (stop-grad target, LN'd targets, mask-query predictor, loss on masked positions only, per-sample masks, bf16).

---

## 2. CURRENT DISCUSSION FOCUS: masking distribution design

This is the topic being "hammered down on." State of the discussion:

### The bug that started it
`main.py` passes `mask_probs=torch.tensor([0.35, 0.4, 0.05, 0.2])` — only 4 entries for 5 objects. Token order is fixed by `build_obs`'s return order: **(self, opp, ball, env, boost)**. So the actual allocation is: self 35%, opp 40%, **ball 5%**, **env 20%**, boost 0% (unreachable). Almost certainly a transposition accident (the loader's canonical order is ball-first; `build_obs`'s is self-first). Env prediction is near-trivial: `drop_noise` guarantees no goals inside windows, so all env features are constant except `seconds_remaining`, which decreases linearly — and the visible state-0 anchor gives everything away. ~20% of updates learn almost nothing; the ball — the richest signal — gets 5%.

### The analysis framework (established in discussion)
Masking object X trains two things: (1) the predictor's model of X's dynamics, and (2) — the important one — the **encoder's representations of all VISIBLE objects**, which are forced to carry X-relevant information. Masking the ball doesn't primarily train the ball encoder (its tokens are hidden); it trains the cars' embeddings to explain the ball. An object earns masking probability only if its future is **genuinely hidden** given visible context and **informative** when recovered.

### Object-by-object findings
- **Ball**: genuinely hidden (no other token references it), physics-rich (bounces, touch outcomes), and its agent-relative targets also exercise ego-motion. Highest value.
- **Opponent**: genuinely hidden (only leak: boost pads flipping to recharging betray opponent visits sparsely). Controller inputs unobserved → predicting it = learning an intent/policy prior ("where does an opponent go in the next second given ball + self"). Second-highest value; the anticipation representation is what transfers to downstream RL (PPO).
- **Self**: **mostly NOT hidden — this is the key structural insight.** The agent-centric frame leaks the masked agent back through every visible relative feature: each of the 34 boost-pad tokens carries `R_agentᵀ(pad − agent)` to a known landmark — trilateration; 3 pads pin down full pose in closed form, every frame. `rel_ball_pos` and `rel_opp_position` leak it again. Masking self therefore trains an easy deterministic coordinate-frame inversion, not dynamics. Only genuinely hidden bits: own boost amount + jump/dodge flags. Deserves a small share, not the current 35%.
- **Env**: deterministic linear clock. Worth ~0, except as a calibration canary (its loss must pin to ~0 quickly; if not, something upstream broke).
- **Boost pads**: deterministic given the visible cars' paths ("did a car cross pad i") — solvable, so the loss floors, but en route it supervises precise car-path geometry. Worth a token amount.

### Current recommendation on the table
| object | P(masked) | rationale |
|---|---|---|
| ball  | 0.45 | only truly-hidden physics-rich object; core downstream skill |
| opp   | 0.35 | only truly-hidden agent; buys anticipation/intent modeling |
| self  | 0.10 | pose leaks via agent-centric frame; only boost/flags hidden |
| env   | 0.05 | near-zero content; kept as calibration canary |
| boost | 0.05 | deterministic but forces path-geometry reasoning |

Load-bearing decisions: ball+opp ≈ 80%, self ≈ 10%, env+boost ≈ 10%; ±5% within that is noise. Caveats: if features are ever de-agent-centered, or action-conditioning is added (controller inputs visible to the predictor), self becomes a real dynamics task and deserves 25–30%.

### Open threads to continue exploring
1. **Verification plan**: log per-object loss (bucket smooth-L1 by `masked_idx[:,0] % 5`). Expected: env →~0 fast (canary), self falls fast/low (confirms leak), boost floors slowly, ball/opp keep a persistent floor (unobserved inputs). If self's loss stays high, the leak analysis is wrong — raise its share. The metric arbitrates.
2. **Joint masking**: mask ball+opp together some fraction (~15%?) to kill residual cross-leaks and force the hardest inference — closest analog to V-JEPA's multi-block masking in this object world. How to implement cleanly (mask 2 objects → 18 queries) and what fraction?
3. **Curriculum**: fixed distribution vs shifting mass toward ball/opp over training?
4. **Anchor design**: state 0 visible (current, = conditional forecasting) vs V-JEPA-style full-tube masking (no anchor)? Currently judged an intentional, defensible divergence.
5. **Feature-frame redesign**: de-agent-centering or adding a world-frame view would change the whole leak calculus — bigger surgery, interacts with findings A3/A4 below.

---

## 3. Full audit findings (ranked worst → least)

1. **Mask probabilities mis-assigned** (`main.py:37`) — see §2. Fix: 5-entry tensor in `(self, opp, ball, env, boost)` order + `assert len(mask_probs) == len(obj_lengths)` in `JEPA.__init__` + a shared `OBJECT_ORDER` constant as single source of truth.
2. **EMA momentum schedule silently ignored** (`jepa.py:45`) — `momentum=(0.995, 1.0, 10_000)` is passed but only `momentum[0]` is read; EMA stays 0.995 forever. V-JEPA anneals 0.998→1.0 linearly over the run. Fix: store (start, final, steps), interpolate per step in `update_target_params`.
3. **Silent schema dependency** — `entity_encoding.build_obs` hardcodes raw column indices (ball 0:9, player 9:31, acts 31:39, opp 39:61, acts 61:69, env 69:76, pads 76:110) and ignores `feature_names`. Internally consistent (proven by smoke test), but MUST be verified against the real shards' `feature_names` on the Windows box; also check no orientation column name contains "rot" (the loader's `_phys_divisor` would divide unit vectors by π).
4. **Run-loss risk**: checkpoint only saved on KeyboardInterrupt (crash/OOM/power = total loss; no resume path), and `main.py` does everything at module level — on Windows spawn, each of the 4 DataLoader workers re-imports main.py and re-executes loader scan + model build + `.to("cuda")` + AdamW. Fix: periodic checkpointing inside train(); guard main.py under `if __name__ == "__main__":`.
5. **Feature-frame corruption** (`entity_encoding.py`): (a) relative ball–car velocity subtracts quantities normalized by different constants (ball vel /60000 vs car vel /23000) → physically meaningless difference; (b) positions are anisotropically normalized (x/4096, y/5120, z/2044) BEFORE rotation into the car frame → rotation of a sheared vector is no longer a rigid transform; "local frame" geometry is distorted (affects rel_ball_pos, rel_opp_position, pad rel_position; pad z is also hardcoded 0 vs normalized agent z). Fix: subtract/rotate in raw units, then scale isotropically. NOTE: changes input distribution → do before a long run; invalidates old checkpoints.
6. **No LR warmup or decay** — flat 1e-4 (V-JEPA: linear warmup 40/300 epochs → cosine; wd ramped 0.04→0.4; clipping only after warmup). Warmup matters extra with EMA targets (early targets are garbage).
7. **Predictor missing final LayerNorm/projection head** (`transformer.py:104`) — raw residual stream must match layer-normed targets; V-JEPA ends with predictor_norm → predictor_proj. Also its mask token is zero-init (R-JEPA: randn std 1.0 — hot).
8. **Identical data order every epoch** (`loader.py:83,408`) — RNG re-seeded with the same seed on each loader re-iteration; verified byte-identical batches across "epochs." Mix an epoch counter into the seed.
9. **Every-step diagnostics overhead** (`training_loop.py:30-36`) — extra full encoder forward (fp32, outside autocast) + 256×256 eigendecomposition + `.item()` sync every step ≈ +30% step time. Gate to every ~50 steps.
10. **Cosmetics**: predictor owns a dead unused ObjectEncoder (~200k grad-less params); dead `JEPA.build_mask` method; module-global `STATES=10` coupling; `EPOCHS` defined in two places; unused `device_type` var in train(); `ds.obj_lengths` (9,30,30,41 prefix counts) vs JEPA `obj_lengths` (19,19,9,7,170 token dims) naming collision.

**Confirmed correct** (verified empirically, matches V-JEPA): no gradient leakage into target encoder; EMA math + after-step order; target layer-norm; loss on masked positions only with aligned index order; pos-encoding added before masking; per-sample masks; bf16 without GradScaler (correct); loader hygiene (no boundary-crossing windows, span-level dead-time filtering, no worker data duplication, fp16 overflow handled).

**V-JEPA reference numbers** (ViT-L/16 pretrain config): ema 0.998→1.0; 300 epochs, warmup 40; lr 2e-4→6.25e-4→1e-6; wd 0.04→0.4; clip 10.0 after warmup; loss mean|Δ|^1 averaged over multiple per-sample masks; predictor narrows 768→384 with final norm+proj; masks = per-sample spatial multiblocks spanning the FULL temporal extent (no anchor frame); optional predictor-variance collapse regularizer `mean(relu(1 − std(z)))`.

**What to watch during real training**: effective rank of pooled latents should sit well above ~30/256 and latent_std should stabilize; a sustained slide of both toward single digits = collapse alarm. Smoke-test effrank ≈7 was a synthetic-data artifact, not a model verdict.

---

## 4. Complete source code (verbatim, appended below)

Files: `main.py`, `jepa.py`, `models/transformer.py`, `models/entity_encoding.py`, `models/pos_encoding.py`, `training/training_loop.py`, `training/functions.py`, `training/loader.py`, `training/boost_pad_state.py`.


---

## `main.py`

```python
import torch
from jepa import JEPA
from training.loader import build_window_loader
from training.functions import save_checkpoint
from training.training_loop import train

LR = 1e-4
WEIGHT_DECAY = 1e-5
SHARDS        = r"C:\Users\charl\R-JEPA2\data\shards_150k"   # <- point at your local shard directory
WINDOW        = 10
BATCH_SIZE    = 2048
EPOCHS        = 100
NUM_WORKERS   = 4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR      = "checkpoints"
MIRROR = False

loader, ds = build_window_loader(
        SHARDS, window=WINDOW, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pad_state=True, normalize="physical", mirror=MIRROR,
    )

obj_lengths = ds.obj_lengths
print(obj_lengths)

R_JEPA = JEPA(
        latent_dim=256,
        encoder_blocks=5,
        encoder_hdim=1024,
        encoder_attheads=4,
        proj_blocks=2,
        proj_hdim=128,
        proj_attheads=4,
        momentum=(0.995, 1.0, 10_000),
        obj_lengths=(19, 19, 9, 7, 170),
        emb_hdim=128,
        mask_probs=torch.tensor([0.35, 0.4, 0.05, 0.2])
)

R_JEPA.to(DEVICE)

optim = torch.optim.AdamW(R_JEPA.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


if __name__ == '__main__':
    try:
        train(R_JEPA, loader, optim, device=DEVICE)
    except KeyboardInterrupt:
        path = save_checkpoint(R_JEPA, f"{CKPT_DIR}/rjepa_interrupt.pt", optim=optim)
        print(f"\nInterrupted — saved checkpoint to {path}")


```

---

## `jepa.py`

```python
import torch
from models.transformer import Transformer
import torch.nn.functional as F
from copy import deepcopy

import torch.nn as nn


STATES = 10

def build_mask(state, mask_probs, num_objects):
    B, T, D = state.shape
    obj = torch.multinomial(mask_probs, B, replacement=True)
    masked_idx = obj.unsqueeze(1) + torch.arange(1, STATES, device=state.device) * num_objects

    mask = torch.zeros(B, T, dtype=torch.bool, device=state.device)
    mask.scatter_(1, masked_idx, True)

    visible_tokens = state[~mask].view(B, T - masked_idx.size(1), D)
    return masked_idx, visible_tokens

class JEPA(nn.Module):
    def __init__(self,
                 latent_dim=128,
                 encoder_blocks=2,
                 encoder_hdim=256,
                 encoder_attheads=4,
                 proj_blocks=2,
                 proj_hdim=128,
                 proj_attheads=4,
                 momentum=(0.995, 1.0, 10_000),
                 obj_lengths=(12, 22, 16, 23),
                 emb_hdim=256,
                 mask_probs=(0.35, 0.4, 0.05, 0.2),
                 STATES=10
                 ):
        super().__init__()

        # build the 3 main architectures for training the encoder
        self.encoder = Transformer(encoder_blocks, latent_dim, encoder_hdim, encoder_attheads, obj_lengths, emb_hdim, STATES)
        self.predictor = Transformer(proj_blocks, latent_dim, proj_hdim, proj_attheads, obj_lengths, emb_hdim, STATES, proj=True)
        self.target_encoder = deepcopy(self.encoder)

        self.target_encoder.requires_grad_(False)
        self.momentum = momentum[0]
        self.objects = len(obj_lengths)
        self.register_buffer("mask_probs", torch.as_tensor(mask_probs))

    def forward(self, raw_state):

        """
        This is the main forward pass of the project, it takes all models and combines them
        for the JEPA learning process.

        1. it takes raw batched state data [B, S, D_S] (D_S is the raw state dim approx 96)
        and then converts them into clean object specific tokens, while also flattening into
        [B, HIST*OBJS, D] where D is the necessary latent dim

        2. then positional encoding is applied to the flattened tokens to preserve information
        about the order of both states and objects, so the model can understand which token is
        from where

        3. a mask is applied to the flattened tokens, the mask works by choosing one object
        which is kept masked for the entire history except the first to give an anchor.
        this forces other tokens to gather context that allows other objects dynamics to be predicted

        4. the tokens which aren't masked goes into the main encoder and then into the predictor
        while the full unmasked states go into the EMA target encoder to then use as labels

        5. once the true encoder gives the context rich non-masked latents they are given to the predictor
        which also have the masked queries given as input, these queries have pos encoding telling the model
        which tokens were actually masked, then during the forward pass the predictor learns to use helpful
        information from the context rich unmasked latents to help guess the masked latents.

        6. take only the masked latent tokens from the predictor then use smooth L1 loss with the targets
        masked latents, ultimately this should train the model to produce latents which hold information
        that is essential for helping predict other objects dynamics.
        """

        state = self.encoder.embedding.build(raw_state)  # [B, HIST, D_S] -> [B, 20, LATENT]
        state = self.encoder.pos(state)  # add pos encoding to all tokens

        # mask the given objects
        masked_indices, non_masked_tokens = build_mask(state, self.mask_probs, self.objects)

        # run the non-masked objects through the necessary encoder
        context_latents = self.encoder(non_masked_tokens)
        masked_latents = self.predictor(context_latents, masked_indices)

        # run the same process with the target encoder and ensure no grad gets through
        with torch.no_grad():
            state = self.target_encoder.embedding.build(raw_state)  # [B, HIST, D_S] -> [B, 20, LATENT]
            state = self.target_encoder.pos(state)  # add pos encoding to all tokens

            target_latents = self.target_encoder(state)
            rows = torch.arange(state.size(0), device=state.device).unsqueeze(1)
            target_latents = F.rms_norm(target_latents, (target_latents.size(-1),))
            masked_target_latents = target_latents[rows, masked_indices]

        return masked_latents, masked_target_latents

    @torch.no_grad()
    def update_target_params(self):
        for new_params, old_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            # works to make target new params an EMA of the true encoder.
            # theta_t = (m * theta_t-1) + (1 - m)(theta_t)
            # lerp is Linear Interpolate between the old params and the new params with weight 1-m
            # it ultimately is the same operation as the EMA above
            old_params.lerp_(new_params, weight=1.0 - self.momentum)

    def build_mask(self, num_tokens, device, num_masked=5):
        mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        mask[torch.randperm(num_tokens, device=device)[:num_masked]] = True
        return mask


```

---

## `models/transformer.py`

```python
try:                                     # imported as models.transformer (main.py)
    from models.pos_encoding import PosEncoding
    from models.entity_encoding import ObjectEncoder
except ImportError:                      # run/imported directly from inside models/
    from pos_encoding import PosEncoding
    from entity_encoding import ObjectEncoder
import torch
import torch.nn as nn
import math

STATES = 10


class Transformer(nn.Module):
    def __init__(self, blocks, residual_dim, hidden_dim, att_heads, obj_lengths, emb_dim, STATES, proj=False):
        super().__init__()

        # make essential variables class global
        self.blocks = blocks
        self.dim = residual_dim
        self.proj = proj

        # otherwise pytorch will error
        if residual_dim % att_heads != 0:
            raise ValueError("residual_dim must be divisible by att_heads")

        # FFN: 1 layer w GeLU activation
        # MHA: n head attention
        # norm: RMS norm
        ffn = lambda: nn.Sequential(nn.Linear(residual_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, residual_dim))
        att = lambda: nn.MultiheadAttention(self.dim, att_heads, batch_first=True)
        norm = lambda: nn.RMSNorm(self.dim, eps=1e-6)

        # Transformer assumes the data is already encoded in shape [B, OBJS * HIST, DIM]
        self.embedding = ObjectEncoder(obj_lengths, residual_dim, emb_dim)
        self.pos = PosEncoding(residual_dim, states=STATES, objects=len(obj_lengths))

        self.attention = nn.ModuleList([att() for _ in range(blocks)])
        self.ffn = nn.ModuleList([ffn() for _ in range(blocks)])
        self.norm = nn.ModuleList([norm() for _ in range(blocks * 2)])

        # projector mode: ONE shared learned mask token. Per-slot identity is NOT
        # baked into separate query vectors (the old 5 anonymous queries couldn't
        # know which position they predicted); instead forward() adds the target
        # slot's position encoding, so each query is mask_token + pos[masked_idx].
        if proj is not False:
            self.mask_token = nn.Parameter(torch.randn(1, residual_dim))

    def block(self, x, i):

        # simple Pre-Norm transformer

        norm_1 = self.norm[2 * i]
        norm_2 = self.norm[2 * i + 1]
        attention = self.attention[i]
        feedforward = self.ffn[i]

        norm_out = norm_1(x)
        att_out, _ = attention(norm_out, norm_out, norm_out, need_weights=False)

        x = x + att_out

        norm_out = norm_2(x)
        ff_out = feedforward(norm_out)

        x = x + ff_out

        return x

    def projection(self, x, masked_indices, encode=True, n_masked=None):
        if encode:

            """
            first grabs the pos encoding specifically for the masked queries, so the model
            can understand which tokens were masked and in what states. then adds the pe to the queries
            and adds them back into x for the proj transformer to decode
            """

            pe = self.pos.table(x.device)[masked_indices]  # [B, n_masked, DIM] (per-sample indices)
            queries = self.mask_token + pe                 # [1, DIM] broadcasts over [B, n_masked, DIM]
            n_masked = queries.size(1)
            x = torch.cat((x, queries), dim=1)
            return x, n_masked

        else:
            # simply just returns only the masked tokens, since they are the ones which are
            # required for the loss function
            return x[:, -n_masked:]

    def forward(self, x, masked_indices=None):

        """
        forward pass for the transformer, projector specific functions before and after main pass
        to deal with masked tokens, otherwise standard multi-block Pre-Norm transformer with
        RMS norm and MHA.
        """

        if self.proj: x, num_masked = self.projection(x, masked_indices)

        for i in range(self.blocks):
            x = self.block(x, i)

        # return only the mask-query outputs, in masked_indices order
        if self.proj: x = self.projection(x, None, encode=False, n_masked=num_masked)

        return x
```

---

## `models/entity_encoding.py`

```python
import torch
import torch.nn as nn
import numpy as np
# imported as models.entity_encoding (via main.py)
from training.boost_pad_state import BOOST_PADS, PAD_XY, IS_BIG



def build_obs(state):

    agent_position = state[:, :, 9:12]
    agent_lin_vel = state[:, :, 12:15]
    agent_ang_vel = state[:, :, 15:18]
    agent_fwd = state[:, :, 18:21]
    agent_right = state[:, :, 21:24]
    agent_up = state[:, :, 24:27]

    agent_boost = state[:, :, 27:28]
    agent_jump_active = state[:, :, 28:29]
    agent_dodge_active = state[:, :, 29:30]
    agent_double_jump_active = state[:, :, 30:31]

    # Columns are the agent's local basis vectors expressed in world coordinates.
    # Transposing the final two dimensions gives world -> agent-local rotation.
    rot = torch.stack(
        [
            agent_fwd,
            agent_right,
            agent_up,
        ],
        axis=-1,
    )

    world_to_local = rot.transpose(-1, -2)

    def to_local(vector):
        return torch.einsum("btij,btj->bti", world_to_local, vector)

    self_vec = torch.cat(
        [
            agent_position,
            to_local(agent_lin_vel),
            to_local(agent_ang_vel),
            agent_fwd,
            agent_up,
            agent_boost,
            agent_jump_active,
            agent_dodge_active,
            agent_double_jump_active,
        ],
        dim=-1,
    )

    """
    ---------------------->
    OPP VECTOR
    ---------------------->
    """

    opp_position = state[:, :, 39:42]
    opp_lin_vel = state[:, :, 42:45]
    opp_ang_vel = state[:, :, 45:48]
    opp_fwd = state[:, :, 48:51]
    opp_up = state[:, :, 54:57]

    rel_opp_position = to_local(opp_position - agent_position)
    rel_opp_lin_vel = to_local(opp_lin_vel - agent_lin_vel)
    opp_ang_vel = to_local(opp_ang_vel)
    opp_fwd = to_local(opp_fwd)
    opp_up = to_local(opp_up)

    opp_boost = state[:, :, 57:58]
    opp_jump_active = state[:, :, 58:59]
    opp_dodge_active = state[:, :, 59:60]
    opp_double_jump_active = state[:, :, 60:61]

    opp_vec = torch.cat(
        [
            rel_opp_position,
            rel_opp_lin_vel,
            opp_ang_vel,
            opp_fwd,
            opp_up,
            opp_boost,
            opp_jump_active,
            opp_dodge_active,
            opp_double_jump_active,
        ],
        dim=-1,
    )

    """
    ---------------------->
    BALL VECTOR
    ---------------------->
    """

    ball_position = state[:, :, 0:3]
    ball_lin_vel = state[:, :, 3:6]
    ball_ang_vel = state[:, :, 6:9]

    rel_ball_pos = to_local(ball_position - agent_position)
    rel_ball_lin_vel = to_local(ball_lin_vel - agent_lin_vel)
    ball_ang_vel = to_local(ball_ang_vel)

    ball_vec = torch.cat(
        [
            rel_ball_pos,
            rel_ball_lin_vel,
            ball_ang_vel,
        ],
        dim=-1,
    )

    """
    ---------------------->
    ENV VECTOR
    ---------------------->
    """

    sec_remain = state[:, :, 69:70]
    is_overtime = state[:, :, 70:71]
    is_ball_hit = state[:, :, 71:72]
    own_score = state[:, :, 72:73]
    opp_score = state[:, :, 73:74]
    score_diff = state[:, :, 74:75]
    is_kickoff = state[:, :, 75:76]

    env_vec = torch.cat(
        [
            sec_remain,
            is_overtime,
            is_ball_hit,
            own_score,
            opp_score,
            score_diff,
            is_kickoff
        ],
        dim=-1,
    )

    """
    ---------------------->
    BOOST VECTOR
    ---------------------->
    """

    boost_vec = []

    for i in range(34):
        active_frac = state[:, :, 76 + i : 77 + i]

        is_big = torch.full_like(active_frac, IS_BIG[i])
        pad_position = torch.tensor([PAD_XY[i, 0] / 4096.0, PAD_XY[i, 1] / 5120.0, 0.0], dtype=state.dtype, device=state.device)

        rel_position = to_local(pad_position - agent_position)

        boost_vec.append(torch.cat([is_big, active_frac, rel_position], dim=-1))

    boost_vec = torch.cat(boost_vec, dim=-1)

    return self_vec, opp_vec, ball_vec, env_vec, boost_vec








class ObjectEncoder(nn.Module):
    def __init__(self, obj_lengths, latent_dim, hdim):
        super().__init__()

        # build a unique single layer MLP which takes the length of the object
        # and upscales it to the necessary residual dim to be used as a token
        embedding = lambda i: nn.Sequential(nn.Linear(obj_lengths[i], hdim), nn.GELU(), nn.Linear(hdim, latent_dim))

        self.objects = len(obj_lengths)
        self.object_lengths = obj_lengths
        self.object_projections = nn.ModuleList([embedding(i) for i in range(self.objects)])

    def build(self, x):
        """
        takes x which is the raw state data [B, D_S] D_s is the state dim (approx 96)
        and transforms it into [B, OBJS, D] where each object is parsed with the obj_lengths
        and then encoded with the embedding MLP
        """

        objs = build_obs(x)
        # [B, T, D_S] -> [B, T*OBJS, D] (time-major: state 0's objects, then state 1's, ...)
        return torch.stack([proj(o) for proj, o in zip(self.object_projections, objs)], dim=2).flatten(1, 2)

```

---

## `models/pos_encoding.py`

```python
import torch
import torch.nn as nn
import math




class PosEncoding(nn.Module):
    def __init__(self, dim, states, objects):
        super().__init__()

        self.states = states
        self.objects = objects

        # builds the two pos enc matrices for state and objs
        self.register_buffer("state_pe", self.sinusoidal(states, dim))
        self.register_buffer("object_pe", self.sinusoidal(objects, dim))

    @staticmethod
    def sinusoidal(length, dim):

        """
        builds the actual pos encoding matrix in shape [TOKENS, DIM]
        this is the standard matrix used in the original paper
        """

        pe = torch.zeros(length, dim)
        pos = torch.arange(length).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        return pe

    def table(self, device):

        """
        built specifically to deal with the flattened [B, HIST*OBJS, DIM] and apply
        pos encoding for each state in the history, and their sub objects
        allowing the model to understand which state and what object it is looking at
        """

        # torch state arrange -> 0, 0, 0, 0  1, 1, 1, 1 .... n, n, n, n
        state_ids = torch.arange(self.states, device=device).repeat_interleave(self.objects)
        # torch obj arrange -> 0, 1, 2, 3  0, 1, 2, 3 .... 0, 1, 2, 3
        object_ids = torch.arange(self.objects, device=device).repeat(self.states)

        # assigns the appropriate pos for a given state and obj position
        return self.state_pe[state_ids] + self.object_pe[object_ids]

    def forward(self, x):
        # adds the positional information into the residual stream
        return x + self.table(x.device).unsqueeze(0)
```

---

## `training/training_loop.py`

```python
import torch
import torch.nn.functional as F
from training.functions import effective_rank

EPOCHS = 100


def train(model, loader, optim, device='cuda'):
    model.train()
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    step = 0

    for epoch in range(EPOCHS):

        for window in loader:
            window = window.to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                z_hat, z = model(window)  # run the forward pass

                loss = F.smooth_l1_loss(z_hat, z)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optim.step()
            model.update_target_params()

            with torch.no_grad():
                b = window.size(0)
                rep = model.encoder.embedding.build(window)
                rep = model.encoder.pos(rep)
                rep = model.encoder(rep).mean(1)
                print(f"itr: {step}, loss={loss.item():.5f} effrank={effective_rank(rep):.1f} "
                      f"latent_std={rep.std(0).mean().item():.4f}")

            step += 1

```

---

## `training/functions.py`

```python
import torch
import torch.nn.functional as F
from pathlib import Path


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


def effective_rank(embeddings):
    # embeddings: [N, D]
    emb = embeddings - embeddings.mean(0, keepdim=True)
    cov = emb.T @ emb / emb.shape[0]
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
    p = eigvals / eigvals.sum()
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()  # ranges from 1 (collapsed) to D (full rank)


def batch_collapse_metrics(embeddings):
    # embeddings: [N, D], N should be decent size (>=64)
    emb = F.normalize(embeddings, dim=-1)
    sim_matrix = emb @ emb.T  # [N, N]
    N = sim_matrix.shape[0]
    off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=emb.device)
    mean_offdiag_sim = sim_matrix[off_diag_mask].mean()
    return mean_offdiag_sim.item()

```

---

## `training/loader.py`

```python
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
            try:
                from training.boost_pad_state import PAD_FEATURE_NAMES
            except ImportError:      # loader.py run directly from training/
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
                try:
                    from training.boost_pad_state import shard_pad_recharge
                except ImportError:
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
```

---

## `training/boost_pad_state.py`

```python
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
```
