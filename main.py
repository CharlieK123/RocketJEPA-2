import json
import time
from pathlib import Path

import torch
from jepa import JEPA
from training.loader import build_window_loader
from training.training_loop import train
from training.functions import param_groups

from pathlib import Path

DATA_DIR = Path("/workspace/data/shards_75k")

LR = (1.5e-4, 4e-4, 1e-5, 100_000)   # start, peak, final, total
WARMUP_STEPS = 7_000               # keep
WEIGHT_DECAY = (0.04, 0.4)   # (start, final) cosine-ramped UP over LR[3] steps (V-JEPA style)
SHARDS        = str(DATA_DIR)   # <- point at your local shard directory
WINDOW        = 15
BATCH_SIZE    = 2500
EPOCHS        = 100
NUM_WORKERS   = 4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR      = "../checkpoints"
MIRROR = True
GAP = 2

# JSON-serializable so it can be dumped to config.json and stamped into every
# checkpoint; mask_probs is converted to a tensor at model construction.
MODEL_CFG = dict(
        latent_dim=384,
        encoder_blocks=10,
        encoder_hdim=1024,
        encoder_attheads=4,
        proj_blocks=4,
        proj_hdim=256,
        proj_attheads=4,
        momentum=(0.998, 1.0, LR[3]),   # anneal over the whole run, in lockstep with the LR
        obj_lengths=(19, 19, 9, 7, 170),
        emb_hdim=512,
        mask_probs=[0.10, 0.35, 0.45, 0.05, 0.05],
)

# everything below is guarded so Window s-spawn DataLoader workers (which re-import
# this module) don't rebuild the loader/model or push extra copies onto the GPU
if __name__ == '__main__':
    # each run gets its own checkpoint directory so runs never overwrite each other
    RUN_ID = time.strftime("%Y%m%d-%H%M%S")
    RUN_DIR = Path(CKPT_DIR) / RUN_ID

    HPARAMS = dict(
            run_id=RUN_ID,
            lr=LR, warmup_steps=WARMUP_STEPS, weight_decay=WEIGHT_DECAY,
            window=WINDOW, batch_size=BATCH_SIZE, epochs=EPOCHS,
            mirror=MIRROR, gap=GAP, normalize="physical", shards=SHARDS,
            model=MODEL_CFG,
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "config.json", "w") as f:
        json.dump(HPARAMS, f, indent=2)
    print(f"run {RUN_ID} — checkpoints & config -> {RUN_DIR.resolve()}")

    loader, ds = build_window_loader(
            SHARDS, window=WINDOW, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
            pad_state=True, normalize="physical", mirror=MIRROR, gap=GAP

        )

    obj_lengths = ds.obj_lengths
    print(obj_lengths)

    R_JEPA = JEPA(**{**MODEL_CFG, "mask_probs": torch.tensor(MODEL_CFG["mask_probs"])})

    for name, m in (("encoder", R_JEPA.encoder), ("predictor", R_JEPA.predictor),
                    ("target_encoder", R_JEPA.target_encoder)):
        print(f"{name}: {sum(p.numel() for p in m.parameters()):,} params")
    print(f"total: {sum(p.numel() for p in R_JEPA.parameters()):,} params")

    R_JEPA.to(DEVICE)

    optim = torch.optim.AdamW(param_groups(R_JEPA, WEIGHT_DECAY[0]), lr=LR[0], eps=1e-6)

    try:
        train(R_JEPA, loader, optim, lr=LR, warmup_steps=WARMUP_STEPS, wd=WEIGHT_DECAY,
              device=DEVICE, run_dir=str(RUN_DIR), hparams=HPARAMS)
    except KeyboardInterrupt:
        # train() already saved rjepa_interrupt_step{N}.pt (with step + train time) on its way out
        print("exiting.")