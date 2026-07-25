import torch
from jepa import JEPA
from training.loader import build_window_loader
from training.functions import save_checkpoint
from training.training_loop import train

LR = (2e-4, 4e-4, 1e-6, 210_000)   # start, peak, final, total
WARMUP_STEPS = 20_000              # keep
WEIGHT_DECAY = (0.04, 0.4)   # (start, final) cosine-ramped UP over LR[3] steps (V-JEPA style)
SHARDS        = r"C:\Users\charl\R-JEPA2\data\shards_150k"   # <- point at your local shard directory
WINDOW        = 15
BATCH_SIZE    = 2048
EPOCHS        = 100
NUM_WORKERS   = 4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR      = "../checkpoints"
MIRROR = True

# everything below is guarded so Windows-spawn DataLoader workers (which re-import
# this module) don't rebuild the loader/model or push extra copies onto the GPU
if __name__ == '__main__':
    loader, ds = build_window_loader(
            SHARDS, window=WINDOW, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
            pad_state=True, normalize="physical", mirror=MIRROR, gap=3

        )

    obj_lengths = ds.obj_lengths
    print(obj_lengths)

    R_JEPA = JEPA(
            latent_dim=256,
            encoder_blocks=6,
            encoder_hdim=1024,
            encoder_attheads=4,
            proj_blocks=3,
            proj_hdim=128,
            proj_attheads=4,
            momentum=(0.998, 1.0, LR[3]),   # anneal over the whole run, in lockstep with the LR
            obj_lengths=(19, 19, 9, 7, 170),
            emb_hdim=128,
            mask_probs=torch.tensor([0.10, 0.35, 0.45, 0.05, 0.05])
    )

    for name, m in (("encoder", R_JEPA.encoder), ("predictor", R_JEPA.predictor),
                    ("target_encoder", R_JEPA.target_encoder)):
        print(f"{name}: {sum(p.numel() for p in m.parameters()):,} params")
    print(f"total: {sum(p.numel() for p in R_JEPA.parameters()):,} params")

    R_JEPA.to(DEVICE)

    optim = torch.optim.AdamW(R_JEPA.parameters(), lr=LR[0], weight_decay=WEIGHT_DECAY[0])

    try:
        train(R_JEPA, loader, optim, lr=LR, warmup_steps=WARMUP_STEPS, wd=WEIGHT_DECAY, device=DEVICE)
    except KeyboardInterrupt:
        path = save_checkpoint(R_JEPA, f"{CKPT_DIR}/rjepa_interrupt.pt", optim=optim)
        print(f"\nInterrupted — saved checkpoint to {path}")

