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

