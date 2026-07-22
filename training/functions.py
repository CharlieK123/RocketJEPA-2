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
