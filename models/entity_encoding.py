import torch
import torch.nn as nn




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

        objs = x.split(self.object_lengths, dim=1)  # [B, D_S] -> [OBJS, B, D_obj]
        # ... -> [B, OBJS, D]  where D is latent dim
        return torch.stack([proj(objs) for proj, objs in zip(self.object_projections, objs)], dim=1)