import torch
from models.transformer import Transformer
import torch.nn.functional as F
from copy import deepcopy

import torch.nn as nn


STATES = 15

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
                 STATES=15
                 ):
        super().__init__()

        # build the 3 main architectures for training the encoder
        self.encoder = Transformer(encoder_blocks, latent_dim, encoder_hdim, encoder_attheads, obj_lengths, emb_hdim, STATES)
        self.predictor = Transformer(proj_blocks, latent_dim, proj_hdim, proj_attheads, obj_lengths, emb_hdim, STATES, proj=True)
        self.target_encoder = deepcopy(self.encoder)

        self.target_encoder.requires_grad_(False)
        self.m_start, self.m_final, self.m_steps = momentum
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

            # target_encoder already applies its final out_norm, so its latents are
            # unit-RMS normalized here (previously done explicitly with F.rms_norm).
            target_latents = self.target_encoder(state)
            rows = torch.arange(state.size(0), device=state.device).unsqueeze(1)
            masked_target_latents = target_latents[rows, masked_indices]

        return masked_latents, masked_target_latents, masked_indices

    @torch.no_grad()
    def update_target_params(self, step):
        # momentum anneals linearly m_start -> m_final over m_steps, then holds
        m = self.m_start + (self.m_final - self.m_start) * min(step / self.m_steps, 1.0)
        for new_params, old_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            # works to make target new params an EMA of the true encoder.
            # theta_t = (m * theta_t-1) + (1 - m)(theta_t)
            # lerp is Linear Interpolate between the old params and the new params with weight 1-m
            # it ultimately is the same operation as the EMA above
            old_params.lerp_(new_params, weight=1.0 - m)

    def build_mask(self, num_tokens, device, num_masked=5):
        mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        mask[torch.randperm(num_tokens, device=device)[:num_masked]] = True
        return mask

