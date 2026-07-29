try:                                     # imported as models.transformer (main.py)
    from models.pos_encoding import PosEncoding
    from models.entity_encoding import ObjectEncoder
except ImportError:                      # run/imported directly from inside models/
    from pos_encoding import PosEncoding
    from entity_encoding import ObjectEncoder
import torch
import torch.nn as nn
import math

STATES = 15


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
        # norm: LayerNorm
        ffn = lambda: nn.Sequential(nn.Linear(residual_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, residual_dim))
        att = lambda: nn.MultiheadAttention(self.dim, att_heads, batch_first=True)
        norm = lambda: nn.LayerNorm(self.dim, eps=1e-6)

        # Transformer assumes the data is already encoded in shape [B, OBJS * HIST, DIM]
        if not proj:
            self.embedding = ObjectEncoder(obj_lengths, residual_dim, emb_dim)
        self.pos = PosEncoding(residual_dim, states=STATES, objects=len(obj_lengths))

        self.attention = nn.ModuleList([att() for _ in range(blocks)])
        self.ffn = nn.ModuleList([ffn() for _ in range(blocks)])
        self.norm = nn.ModuleList([norm() for _ in range(blocks * 2)])

        # Final norm on the output tokens. A pre-norm transformer never normalizes
        # its last block's output, so encoder latents (context + target) and the
        # predictor's mask-query outputs must be pinned to the same normalized manifold
        # before the loss compares them. Affine-free to match I-JEPA's
        # F.layer_norm-on-targets behaviour and to keep no learnable gain on the
        # stop-grad target (which would open a scale-collapse path).
        self.out_norm = nn.LayerNorm(self.dim, eps=1e-6, elementwise_affine=False)

        # projector mode: ONE shared learned mask token. Per-slot identity is NOT
        # baked into separate query vectors (the old 5 anonymous queries couldn't
        # know which position they predicted); instead forward() adds the target
        # slot's position encoding, so each query is mask_token + pos[masked_idx].
        if proj is not False:
            self.mask_token = nn.Parameter(torch.zeros(1, residual_dim))
            self.out_proj = nn.Linear(self.dim, self.dim)

        # Depth-dependent residual rescaling (BEiT/DINO/MAE, carried into I-JEPA &
        # V-JEPA). Every residual branch pours variance into the stream, so left
        # alone the stream norm grows ~sqrt(depth) and the last blocks dominate at
        # init. Scaling each block's OUTPUT projections (attn out_proj + FFN fc2)
        # by 1/sqrt(2 * layer_id) holds the residual-stream variance ~constant with
        # depth, for a calmer, better-conditioned start.
        self.fix_init_weight()

    def fix_init_weight(self):
        # layer_id is 1-indexed: block 0 -> /sqrt(2*1), block 1 -> /sqrt(2*2), ...
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id in range(self.blocks):
            rescale(self.attention[layer_id].out_proj.weight.data, layer_id + 1)
            rescale(self.ffn[layer_id][2].weight.data, layer_id + 1)

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
        LayerNorm and MHA.
        """

        if self.proj: x, num_masked = self.projection(x, masked_indices)

        for i in range(self.blocks):
            x = self.block(x, i)

        # return only the mask-query outputs, in masked_indices order
        if self.proj: x = self.projection(x, None, encode=False, n_masked=num_masked)

        # encoder: normalizes all output latents; predictor: normalizes the mask-query
        # outputs — either way the returned tokens land on the LayerNorm manifold.
        x = self.out_norm(x)
        if self.proj: x = self.out_proj(x)

        return x