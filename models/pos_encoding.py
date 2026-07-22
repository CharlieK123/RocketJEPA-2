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