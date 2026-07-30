import torch
import torch.nn as nn
import math




import torch
import torch.nn as nn
import math




class PosEncoding(nn.Module):
    def __init__(self, dim, states, objects):
        super().__init__()

        self.states = states
        self.objects = objects

        # the two axes get DISJOINT halves of the feature dim and are concatenated
        # in table(), never summed. Summing two sinusoidal tables built from the same
        # frequency set is symmetric in its arguments, so PE(state=s, obj=o) came out
        # bit-identical to PE(state=o, obj=s) for every s, o < objects -- 10 exact
        # collisions among the 75 tokens, e.g. state1/ball == state2/opp (two mask
        # queries) and state1/self == state0/opp (a query vs a VISIBLE token).
        # Queries are mask_token + PE with no content of their own, so colliding
        # slots were the same input vector trained toward different targets.
        # Disjoint slices make cross-axis collisions structurally impossible
        # (V-JEPA does the same in get_3d_sincos_pos_embed: one slice per axis,
        # concatenated). State takes the larger half when dim is odd.
        state_dim = dim - dim // 2
        object_dim = dim // 2

        # builds the two pos enc matrices for state and objs
        self.register_buffer("state_pe", self.sinusoidal(states, state_dim))
        self.register_buffer("object_pe", self.sinusoidal(objects, object_dim))

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

        # assigns the appropriate pos for a given state and obj position.
        # concatenated (not added) so the state half and the object half occupy
        # separate dims and can never alias into each other -- see __init__.
        return torch.cat([self.state_pe[state_ids], self.object_pe[object_ids]], dim=-1)

    def forward(self, x):
        # adds the positional information into the residual stream
        return x + self.table(x.device).unsqueeze(0)