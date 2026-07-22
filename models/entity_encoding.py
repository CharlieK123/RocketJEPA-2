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
