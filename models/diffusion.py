import itertools
import math
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.append("External")

from UPU.layers import Gate
from UPU.layers.residual.blocks import LSA_block

Group_cnt = 32


class Residual_Block(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels

        self.norm = nn.Sequential(
            *[
                torch.nn.GroupNorm(
                    num_groups=Group_cnt, num_channels=channels, eps=1e-6, affine=False
                )
                for _ in range(3)
            ]
        )

        self.conv = nn.Sequential(
            *[
                torch.nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=bias,
                )
                for bias in [True, False]
            ]
        )
        self.gate_diffusion = Gate()
        self.gate_residual = Gate()

    def forward(self, input, temb):
        NORM = iter(self.norm)
        CONV = iter(self.conv)
        x = input

        x = next(NORM)(x)
        x = next(CONV)(x)
        x = self.gate_diffusion(x, temb[..., None, None])

        x = nn.functional.silu(x)
        x = next(NORM)(x)
        x = next(CONV)(x)
        x = next(NORM)(x)
        return self.gate_residual(input, x)


class Upsample(nn.Module):
    def __init__(self, *, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False
        )
        self.norm = nn.GroupNorm(Group_cnt, out_channels, affine=False)

    def forward(self, x):
        return self.norm(self.conv(x))


class Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int = 8):
        assert hidden_size % heads == 0
        super(Attention, self).__init__()
        self.heads = heads
        self.ctx_QKV = nn.Linear(hidden_size, hidden_size * 3)
        self.ctx_QKV.bias.data.zero_()
        self.act = lambda x: nn.functional.elu(x) + 1

    def forward(self, hidden_states: torch.Tensor):
        B, S, F = hidden_states.size()
        ctx_Q, ctx_K, ctx_V = (
            self.ctx_QKV(hidden_states).view(B, S, 3, self.heads, -1).unbind(dim=2)
        )
        ctx_Q, ctx_K = map(self.act, [ctx_Q, ctx_K])
        KV = torch.einsum("blhf,blhF->bhFf", ctx_K, ctx_V)

        # Original implementation.
        # Z = torch.einsum("blhf,bhf->blh", ctx_Q, ctx_K.sum(dim=1)).reciprocal()
        # V = torch.einsum("blhf,bhFf,blh->blhF", ctx_Q, KV, Z)

        # Memory efficient implementation
        QZ = ctx_Q / (ctx_Q * ctx_K.sum(dim=1, keepdims=True)).sum(
            dim=-1, keepdims=True
        )
        V = torch.einsum("blhf,bhFf->blhF", QZ, KV)

        return V.reshape(B, S, F)


class Block_set(nn.Module):
    def __init__(self, channels, f_size, config):
        # assume Attention include last linear layer
        super(Block_set, self).__init__()
        self.layers = nn.ModuleList(
            [
                Residual_Block(channels),  # residual for diffusion
                Attention(
                    channels, heads=config.heads
                ),  # frequency-time axis attention
                LSA_block(
                    channels,
                    channels,
                    local_window_size=config.local_window_size,
                    normalize_group_size=Group_cnt,
                ),
            ]
        )
        self.gate = Gate()

    def forward(self, x, temb):
        # x order = BCFT
        layer = iter(self.layers)
        x = next(layer)(x, temb)
        B, C, F, T = x.shape

        x = x.permute(0, 3, 2, 1).reshape(B, T * F, C)
        y = nn.functional.group_norm(
            next(layer)(x).view(-1, x.size(-1)), num_groups=Group_cnt
        )
        x = self.gate(x, y.view(x.size()))

        x = x.reshape(B, T, F, C).permute(0, 3, 2, 1)
        x = next(layer)(x)
        return x


def get_embedding(a_sqrt, config):
    X = (
        a_sqrt[..., None]
        * 10
        ** (
            -torch.arange(
                config.pos_emb_dim // 2, dtype=a_sqrt.dtype, device=a_sqrt.device
            )
            * config.gamma
        )
        * config.scale
    )
    return torch.cat([torch.sin(X), torch.cos(X)], dim=-1)


class BetaEmbedding(nn.Module):
    def __init__(self, channel_sz, config):
        super().__init__()
        pos_ch = config.pos_emb_dim

        self.config = config
        emb_ch = 512

        self.weight = nn.Sequential(
            torch.nn.Linear(pos_ch, emb_ch, bias=True),
            torch.nn.Linear(emb_ch, emb_ch, bias=True),
            torch.nn.Linear(emb_ch, channel_sz, bias=False),
        )

    def forward(self, input):
        WIGT = iter(self.weight)

        x = get_embedding(input, self.config)
        x = next(WIGT)(x)
        x = torch.nn.functional.silu(x)
        x = nn.functional.group_norm(x, num_groups=Group_cnt)
        x = next(WIGT)(x)
        x = torch.nn.functional.silu(x)
        x = nn.functional.group_norm(x, num_groups=Group_cnt)
        x = next(WIGT)(x)
        x = nn.functional.layer_norm(x, x.shape[1:])

        return x


class G_mapping(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Conv2d(in_features, out_features * 2, kernel_size=1, bias=True)
        self.weight.bias.data.zero_()

    def forward(self, input):
        x = input
        x = self.weight(x)
        μ, σ = x.chunk(2, 1)
        return μ, torch.sigmoid(σ)


class Model(nn.Module):
    def __init__(self, config):
        # get full config

        self.config = config.model
        super().__init__()

        res_module_ref_index = list(
            itertools.chain(
                range(len(self.config.res)), range(len(self.config.res) - 2, -1, -1)
            )
        )
        self.embedding_size = [
            self.config.ch[i]
            for i in res_module_ref_index
            for _ in range(self.config.res[i])
        ]
        self.temb = BetaEmbedding(sum(self.embedding_size), self.config.embedding)

        input_ch = self.config.io.patch_size**2 * self.config.io.channels

        self.gate = nn.ModuleList([Gate() for _ in self.config.res[:-1]])

        # downsample can place start of chunks
        # upsample can place end of chunks

        self.residual_modules = nn.ModuleList()

        f_size = self.config.io.f_size // self.config.io.patch_size
        for i, I in enumerate(res_module_ref_index):
            f_current = f_size // (2**I)
            Blocks = nn.ModuleList()
            for _ in range(self.config.res[I]):
                Blocks.append(Block_set(self.config.ch[I], f_current, self.config))
            self.residual_modules.append(Blocks)

        conv1 = nn.Conv2d(input_ch, self.config.ch[0], 1, bias=False)
        nn.init.orthogonal_(conv1.weight)
        conv1 = nn.Sequential(conv1, nn.GroupNorm(Group_cnt, 0, affine=False))

        self.residual_modules[0].insert(0, conv1)
        for prev_ch, ch, res_m in zip(
            self.config.ch, self.config.ch[1:], self.residual_modules[1:]
        ):
            down = LSA_block(
                prev_ch, ch, stride=2, local_window_size=self.config.local_window_size
            )
            res_m.insert(0, down)
        for ch, next_ch, res_m in zip(
            self.config.ch[-1::-1],
            self.config.ch[-2::-1],
            self.residual_modules[max(res_module_ref_index) :],
        ):
            up = Upsample(in_channels=ch, out_channels=next_ch)
            res_m.append(up)
        self.residual_modules[-1].append(G_mapping(self.config.ch[0], input_ch))

        self.type(self.config.dtype)

    def forward(self, input, a) -> Tuple[torch.Tensor, torch.Tensor]:
        # input: [B, T, F, C]
        gate = iter(self.gate[::-1])

        B, T, F, C = input.shape
        P = self.config.io.patch_size
        x = input.view(B, T // P, P, F // P, P, C)
        x = x.permute(0, 2, 4, 5, 3, 1).reshape(B, -1, F // P, T // P)

        temb = self.temb(a.view(-1))
        temb = torch.split(temb, self.embedding_size, dim=-1)
        temb = iter(temb)

        HS = []
        for i, I in enumerate(self.residual_modules):
            if i >= len(self.config.res):
                x = next(gate)(x, HS.pop(-1))

            for M in I:
                if isinstance(M, Block_set):
                    x = M(x, next(temb))
                else:
                    x = M(x)

            if i + 1 < len(self.config.res):
                HS.append(x)

        x = [
            I.reshape(B, P, P, C, F // P, T // P)
            .permute(0, 5, 1, 4, 2, 3)
            .reshape(B, T, F, C)
            for I in x
        ]

        return x
