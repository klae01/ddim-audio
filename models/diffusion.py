import itertools
import math
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from rfconv import RFConv2d

Conv2d = RFConv2d  # nn.Conv2d

sys.path.append("External")

from UPU.layers import Gate

Group_cnt = 32


def Norm_layer(channels, affine=True, group_cnt=None):
    if group_cnt is None:
        group_cnt = max(channels // 16, 1)
    return nn.GroupNorm(group_cnt, channels, eps=1e-8, affine=affine)


class shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        if self.stride != 1:
            x = nn.functional.avg_pool2d(
                x, self.stride, self.stride, ceil_mode=True, count_include_pad=False
            )
        if self.in_channels != self.out_channels:
            padding = self.out_channels - self.in_channels
            x = nn.functional.pad(x, (0, 0, 0, 0, padding, 0))

        return x


class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        chs = [in_channels, out_channels // 4, out_channels // 4, out_channels]
        self.norm = nn.ModuleList([Norm_layer(ch) for ch in chs])
        self.conv = nn.ModuleList(
            [
                Conv2d(ch_in, ch_out, k, 1, k // 2, bias=bias)
                for ch_in, ch_out, k, bias in zip(
                    chs, chs[1:], [1, kernel_size, 1], [True, True, False]
                )
            ]
        )
        self.gate_diffusion = Gate(torch.tensor([0.0] * chs[1]).view(-1, 1, 1))
        self.gate_residual = Gate(torch.tensor([0.0] * chs[-1]).view(-1, 1, 1))
        self.shortcut = shortcut(in_channels, out_channels, stride)

    def forward(self, input, temb):
        NORM = iter(self.norm)
        CONV = iter(self.conv)
        x = input
        identity = self.shortcut(x)

        x = next(NORM)(x)
        x = next(CONV)(x)
        x = self.gate_diffusion(x, temb[..., None, None])
        x = nn.functional.silu(x)

        x = next(NORM)(x)
        x = next(CONV)(x)
        x = nn.functional.silu(x)

        x = next(NORM)(x)
        x = next(CONV)(x)
        x = next(NORM)(x)

        return self.gate_residual(identity, x)


class Upsample(nn.Module):
    def __init__(self, *, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=False),
            Norm_layer(out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class Downsample(nn.Module):
    def __init__(self, *, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            Norm_layer(out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int = 8):
        assert hidden_size % heads == 0
        super(Attention, self).__init__()
        self.ctx_QKV_weight = nn.Parameter(
            torch.randn((3, heads, hidden_size // heads, hidden_size)) * 0.02
        )
        self.ctx_QKV_bias = nn.Parameter(
            torch.zeros((3, 1, 1, heads, hidden_size // heads))
        )
        self.out_proj = nn.Parameter(torch.randn((hidden_size, hidden_size)) * 0.02)
        self.act = lambda x: nn.functional.elu(x) + 1
        self.norm = nn.ModuleList([Norm_layer(hidden_size) for _ in range(2)])
        self.gate = Gate(torch.tensor([0.0] * hidden_size).view(-1, 1))

    def forward(self, hidden_states: torch.Tensor):
        NORM = iter(self.norm)
        B, F, S = hidden_states.size()
        x = hidden_states

        x = next(NORM)(x)
        ctx_Q, ctx_K, ctx_V = (
            torch.einsum("bfl,ihFf->iblhF", x, self.ctx_QKV_weight) + self.ctx_QKV_bias
        )
        ctx_Q, ctx_K = map(self.act, [ctx_Q, ctx_K])
        KV = torch.einsum("blhf,blhF->bhFf", ctx_K, ctx_V)

        # Original implementation.
        # Z = torch.einsum("blhf,bhf->blh", ctx_Q, ctx_K.sum(1)).reciprocal()
        # V = torch.einsum("blhf,bhFf,blh->blhF", ctx_Q, KV, Z)

        # Memory efficient implementation
        QZ = ctx_Q / (ctx_Q * ctx_K.sum(1, True)).sum(-1, True)
        V = torch.einsum("blhf,bhFf->blhF", QZ, KV)

        x = torch.einsum("blf,Ff->bFl", V.reshape(B, S, F), self.out_proj)
        x = next(NORM)(x)
        return self.gate(hidden_states, x)


class Block_set(nn.Module):
    class Attention_wrapper(Attention):
        def forward(self, x):
            B, C, F, T = x.shape
            x = x.reshape(B, C, F * T)
            x = super().forward(x)
            x = x.reshape(B, C, F, T)
            return x

    def __init__(self, channels, f_size, config, use_attention=False):
        # assume Attention include last linear layer
        super(Block_set, self).__init__()
        layers = [Residual_Block(channels)]  # residual for diffusion
        if use_attention:
            # frequency-time axis attention
            layers.append(Block_set.Attention_wrapper(channels, heads=config.heads))
        else:
            layers.append(nn.Identity())

        self.layers = nn.ModuleList(layers)

    def forward(self, x, temb):
        # x order = BCFT
        layer = iter(self.layers)
        x = next(layer)(x, temb)
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
    return torch.cat([X.sin(), X.cos()], dim=-1)


class BetaEmbedding(nn.Module):
    def __init__(self, channel_sz, config):
        super().__init__()
        pos_ch = config.pos_emb_dim

        self.config = config
        emb_ch = 512

        self.weight = nn.Sequential(
            nn.Linear(pos_ch, emb_ch, bias=True),
            nn.Linear(emb_ch, emb_ch, bias=True),
            nn.Linear(emb_ch, channel_sz, bias=True),
        )
        self.norm = nn.ModuleList([Norm_layer(emb_ch) for _ in range(2)])

    def forward(self, input):
        WIGT = iter(self.weight)
        NORM = iter(self.norm)

        x = get_embedding(input, self.config)
        x = next(WIGT)(x)
        x = nn.functional.silu(x)
        x = next(NORM)(x)
        x = next(WIGT)(x)
        x = nn.functional.silu(x)
        x = next(NORM)(x)
        x = next(WIGT)(x)

        return x


class G_mapping(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Conv2d(in_features, out_features * 2, 3, 1, 1, bias=True)
        self.weight.bias.data.zero_()

    def forward(self, input):
        x = input
        x = self.weight(x)
        μ, σ = x.chunk(2, 1)
        return μ, torch.sigmoid(σ)


class projection(nn.Module):
    def __init__(self, in_features, out_features, use_gaussian):
        super().__init__()
        self.norm = Norm_layer(in_features)
        if use_gaussian:
            self.conv = G_mapping(in_features, out_features)
        else:
            self.conv = Conv2d(in_features, out_features, 3, 1, 1, bias=True)
            self.conv.bias.data.zero_()

    def forward(self, input):
        x = input
        x = self.norm(x)
        x = nn.functional.silu(x)
        x = self.conv(x)
        return x


class Model(nn.Module):
    def __init__(self, config):
        # get full config

        self.config = config.model
        self.mapping = config.mapping
        super().__init__()

        res_module_ref_index = list(
            itertools.chain(
                range(len(self.config.res)), range(len(self.config.res) - 2, -1, -1)
            )
        )
        self.embedding_size = [
            self.config.ch[i] // 4
            for i in res_module_ref_index
            for _ in range(self.config.res[i])
        ]
        self.temb = BetaEmbedding(sum(self.embedding_size), self.config.embedding)

        input_ch = self.config.io.patch_size**2 * self.config.io.channels

        self.gate = nn.ModuleList(
            [
                Gate(torch.tensor([0.0] * ch).view(-1, 1, 1))
                for ch in self.config.ch[:-1]
            ]
        )

        # downsample can place start of chunks
        # upsample can place end of chunks

        self.residual_modules = nn.ModuleList()

        patch = self.config.io.patch_size
        f_size = self.config.io.f_size // patch
        for i, I in enumerate(res_module_ref_index):
            f_current = f_size // (2**I)
            Blocks = nn.ModuleList()
            use_attention = self.config.use_attention[I]
            for _ in range(self.config.res[I]):
                Blocks.append(
                    Block_set(self.config.ch[I], f_current, self.config, use_attention)
                )
            self.residual_modules.append(Blocks)

        conv1 = Conv2d(input_ch, self.config.ch[0], 7, patch, 3, bias=False)
        conv1 = nn.Sequential(conv1, Norm_layer(self.config.ch[0]))

        self.residual_modules[0].insert(0, conv1)
        for prev_ch, ch, res_m in zip(
            self.config.ch, self.config.ch[1:], self.residual_modules[1:]
        ):
            down = Downsample(in_channels=prev_ch, out_channels=ch)
            res_m.insert(0, down)
        for ch, next_ch, res_m in zip(
            self.config.ch[-1::-1],
            self.config.ch[-2::-1],
            self.residual_modules[max(res_module_ref_index) :],
        ):
            up = Upsample(in_channels=ch, out_channels=next_ch)
            res_m.append(up)
        self.residual_modules[-1].append(
            projection(self.config.ch[0], input_ch, self.mapping.gaussian)
        )

        self.type(self.config.dtype)

    def forward(self, input, a) -> Tuple[torch.Tensor, torch.Tensor]:
        # input: [B, T, F, C]
        gate = iter(self.gate[::-1])

        B, T, F, C = input.shape
        P = self.config.io.patch_size
        x = input.view(B, T // P, P, F // P, P, C)  # B T tP F fP C
        x = x.permute(0, 2, 4, 5, 3, 1)  # B tP fP C F T
        x = x.reshape(B, -1, F // P, T // P)

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

        convert = (
            lambda x: x.reshape(B, P, P, C, F // P, T // P)  # B tP fP C F T
            .permute(0, 5, 1, 4, 2, 3)  # B T tP F fP C
            .reshape(B, T, F, C)
        )
        if isinstance(x, tuple):
            x = map(convert, x)
        else:
            x = convert(x)

        return x
