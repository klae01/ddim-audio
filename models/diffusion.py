import math
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import transformers.activations

sys.path.append("External")

from UPU.layers.normalize.groupnorm import GroupNorm1D


class PoolingAttention(nn.Module):
    def __init__(
        self,
        in_features: int,
        attention_features: int,
        segments: int,
        max_pool_kernel: int,
    ):
        super(PoolingAttention, self).__init__()
        self.attn = nn.Linear(in_features, attention_features * 6, bias=False)
        self.segments = segments
        self.max_pool_kernel = max_pool_kernel

    def forward(self, inp: torch.Tensor):  # Shape: [Batch, Sequence, Features]
        batch, sequence, features = inp.size()
        assert sequence % self.segments == 0

        qry, key, val, seg, loc, out = self.attn(inp).chunk(
            6, 2
        )  # 6x Shape: [Batch, Sequence, AttentionFeatures]

        aggregated = qry.mean(1)  # Shape: [Batch, AttentionFeatures]
        aggregated = torch.einsum(
            "ba,bsa->bs", aggregated, key
        )  # Shape: [Batch, Sequence]
        aggregated = nn.functional.softmax(aggregated, 1)
        aggregated = torch.einsum(
            "bs,bsa,bza->bza", aggregated, val, out
        )  # Shape: [Batch, Sequence, AttentionFeatures]

        segment_max_pooled = seg.view(
            batch, sequence // self.segments, self.segments, -1
        )
        segment_max_pooled = segment_max_pooled.amax(
            2, keepdim=True
        )  # Shape: [Batch, PooledSequence, 1, AttentionFeatures]
        segment_max_pooled = segment_max_pooled * out.view(
            batch, sequence // self.segments, self.segments, -1
        )  # Shape: [Batch, PooledSequence, PoolSize, AttentionFeatures]
        segment_max_pooled = segment_max_pooled.view(
            batch, sequence, -1
        )  # Shape: [Batch, Sequence, AttentionFeatures]

        loc = loc.transpose(1, 2)  # Shape: [Batch, AttentionFeatures, Sequence]
        local_max_pooled = nn.functional.max_pool1d(
            loc, self.max_pool_kernel, 1, self.max_pool_kernel // 2
        )
        local_max_pooled = local_max_pooled.transpose(
            1, 2
        )  # Shape: [Batch, Sequence, AttentionFeatures]

        return aggregated + segment_max_pooled + local_max_pooled


class T_Layer(nn.Module):
    # https://github.com/huggingface/transformers/blob/1c220ced8ecc5f12bc979239aa648747411f9fc4/src/transformers/activations.py#L37
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.ModuleList(
            [
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                for _ in range(2)
            ]
        )
        self.Attention = PoolingAttention(
            config.hidden_size,
            config.hidden_size,
            config.segment_size,
            config.local_kernel_size,
        )
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

        self.dense_0 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.dense_1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.dense_2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.activation = transformers.activations.gelu_new()
        self.dropout = nn.ModuleList(
            [nn.Dropout(config.hidden_dropout_prob) for _ in range(3)]
        )

        self.rezero = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden_states):
        dropout = iter(self.dropout)
        LayerNorm = iter(self.LayerNorm)

        stream = x = hidden_states
        x = next(LayerNorm)(x)
        x = self.Attention(x)
        x = self.output(x)
        x = next(dropout)(x)
        stream = x = stream + x * self.rezero

        x = next(LayerNorm)(x)
        x = self.activation(self.dense_0(x)) * self.dense_1(x)
        x = next(dropout)(x)
        x = self.dense_2(x)
        x = next(dropout)(x)
        stream = x = stream + x * self.rezero

        return (stream,)


class T_Encoder(nn.Module):
    def __init__(self, config):
        # take only model config
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [T_Layer(config) for _ in range(config.num_hidden_layers)]
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        for i, layer_module in enumerate(self.layer):
            hidden_states, _ = layer_module(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


@torch.no_grad()
def Add_Encoding(data):
    length = data.shape[-2]
    channel = data.shape[-1]
    position = torch.arange(length, dtype=data.dtype, device=data.device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, channel, 2, dtype=data.dtype, device=data.device)
        * (-math.log(10000.0) / channel)
    )
    x = position * div_term
    data[..., 0::2] += torch.sin(x)
    data[..., 1::2] += torch.cos(x)


class T_Embedding(nn.Module):
    def __init__(self, config):
        # take only model config
        super().__init__()
        self.te = None
        in_channels = config.hidden_size
        self.LayerNorm = nn.LayerNorm(in_channels, eps=config.layer_norm_eps)
        self.projection = nn.Linear(in_channels, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input):
        if self.te == None or self.te.size(0) > input.size(1):
            size = input.size(1)
            size = 2 ** math.ceil(math.log2(size))
            self.te = torch.zeros(
                size, input.size(2), dtype=input.dtype, device=input.device
            )
            Add_Encoding(self.te)

        x = input + self.te[: input.size(1)]

        x = self.LayerNorm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class T_Module(nn.Module):
    def __init__(self, config):
        # config contains only model.transformers
        super().__init__()
        self.embedding = T_Embedding(config)
        self.encoder = T_Encoder(config)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=module.in_features**-0.5)
            if hasattr(module, "bias"):
                module.bias.data.zero_()

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return x


def get_embedding(a_sqrt, gamma, scale):
    X = (
        a_sqrt[..., None]
        * 10
        ** (
            -torch.linspace(0, 63, 64, dtype=a_sqrt.dtype, device=a_sqrt.device) / gamma
        )
        * scale
    )
    return torch.cat([torch.sin(X), torch.cos(X)], dim=-1)


class Alpha_weight_Embedding(nn.Module):
    def __init__(self, out_shape, divide, **embedding_config):
        self.out_shape = out_shape
        self.divide = divide
        self.embedding_config = embedding_config
        pos_ch = 128
        emb_ch = 512
        out_features = np.prod(out_shape)
        self.weight = nn.ModuleList(
            [
                nn.Linear(pos_ch, emb_ch, bias=True),
                nn.Linear(emb_ch, emb_ch, bias=True),
                nn.Linear(emb_ch // divide, out_features // divide, bias=True),
            ]
        )
        self.weight[-1].bias.zero_()
        self.normal = nn.ModuleList(
            [
                nn.LayerNorm(emb_ch),
                nn.LayerNorm(emb_ch),
                nn.LayerNorm(out_features),
            ]
        )
        self.normal[-1].weight.zero_()
        torch.nn.init.orthogonal_(self.normal[-1].bias.view(out_shape))

    def forward(self, input):
        WIGT = iter(self.weight)
        NORM = iter(self.normal)
        x = get_embedding(input, self.embedding_config)

        x = next(WIGT)(x)
        x = nn.functional.silu(x)
        x = next(NORM)(x)
        x = next(WIGT)(x)
        x = nn.functional.silu(x)
        x = next(NORM)(x)

        W = next(WIGT)
        x = torch.cat([W(I) for I in x.chunk(self.divide, -1)], axis=-1)
        x = next(NORM)(x)
        x = x.reshape(self.out_shape)

        return x


class G_mapping(nn.Module):
    def __init__(self, in_features, out_features):
        self.weight = nn.Linear(in_features, out_features * 2, bias=True)
        self.weight.bias.zero_()

    def forward(self, input):
        x = input
        x = self.weight(x)
        μ, σ = x.chunk(2, -1)
        return μ, torch.sigmoid(σ)


class Model(nn.Module):
    def __init__(self, config):
        # get full config

        self.config = config.model
        super().__init__()
        hidden_size = config.transformers.kwargs.hidden_size
        io_size = config.channels * config.f_size
        self.AW_EMB = Alpha_weight_Embedding(
            (hidden_size, io_size),
            config.devide,
            {"gamma": config.gamma, "scale": config.scale},
        )
        self.transformer = T_Module(self.config.transformers.kwargs)
        self.Gaussian_mapping = G_mapping(hidden_size, io_size)

        if self.config.dtype:
            self.AW_EMB.type(self.config.dtype)
            self.Gaussian_mapping.type(self.config.dtype)
        if self.config.transformers.dtype:
            self.transformer.type(self.config.transformers.dtype)

    def forward(self, input, t) -> Tuple[torch.Tensor, torch.Tensor]:
        # input: [B, T, F, C]

        x = input.view(*input.shape[:2], -1)
        t = t.type(self.type())

        weight = self.AW_EMB(t)
        x = nn.functional.linear(x, weight)
        x = self.transformer(x)
        x = self.Gaussian_mapping(x)

        return [I.view(*input.shape) for I in x]
