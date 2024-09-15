"""
Implements the Qwen Vision Encoder.
"""

import dataclasses
from functools import partial
import logging
from typing import Any, Dict, Tuple

from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Module, Tensor
from tvm.relax.frontend.nn.modules import Conv3D
from tvm.relax.frontend.nn.op import (
    broadcast_to,
    concat,
    permute_dims,
    reshape,
    wrap_nested,
    split,
    squeeze
)
from tvm.relax.op import arange

from mlc_llm import op as op_ext
from mlc_llm.support.config import ConfigBase

logger = logging.getLogger(__name__)

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.silu,
    "swish": nn.silu,
    "gelu_new": partial(nn.gelu, approximate=True),
}

class PatchEmbed(Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = Conv3D(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)


    def forward(self, hidden_states: Tensor) -> Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = reshape(
                hidden_states,
            [-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size]
        )
        hidden_states = hidden_states.astype(dtype=target_dtype)
        hidden_states = self.proj(hidden_states)
        hidden_states = reshape(hidden_states, [-1, self.embed_dim])
        return hidden_states

class PatchMerger(Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp_0 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mlp_1 = ACT2FN["gelu"]
        self.mlp_2 = nn.Linear(self.hidden_size, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ln_q(x)
        x = reshape(x, [-1, self.hidden_size])
        x = self.mlp_0(x)
        x = self.mlp_1(x)
        return self.mlp_2(x)

class Qwen2VLAttention(Module):
    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden_states: Tensor):
        seq_length = hidden_states.shape[0]
        hidden_states = self.qkv(hidden_states)
        hidden_states = reshape(hidden_states, [seq_length, 3, self.num_heads, -1])
        hidden_states = permute_dims(hidden_states, axes=[1, 0, 2, 3])
        q, k, v = split(hidden_states, 3, axis=0)
        attn_output = op_ext.attention(q, k, v, None)
        attn_output = reshape( attn_output, [seq_length, -1])
        attn_output = self.proj(attn_output)
        return attn_output

class VisionMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str = "gelu") -> None:
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x) -> Tensor:
        x = self.fc1(x)
        x = self.act_fn(x)
        return self.fc2(x)

class Qwen2VisionBlock(Module):
    def __init__(self, config: ConfigBase, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        embed_dim = config.vision_config["embed_dim"]
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        mlp_hidden_dim = int(embed_dim * config.vision_config["mlp_ratio"])
        self.attn = Qwen2VLAttention(embed_dim, config.vision_config["num_heads"])
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        norm1_out = self.norm1(hidden_states)
        hidden_states = hidden_states + self.attn(norm1_out)
        norm2_out = self.norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(norm2_out)
        return hidden_states


class Qwen2VisionModel(Module):
    def __init__(self, config: ConfigBase):
        super().__init__()
        self.patch_embed = PatchEmbed(
            patch_size=config.vision_config["patch_size"],
            temporal_patch_size=config.vision_config["temporal_patch_size"],
            in_channels=config.vision_config["in_chans"],
            embed_dim=config.vision_config["embed_dim"],
        )

        self.blocks = nn.ModuleList(
            [Qwen2VisionBlock(config) for _ in range(config.vision_config["depth"])]
        )
        self.merger = PatchMerger(dim=config.vision_config["hidden_size"], context_dim = config.vision_config["embed_dim"])

    def forward(self, hidden_states: Tensor) -> Tensor:
        #hidden_states = self.patch_embed(hidden_states)
        for layer_id, layer in enumerate(self.blocks):
            hidden_states = layer(hidden_states)

        return self.merger(hidden_states)
