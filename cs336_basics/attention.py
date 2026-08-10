import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import einops
import math
import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.adamw import AdamW
from cs336_basics.rope import RoPE
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe
from cs336_basics.transformer_lm import TransformerLM
import cs336_basics.utils as utils

class Attention(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        super().__init__()
        self.q_proj_weight = utils.init_random_weights(d_model, d_model, dtype, device)
        self.k_proj_weight = utils.init_random_weights(d_model, d_model, dtype, device)
        self.v_proj_weight = utils.init_random_weights(d_model, d_model, dtype, device)
        self.o_proj_weight = utils.init_random_weights(d_model, d_model, dtype, device)
    
    def forward(
        self, 
        num_heads,
        in_features: Float[Tensor, " ... sequence_length d_model"],
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        q = einops.einsum(
            in_features,
            self.q_proj_weight,
            "... sequence_length d_in, d_model d_in -> ... sequence_length d_model"
        )
        multihead_q = einops.rearrange(
            q,
            "... sequence_length (num_heads d_q) -> ... num_heads sequence_length d_q",
            num_heads = num_heads
        )
        k = einops.einsum(
            in_features,
            self.k_proj_weight,
            "... sequence_length d_in, d_model d_in -> ... sequence_length d_model"
        )
        multihead_k = einops.rearrange(
            k,
            "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k",
            num_heads = num_heads
        )
        v = einops.einsum(
            in_features,
            self.v_proj_weight,
            "... sequence_length d_in, d_model d_in -> ... sequence_length d_model"
        )
        multihead_v = einops.rearrange(
            v,
            "... sequence_length (num_heads d_v) -> ... num_heads sequence_length d_v",
            num_heads = num_heads
        )
        multihead_attention = utils.run_scaled_dot_product_attention(multihead_q, multihead_k, multihead_v)
        rearranged_multihead_attention = einops.rearrange(
            multihead_attention,
            "... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)"
        )
        return einops.einsum(
            rearranged_multihead_attention,
            self.o_proj_weight,
            "... sequence_length d_in, d_model d_in -> ... sequence_length d_model"
        )