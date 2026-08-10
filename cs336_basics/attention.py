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
        num_heads: int,
        max_seq_len: int,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        super().__init__()
        self.num_heads = num_heads
        self.q_proj_weight = utils.init_random_weights(d_model, d_model, dtype, device)
        self.k_proj_weight = utils.init_random_weights(d_model, d_model, dtype, device)
        self.v_proj_weight = utils.init_random_weights(d_model, d_model, dtype, device)
        self.o_proj_weight = utils.init_random_weights(d_model, d_model, dtype, device)
        if theta is not None:
            self.rope = RoPE(theta, d_model / num_heads, max_seq_len, device)
        else:
            self.rope = None
    
    def forward(
        self, 
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        num_heads = self.num_heads

        multihead_token_positions = einops.repeat(
            token_positions,
            "... sequence_length -> ... num_heads sequence_length",
            num_heads=num_heads
        )

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
        
        if self.rope is not None:
            rotated_multihead_q = self.rope.forward(
                    multihead_q,
                    multihead_token_positions,
                )
            rotated_multihead_k = self.rope.forward(
                multihead_k,
                multihead_token_positions,
            )
            multihead_attention = utils.run_scaled_dot_product_attention(rotated_multihead_q, rotated_multihead_k, multihead_v)
        else:
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