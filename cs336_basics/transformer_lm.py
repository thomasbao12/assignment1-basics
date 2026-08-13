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

from cs336_basics.rope import RoPE
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.transformer_block import TransformerBlock
import cs336_basics.utils as utils

class TransformerLM(torch.nn.Module):

    def __init__(
        self,
        *,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.token_embeddings = Embedding(
            vocab_size,
            d_model,
            device = device,
            dtype = dtype,
        )
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, rope_theta, device = device, dtype=dtype
            ) for layer in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device = device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device = device, dtype=dtype)
        
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        embedding = self.token_embeddings.forward(x)
        layer_input = embedding
        for layer in self.layers:
            layer_input = layer.forward(layer_input, token_positions)
        last_norm = self.ln_final.forward(layer_input)
        return self.lm_head.forward(last_norm)