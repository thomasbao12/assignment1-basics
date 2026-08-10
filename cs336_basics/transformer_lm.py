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
import cs336_basics.utils as utils

class TransformerLM(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ):
        super().__init__()

        self.embedding_layer = Embedding(
            vocab_size,
            d_model,
        )
        self.embedding_layer.load_state_dict({
                'weights': weights["token_embeddings.weight"]
        })
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer.forward(x)