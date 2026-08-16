import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE
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
    ):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, dtype = dtype, device = device)
        self.k_proj = Linear(d_model, d_model, dtype = dtype, device = device)
        self.v_proj = Linear(d_model, d_model, dtype = dtype, device = device)
        self.output_proj = Linear(d_model, d_model, dtype = dtype, device = device)
        
        if theta is not None:
            self.rope = RoPE(theta, d_model / num_heads, max_seq_len, device = device)
        else:
            self.rope = None
    
    def forward(
        self, 
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        num_heads = self.num_heads

        token_positions_with_head_dimension = einops.rearrange(
            token_positions,
            "... sequence_length -> ... 1 sequence_length",
        )

        q = self.q_proj.forward(in_features)
        multihead_q = einops.rearrange(
            q,
            "... sequence_length (num_heads d_q) -> ... num_heads sequence_length d_q",
            num_heads = num_heads
        )
        
        k = self.k_proj.forward(in_features)
        multihead_k = einops.rearrange(
            k,
            "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k",
            num_heads = num_heads
        )

        v = self.v_proj.forward(in_features)
        multihead_v = einops.rearrange(
            v,
            "... sequence_length (num_heads d_v) -> ... num_heads sequence_length d_v",
            num_heads = num_heads
        )
        
        if self.rope is not None:
            rotated_multihead_q = self.rope.forward(
                    multihead_q,
                    token_positions_with_head_dimension,
                )
            rotated_multihead_k = self.rope.forward(
                multihead_k,
                token_positions_with_head_dimension,
            )
            multihead_attention = utils.run_scaled_dot_product_attention(
                rotated_multihead_q, rotated_multihead_k, multihead_v, device=self.device)
        else:
            multihead_attention = utils.run_scaled_dot_product_attention(
                multihead_q, multihead_k, multihead_v, device=self.device)
        
        
        rearranged_multihead_attention = einops.rearrange(
            multihead_attention,
            "... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)"
        )
        return self.output_proj.forward(rearranged_multihead_attention)