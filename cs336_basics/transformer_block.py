import torch
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.attention import Attention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU

class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = Attention(
            d_model,
            num_heads,
            max_seq_len,
            theta,
            device=device,
            dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(
        self, 
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        
        mha = self.attn.forward(
            in_features,
            token_positions
        )
        rms_norm1 = self.ln1.forward(in_features + mha)
        ff = self.ffn.forward(
            rms_norm1
        )
        return self.ln2.forward(in_features + mha + ff)