import einops
import torch

from jaxtyping import Float
from torch import Tensor

class RoPE(torch.nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_length: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_length = max_seq_length
        self.device = device        
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        k = torch.arange(0, self.d_k // 2, device = self.device)
        frequencies = 1 / self.theta ** (2 * k / self.d_k) # shape(pairs)
        angles = token_positions[:, None] * frequencies[None, :] # shape(seq, pairs)
        cos = angles.cos() # shape(seq, pairs)
        sin = angles.sin() # shape(seq, pairs)
        rotation_matrices = torch.stack(
            [
                torch.stack([cos, -sin], dim = -1),
                torch.stack([sin, cos], dim = -1)
            ],
            dim = -2
        ) # shape(seq, pairs, rows, cols) rows = 2 cols = 2
        x_pairs = einops.rearrange(
            x, 
            "... seq (pairs two) -> ... seq pairs two",
            two = 2
        ) 
        rotated_x_pairs = einops.einsum(
            x_pairs,
            rotation_matrices,
            "... seq pairs two, seq pairs rows two  -> ... seq pairs rows"
        )
        return einops.rearrange(
            rotated_x_pairs,
            "... pairs two -> ... (pairs two)"
        )