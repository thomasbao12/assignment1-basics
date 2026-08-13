from cs336_basics.linear import Linear

import einops
import torch

class SiLU(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1.forward(x)
        return w1_x * torch.sigmoid(w1_x)
