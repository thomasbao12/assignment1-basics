from cs336_basics.linear import Linear

import einops
import torch

class SwiGLU(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1.forward(x)
        silu = w1_x * torch.sigmoid(w1_x)
        w3_x = self.w3.forward(x)
        return self.w2.forward(silu * w3_x)
