from cs336_basics.linear import Linear

import cs336_basics.utils as utils
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1.forward(x)
        silu = utils.run_silu(w1_x)
        return self.w2.forward(silu)
