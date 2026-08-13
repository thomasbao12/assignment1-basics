import einops
import math
import torch

class Linear(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    (
                        out_features,
                        in_features,
                    ),
                    dtype=dtype,
                    device=device
                ),
                mean = 0,
                std=math.sqrt(2 / (in_features + out_features)),
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(
            x,
            self.weight,
            "... d_in, d_out d_in -> ... d_out"
        )