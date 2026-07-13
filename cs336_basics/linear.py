import einops
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
        self.weights = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    (
                        out_features,
                        in_features,
                    ),
                    dtype=dtype,
                    device=device
                )
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(
            x,
            self.weights,
            "... d_in, d_out d_in -> ... d_out"
        )