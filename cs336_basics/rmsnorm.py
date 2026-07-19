import einops
import torch

class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.weights = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    (
                        d_model,
                    ),
                    dtype=dtype,
                    device=device
                )
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_detype = x.dtype
        x_upcast = x.to(torch.float32)
        rms = (x_upcast.square().mean(dim=-1, keepdim = True) + self.eps).sqrt()
        return einops.einsum(
            (x_upcast / rms).to(in_detype),
            self.weights,
            "... d_model, d_model -> ... d_model"
        )