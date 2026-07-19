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
        self.w1 = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    (
                        d_ff,
                        d_model,
                    ),
                    dtype=dtype,
                    device=device
                )
            )
        )

        self.w2 = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    (
                        d_model,
                        d_ff,
                    ),
                    dtype=dtype,
                    device=device
                )
            )
        )

        self.w3 = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    (
                        d_ff,
                        d_model,
                    ),
                    dtype=dtype,
                    device=device
                )
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einops.einsum(
            x,
            self.w1,
            "... d_model, d_ff d_model -> ... d_ff"
        )
        silu = w1_x * torch.sigmoid(w1_x)
        w3_x = einops.einsum(
            x,
            self.w3,
            "... d_model, d_ff d_model -> ... d_ff"
        )
        return einops.einsum(
            silu * w3_x,
            self.w2,
            "... d_ff, d_model d_ff -> ... d_model"
        )
