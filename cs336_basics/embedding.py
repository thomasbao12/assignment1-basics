import torch

class Embedding(torch.nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    (
                        num_embeddings,
                        embedding_dim,
                    ),
                    dtype=dtype,
                    device=device
                )
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights[x]
