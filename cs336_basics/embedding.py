import torch
import cs336_basics.utils as utils

class Embedding(torch.nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = utils.init_random_weights(num_embeddings, embedding_dim, dtype, device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]