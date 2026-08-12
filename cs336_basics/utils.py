from jaxtyping import Bool, Float, Int
from torch import Tensor

import einops
import numpy as np
import numpy.typing as npt
import os
import torch
from typing import IO, Any, BinaryIO

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    n = len(dataset)
    indices = np.random.randint(low = 0, high = n - context_length - 1, size = batch_size)
    input_sequences = torch.stack([
        torch.from_numpy(dataset[i : i + context_length])
        for i in indices
    ]).to(device)
    output_sequences = torch.stack([
        torch.from_numpy(dataset[i + 1 : i + context_length + 1])
        for i in indices
    ]).to(device)
    return (input_sequences, output_sequences)

def init_random_weights(dim_rows, dim_cols, dtype, device) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.nn.init.trunc_normal_(
            torch.empty(
                dim_rows,
                dim_cols,
                dtype = dtype,
                device = device,
            )
        )
    )

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    # subtract max to avoid numerical stability issues caused by exp(x) = inf
    centered = in_features - in_features.max(dim=dim, keepdim=True).values
    exponentiated = centered.exp()
    return exponentiated / exponentiated.sum(dim=dim, keepdim=True)

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None, # default is causal mask
    device: torch.device | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    scaled_dot = einops.einsum(
        Q,
        K,
        "... queries d_k, ... keys d_k -> ... queries keys"
    ) / d_k ** 0.5
    if mask is None:
        ones = torch.ones(scaled_dot.shape, dtype=torch.bool, device = device)
        mask = torch.tril(ones)

    masked = torch.where(
        mask,
        scaled_dot,
        torch.tensor(
            float("-inf")
        )
    )
    softmax = run_softmax(masked, -1)
    return einops.einsum(
        softmax,
        V,
        "... queries keys, ... keys d_v -> ... queries d_v"
    )

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # subtract max to avoid numerical stability issues caused by exp(x) = inf
    centered_probabilities = inputs - inputs.max(dim=-1, keepdim=True).values
    target_logits = torch.gather(
        centered_probabilities,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = -target_logits + centered_probabilities.exp().sum(dim=-1).log()
    return losses.mean()

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    obj = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(obj, out)

def load_model_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module
):
    obj = torch.load(src)
    model.load_state_dict(obj["model_state_dict"])

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    obj = torch.load(src)
    load_model_checkpoint(src, model)
    optimizer.load_state_dict(obj["optimizer_state_dict"])
    return obj["iteration"]