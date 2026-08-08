from collections.abc import Callable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params, 
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                
                grad = p.grad.data # Get the gradient of loss with respect to p.
                
                lr_t = lr * math.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1))

                p.data -= lr * weight_decay * p.data

                m = state.get("m", 0)
                m = beta1 * m + (1 - beta1) * grad
                state["m"] = m

                v = state.get("v", 0)
                v = beta2 * v + (1 - beta2) * grad * grad
                state["v"] = v

                p.data -= lr_t * m / (v.sqrt() + eps)
                
                state["t"] = t + 1 # Increment iteration number.

        return loss