# RMSNorm Ablation Hypothesis

## Hypothesis

Removing RMSNorm will make optimization less stable. At the previously
optimal learning rate, I expect validation loss to oscillate, plateau, or
diverge rather than decrease smoothly.

I also expect residual-stream magnitudes to increase with Transformer depth
and possibly grow over training steps. Lowering the learning rate may improve
stability, but may not fully recover the normalized model's performance.

## Measurements

- Training and validation loss versus step
- Residual activation norm after each Transformer block
- Residual activation norm versus training step
- Gradient norm, if available

## Controls

Keep the model architecture, dataset, batch size, optimizer, schedule, training
duration, and random seed fixed across runs. Compare the original model at its
optimal learning rate against the no-RMSNorm model at that same rate and at
lower rates.
