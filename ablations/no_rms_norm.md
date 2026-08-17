# RMSNorm Ablation

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

## Initial Findings

The no-RMSNorm model was trained for 1,000 iterations at the previous optimal
learning rate of `1e-3`, with batch size `128` and the existing cosine schedule.
Validation loss decreased from `13.28` at step 0 to approximately `1.93` by step
800.

The run showed intermittent training-loss spikes rather than complete
divergence. Training loss reached `9.53` at step 850 and `157017.625` at step
920, then recovered to approximately `1.90` by the end of the run. Validation
loss was approximately `2.00` at step 900.

Mean layer-output norms were strongly depth-dependent, especially at
initialization:

| Layer | Step 0 | Step 990 |
| --- | ---: | ---: |
| 0 | 21.5 | 15.9 |
| 1 | 24.3 | 17.0 |
| 2 | 33.1 | 21.7 |
| 3 | 193.1 | 50.9 |

Contrary to the original hypothesis, the mean activation norms decreased over
training. This suggests that optimization can reduce the average activation
scale even without RMSNorm. The remaining loss spikes may be caused by rare
activation or gradient outliers that are hidden by the mean norm statistic.

These findings support the claim that RMSNorm affects stability and activation
scale, but they do not establish that removing it causes monotonic activation
growth or long-run divergence.

## Lower Learning Rate

The no-RMSNorm model was also trained for 1,000 iterations at `3e-4`, using
the same batch size, optimizer, schedule, and model configuration. The lower
learning rate produced a smoother trajectory:

| Learning rate | Validation loss at step 800 | Validation loss at step 900 | Maximum training loss |
| --- | ---: | ---: | ---: |
| `1e-3` | 1.93 | 2.00 | 157018 |
| `3e-4` | 2.13 | 2.04 | 13.2 |

The `3e-4` run did not show the large transient training-loss spikes seen at
`1e-3`, although it learned more slowly and had slightly worse validation loss
by step 900. It is therefore the best stable lower-learning-rate candidate
tested so far.

Lowering the learning rate did not simply reduce the final mean activation
norms. At step 990, the per-layer norms for `1e-3` were approximately
`[15.9, 17.0, 21.7, 50.9]`, while the norms for `3e-4` were approximately
`[18.7, 20.9, 28.9, 94.9]`. This reinforces that average activation scale and
optimization stability are related but distinct measurements.

Overall, these 1,000-step runs suggest that RMSNorm improves robustness to a
larger learning rate. Removing it does not force monotonic activation growth
or immediate divergence, but it can produce intermittent large loss spikes at
the previous optimal learning rate. A lower learning rate improves stability
at the cost of slower optimization. The runs used gradient clipping with a
maximum norm of `1.0` and did not use an explicitly fixed random seed, so the
results should be interpreted as an early optimization comparison rather than
a definitive long-run performance ranking.

## Learning Curves

The combined validation- and training-loss comparison is shown in
[no_rms_norm_loss_curves.svg](no_rms_norm_loss_curves.svg). The validation-only
view is also available in
[no_rms_norm_learning_curves.svg](no_rms_norm_learning_curves.svg).

## Conclusion

Removing RMSNorm did not prevent the Transformer from learning, but it made the
previously optimal learning rate of `1e-3` less stable. That run had large
transient training-loss spikes, while reducing the learning rate to `3e-4`
produced a smoother, more stable trajectory at the cost of slower optimization
and slightly higher validation loss after 1,000 iterations. RMSNorm therefore
appears to improve optimization robustness and reduce sensitivity to learning
rate, even though the average activation norms in this experiment decreased
over training rather than growing monotonically.
