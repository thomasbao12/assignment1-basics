# Post-Norm Transformer

## Setup

The pre-norm Transformer was changed to a post-norm Transformer by applying
RMSNorm after each residual addition. The post-norm model was trained for 1,000
iterations at the same settings as the existing cosine-scheduling pre-norm
control:

- Learning rate: `1e-3`
- Batch size: `128`
- `beta2`: `0.995`
- Warmup: `100` steps
- Cosine cycle: `10000` steps
- Minimum learning rate: `1e-4`

The post-norm run used the same model and dataset dimensions as the control.

## Results

The combined training- and validation-loss curves are shown in
[post_norm_vs_pre_norm_loss_curves.svg](post_norm_vs_pre_norm_loss_curves.svg).

| Step | Pre-norm validation loss | Post-norm validation loss |
| ---: | ---: | ---: |
| 0 | 9.253 | 9.248 |
| 100 | 3.125 | 3.036 |
| 200 | 2.487 | 2.440 |
| 300 | 2.215 | 2.230 |
| 400 | 2.077 | 1.996 |
| 500 | 1.967 | 1.974 |
| 600 | 1.929 | 1.821 |
| 700 | 1.825 | 1.841 |
| 800 | 1.808 | 1.818 |
| 900 | 1.847 | 1.827 |

Both models trained smoothly without large loss spikes. The post-norm model
was slightly better at the final validation checkpoint (`1.827` versus
`1.847`), but the curves are close overall.

## Conclusion

For this 1,000-step run, changing from pre-norm to post-norm did not cause the
instability observed in the no-RMSNorm experiment. With warmup and the matched
learning-rate schedule, post-norm training was stable and achieved validation
loss comparable to the pre-norm control. This result is specific to the model
size, optimizer, schedule, and short training horizon; it does not establish
that post-norm is equally stable in deeper or longer runs.

The pre-norm and post-norm runs were not made with an explicitly fixed random
seed, so small differences in their curves should not be overinterpreted.
