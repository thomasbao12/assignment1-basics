# SiLU vs. SwiGLU Ablation

## Setup

The SiLU run used `d_ff = 2016`, while the SwiGLU control used `d_ff = 1344`.
These widths approximately match the feed-forward parameter count because the
SiLU FFN uses two projections and the SwiGLU FFN uses three.

Both runs used:

- Learning rate: `1e-3`
- Batch size: `128`
- `beta2`: `0.995`
- Warmup: `100` steps
- Cosine cycle: `10000` steps
- Minimum learning rate: `1e-4`
- Same dataset and four-layer Transformer configuration

## Spot Check

The SiLU model was trained for 100 steps on Modal. It completed successfully,
with validation loss decreasing from `9.285` at step 0 and a checkpoint saved
at step 99.

## 5k Results

| Step | SiLU | SwiGLU | SiLU - SwiGLU |
| ---: | ---: | ---: | ---: |
| 0 | 9.258 | 9.294 | -0.036 |
| 1,000 | 1.796 | 1.796 | 0.000 |
| 2,000 | 1.667 | 1.625 | +0.042 |
| 3,000 | 1.558 | 1.541 | +0.017 |
| 4,000 | 1.506 | 1.483 | +0.023 |
| 4,900 | 1.443 | 1.450 | -0.007 |

The loss curves are shown in
[silu_vs_swiglu_5k_validation.svg](silu_vs_swiglu_5k_validation.svg).

Over this 5k-step run, SiLU and SwiGLU had very similar validation
trajectories. SwiGLU was modestly ahead through most intermediate checkpoints,
but SiLU finished slightly lower. This is an early optimization comparison,
not evidence of a robust ranking; the runs did not use an explicitly fixed
random seed.
