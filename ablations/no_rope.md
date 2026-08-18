# No-RoPE Ablation

## Hypotheses

Before inspecting the logs, we predicted:

1. Validation loss would decrease somewhat more slowly and finish slightly
   above the matched RoPE control, rather than showing a dramatic degradation.
2. Training loss might show greater noise because the model lacks the
   positional signal provided by RoPE, but catastrophic instability was not
   expected.
3. The model would still learn over 1,000 steps rather than immediately
   diverging.

## Setup

- Learning rate: `1e-3`
- Batch size: `128`
- `beta2`: `0.995`
- Warmup: `100` steps
- Cosine cycle: `10000` steps
- Minimum learning rate: `1e-4`
- Training duration: `5,000` iterations for the extended comparison
- RoPE: disabled by omitting `rope_theta`

The initial run writes logs to `no_rope_lr1e-3_1k`. The extended no-RoPE run
writes logs to `no_rope_lr1e-3_5k_final`; the downloaded CSV logs are in the
corresponding directories under `assignment1/logging/`.

## Initial Findings

The no-RoPE model trained normally over the 1,000-step run. Validation loss
fell from `9.27` at step 0 to `1.88` at step 900. This is only slightly worse
than the matched RoPE controls, which were approximately `1.83-1.85` at the
same stage. The short run therefore does not show a large validation-loss
penalty or a clear higher plateau.

The no-RoPE run also did not show catastrophic loss spikes. The main result is
that RoPE provided a modest early-training advantage in this configuration,
not that it was necessary for stability. A longer run or a task requiring
stronger positional generalization would be needed to test whether the small
gap widens over time.

## 5k Comparison

To test whether the small gap widened, both configurations were trained for
5,000 iterations with the same optimizer and schedule. The no-RoPE run
completed without catastrophic instability and reached validation loss `1.548`
at step 4,900. The matched RoPE control reached `1.450` at the same step.

| Step | No RoPE | RoPE | Difference |
| ---: | ---: | ---: | ---: |
| 1,000 | 1.937 | 1.796 | +0.140 |
| 2,000 | 1.717 | 1.625 | +0.092 |
| 3,000 | 1.546 | 1.541 | +0.005 |
| 4,000 | 1.536 | 1.483 | +0.053 |
| 4,900 | 1.548 | 1.450 | +0.098 |

The longer run provides evidence for a modest RoPE advantage, especially late
in training, but not for a qualitative failure of the no-RoPE model. Because
the runs did not use an explicitly fixed random seed, the exact differences
should not be treated as a precise estimate of RoPE's effect.
