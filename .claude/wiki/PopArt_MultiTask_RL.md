---
type: algorithm_reference
tags: [algorithm, reinforcement-learning, multitask, normalization, value-function, concept]
source: https://arxiv.org/abs/1809.04474
date_ingested: 2026-07-09
---

# PopArt_MultiTask_RL

An output-normalisation method for multi-task actor-critic RL that automatically rescales each task's value-function head so all tasks have equal influence on the shared policy gradient — without manual reward engineering or loss weighting. First agent to exceed human-level median performance across all 57 Atari games with a single policy.

**Reference:** Hessel, Soyer, Espeholt, Czarnecki, Schmitt, van Hasselt. "Multi-task Deep Reinforcement Learning with PopArt." AAAI 2019. arXiv:1809.04474.

## Formulation

### Two components: POP + ART

**ART (Adaptive Rescaling of Targets):** Maintain per-task running statistics `(μ_i, σ_i)` and normalise targets before computing value loss. The value network outputs a normalised prediction `n(s)`.

**POP (Preserve Outputs Precisely):** When statistics update, immediately adjust the last-layer weights so the network's unnormalised output is unchanged — preventing target non-stationarity from corrupting learning.

### Single-task equations (foundation)

```
# Value parameterisation:
v(s) = σ · n(s) + μ           # unnormalised value = scale * net output + shift

# Running statistics (exponential moving average, decay β):
μ_t = (1-β)·μ_{t-1} + β·G_t^v
ν_t = (1-β)·ν_{t-1} + β·(G_t^v)²
σ_t = sqrt(ν_t - μ_t²)

# Normalised actor-critic updates:
Δθ ∝ ((G_t^v - μ)/σ - n(s_t)) · ∇_θ n(s_t)       # critic gradient
Δη ∝ ((G_t^π - μ)/σ - n(s_t)) · ∇_η log π(a|s)    # actor gradient

# POP — output-preserving weight update after statistics change:
w' = (σ/σ') · w
b' = (σ·b + μ - μ') / σ'
```

Applied after every statistics update so `σ'·w'·x + b' = σ·w·x + b` for all x.

### Multi-task extension

```
# Per-task statistics vectors (N tasks):
μ ∈ ℝ^N,  σ ∈ ℝ^N

# Vector-valued value head (N task-specific outputs):
v(s) = σ ⊙ n(s) + μ          # ⊙ = element-wise

# Per-task normalised updates (only task-i head active for task-i transition):
Δθ ∝ ((G_t^{v,i} - μ_i)/σ_i - n_i(s_t)) · ∇_θ n_i(s_t)
Δη ∝ ((G_t^{π,i} - μ_i)/σ_i - n_i(s_t)) · ∇_η log π(a|s)
```

The **policy `η` is shared and task-agnostic**; only value heads are task-specific. The policy gradient is automatically balanced because all value estimates are normalised to the same scale before computing advantages.

## Parameters

| Parameter | Value | Role |
|---|---|---|
| Statistics decay β | 3×10⁻⁴ | EMA decay for μ, σ. Slow enough to track non-stationarity, fast enough to react to scale changes. Almost never needs tuning. |
| σ lower bound | 0.0001 | Numerical stability clip |
| σ upper bound | 1e6 | Numerical stability clip |
| Discount γ | 0.99 | Standard |
| Unroll length | 20 (Atari) / 100 (DmLab) | V-trace rollout |
| Batch size | 32 | Standard |
| Optimizer | RMSProp (momentum=0) | PBT-tuned lr, epsilon |

PBT-tuned: learning rate ∈ [5e-6, 5e-3], entropy cost ∈ [5e-5, 1e-2], grad norm clip ∈ [10, 100].

## Architecture (IMPALA backbone)

```
Conv stack:
  Section 1: 16 channels, 2 ResNet blocks (3×3 conv)
  Section 2: 32 channels, 2 ResNet blocks
  Section 3: 32 channels, 2 ResNet blocks
  → ReLU → FC(256) → [LSTM(256) for DmLab only]

Output heads:
  Policy: FC(num_actions), softmax   # shared across all tasks
  Value:  N × FC(1)                  # one normalised head per task
  PopArt layer: applied to value heads only
```

## Implementation Notes

**Per-update procedure:**

1. Sample batch of (s, a, r, G^v, G^π, task_id) tuples.
2. Compute actor-critic losses using normalised targets: `(G_t^{v,i} - μ_i)/σ_i`.
3. Backpropagate gradient through value heads (task-specific) and policy (shared).
4. Update statistics: `μ_i, ν_i` via EMA for each task i observed in the batch.
5. Recompute `σ_i = sqrt(ν_i - μ_i²)`, clip to [0.0001, 1e6].
6. Apply POP weight update: `w_i' = (σ_i/σ_i') · w_i`, `b_i' = (σ_i·b_i + μ_i - μ_i')/σ_i'`.
7. **No gradient flows into μ/σ** — they are updated by running average only.

**V-trace compatibility:** Off-policy importance sampling corrections in V-trace apply to targets before normalisation. No modification needed — PopArt wraps around any target computation.

**Where statistics are stored:** μ and σ are parameters of the final linear layer (last-layer bias and scale), not part of the gradient graph.

## When to Use

- **Use PopArt when:** Tasks have very different return scales or densities (exactly ariel's situation — hover's cumulative return is ~0.0125/step × 600 steps ≈ 7.5; figure8 gate-passing can give returns of ±50+).
- **Especially when:** Reward clipping is undesirable (PopArt makes clipping unnecessary by normalising internally).
- **Comparison to reward-side normalisation** (ariel's `_PerTaskRewardNormalizer`): PopArt operates in the **value head** (output space), not the reward/return stream. See §In Ariel below.
- **Not needed when:** All tasks have similar reward scales — plain multi-task with shared policy suffices.

## Key Results

### Atari-57 (median human-normalised score)

| Agent | Clipped rewards | Unclipped rewards |
|---|---|---|
| IMPALA | 59.7% | 0.3% |
| MultiHead-IMPALA (no PopArt) | ~55% | ~0.2% |
| **PopArt-IMPALA** | **110.7%** | **107.0%** |

The near-zero IMPALA score without reward clipping confirms that native reward scales completely dominate naive multi-task training. PopArt is almost unaffected (110% → 107%) — demonstrating true scale invariance.

### DmLab-30

| Agent | Train | Test |
|---|---|---|
| IMPALA (improved) | ~65% | ~63% |
| **PopArt-IMPALA** | **73.5%** | **72.8%** |

### Ablation: capacity vs. normalisation

MultiHead-IMPALA adds N value heads without PopArt normalisation → performs *slightly worse* than vanilla IMPALA. This confirms that the gain is from **adaptive rescaling, not added capacity**.

## In Ariel

### Current approach vs. PopArt

| Aspect | Ariel `_PerTaskRewardNormalizer` | PopArt |
|---|---|---|
| Where normalisation happens | Reward/return stream (input to value loss) | Value head output layer |
| Statistics tracked | Welford running discounted-return std per task | EMA of μ and ν = E[G²] per task |
| POP weight correction | ❌ not implemented | ✅ prevents output shift when stats change |
| Policy gradient normalisation | Indirect (through normalised advantage) | Direct (advantage computed from normalised head) |
| Implementation complexity | Low (pre-processing) | Medium (modify last layer + weight correction) |
| Works with SB3 PPO | ✅ (current) | Requires custom policy class |

**Ariel's `_PerTaskRewardNormalizer`** computes a Welford running standard deviation of discounted returns per task and divides rewards by that std before they enter the value loss. This is equivalent to ART (the target normalisation step) without POP (the output-preserving weight correction). It's a valid and simpler implementation of the same idea.

**When to upgrade to full PopArt:** If the Stage-3 training run shows that per-task value functions diverge (critic loss spikes when return scale of a task changes — e.g. after the residual starts successfully passing gates and rewards jump from near-0 to +1 spikes), add the POP weight correction. Without POP, sudden statistics changes can temporarily corrupt the value head's outputs.

### Implementation path in SB3

To add PopArt to `37_train_residual_mtrl.py`:

1. Add per-task `(mu_i, nu_i)` buffers alongside `_PerTaskRewardNormalizer`.
2. After each PPO epoch, update statistics from the rollout buffer's per-task returns.
3. Before statistics update, save `(mu_old, sigma_old)` per task.
4. Apply POP correction to last linear layer of value head: `w = (sigma_old/sigma_new) * w`.
5. Normalise PPO value targets with `(G - mu_i)/sigma_i` instead of dividing by running std.

The key code change is in the value head's last `nn.Linear` layer — wrap it in a `PopArtLinear` that stores `(mu, sigma)` and applies weight correction on statistics update.

## See Also

- [[multitask_gradient_interference]] — complementary problem (gradient direction conflict vs. scale imbalance)
- [[PCGrad_Gradient_Surgery]] — gradient direction fix; PopArt is the scale fix
- [[Residual_Policy_Learning]] — ariel's MTRL context where PopArt would be applied
