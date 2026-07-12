---
type: algorithm_reference
tags: [algorithm, reinforcement-learning, multitask, optimization, gradient, concept]
source: https://arxiv.org/abs/2001.06782
date_ingested: 2026-07-09
---

# PCGrad_Gradient_Surgery

A model-agnostic multi-task learning optimizer that resolves destructive gradient interference between tasks by projecting each task's gradient onto the normal plane of any conflicting task gradient before the shared-parameter update. Adds zero hyperparameters; works with any underlying optimizer (Adam, SGD) and any architecture.

**Reference:** Yu, Kumar, Gupta, Levine, Hausman, Finn. "Gradient Surgery for Multi-Task Learning." NeurIPS 2020. arXiv:2001.06782.
**Code:** https://github.com/tianheyu927/PCGrad

## Formulation

### The "Tragic Triad" — three causes of gradient interference

```
# Definition 1: Conflicting gradients
cos(φ_ij) < 0       # task i and task j gradients point in opposing directions

# Definition 2: Gradient magnitude similarity
Φ(g_i, g_j) = 2‖g_i‖‖g_j‖ / (‖g_i‖² + ‖g_j‖²)
# = 1 when magnitudes equal; → 0 when magnitudes differ greatly

# Definition 3: Multi-task curvature
H(L; θ, θ') = ∫₀¹ ∇L(θ)ᵀ ∇²L(θ + a(θ'-θ)) ∇L(θ) da
```

### PCGrad projection (Algorithm 1)

```
# For each task i, iterate through other tasks j (random order):
for j in tasks \ {i}:
    if dot(g_i_PC, g_j) < 0:          # conflict detected
        g_i_PC = g_i_PC - (dot(g_i_PC, g_j) / ‖g_j‖²) * g_j

# Final update: sum of modified per-task gradients
g_update = Σ_i g_i_PC
```

The projection removes the component of `g_i` that directly opposes `g_j`, while leaving non-conflicting components intact. When `cos(φ_ij) ≥ 0`, `g_i` is unchanged.

### Convergence (Theorem 1)

Under convex assumptions with L-smooth gradients and step size `t ≤ 1/L`, PCGrad converges to either the optimal value or a critical point where `cos(φ₁₂) = -1` (full opposition, which cannot be improved further).

### When PCGrad strictly improves on standard gradient descent (Theorem 2)

```
Condition (a): cos φ₁₂ ≤ -Φ(g₁, g₂)         # conflict severe enough
Condition (b): curvature H ≥ ξ(g₁, g₂) · L
Condition (c): step size t ≥ 2 / (H - ξ(g₁,g₂)·L)

# Curvature bound:
ξ(g₁, g₂) = (1 - cos²φ₁₂) · ‖g₁-g₂‖² / ‖g₁+g₂‖²
```

These three conditions directly correspond to the three components of the tragic triad.

## Parameters

| Name | Default | Role |
|---|---|---|
| No new hyperparameters | — | PCGrad inherits all hyperparameters from the base optimizer |
| Random task ordering | per-iteration | Order in which tasks j are iterated for conflict resolution |
| Underlying optimizer | any | SGD, Adam, etc. — PCGrad modifies gradients before passing to optimizer |

## Implementation Notes

**Supervised learning:**
1. Sample minibatch; group by task into ℬ_i.
2. Compute per-task gradients `g_i = ∇L_i(θ)`.
3. Apply PCGrad projection (random ordering of j per task i).
4. Sum modified gradients → pass to optimizer.
5. No extra backward pass needed — pre-compute pairwise cosine similarities.

**Reinforcement learning (actor-critic):**
- Modify both actor and critic gradients with PCGrad.
- Compatible with off-policy (SAC) and on-policy methods.
- In ariel's `_PerTaskRewardNormalizer` + PPO setup: apply PCGrad to the shared actor/critic gradient before the PPO update step.

**Complexity:** O(N²) pairwise gradient comparisons per step (N = number of tasks). For N=5 tasks, this is 10 comparisons — negligible vs. forward/backward pass cost.

**Ordering randomness:** The random order in which tasks j are iterated matters because projections are applied sequentially (not simultaneously). In practice the variance is low for N ≤ 10.

## When to Use

- **Use PCGrad when:** Per-task reward curves diverge or oscillate in MTRL training — a symptom of gradient conflict.
- **Especially suited for:** Heterogeneous task losses on different scales (e.g. hover's continuous +0.0125/step shaping vs. figure8's sparse +1 gate spikes) that cause one task's gradient to dominate.
- **Complementary to per-task reward normalisation** (`_PerTaskRewardNormalizer` in ariel): normalisation addresses magnitude disparities (Condition 2 of the triad); PCGrad addresses directional conflict (Condition 1). Together they address all three tragic triad conditions.
- **Less useful when:** Tasks share complementary gradients (positive cosine similarity throughout training) — then PCGrad does nothing.

## Empirical Baseline Results

### Multi-Task RL (Meta-World)

| Method | MT10 success | MT50 success |
|---|---|---|
| Multi-head policy | <50% | — |
| Independent SAC (2M more steps) | ~same as PCGrad | — |
| **PCGrad + SAC** | **100% (all 10 tasks)** | **~70%** |
| SAC alone (equal budget) | substantially less | — |

On MT50: PCGrad+SAC achieves >30% absolute improvement over SAC alone.

### Multi-Task Supervised (CIFAR-100, 20 tasks)

| Method | Accuracy |
|---|---|
| Cross-Stitch | 53% |
| PCGrad alone | 71% |
| Routing networks + PCGrad | 77.5% |

### Ablation

Removing direction modification (magnitude-only) ≈ GradNorm. Removing magnitude (direction-only) significantly underperforms full PCGrad. Both components matter.

## In Ariel

**Where to apply:** `37_train_residual_mtrl.py` (Stage 3 MTRL-PPO). The per-task gradient conflict risk is high because:
1. Tasks have structurally different reward landscapes (hover: dense continuous; trajectory tasks: sparse gate-passing).
2. Despite per-task reward normalisation, the PPO actor gradient can still be dominated by the trajectory tasks' high-magnitude gate-spike gradients after normalisation.

**Integration point:** Wrap the PPO optimizer's gradient accumulation. After `loss.backward()`, before `optimizer.step()`, split the computed gradient by task (using `task_ids` tracked per rollout step), apply PCGrad projection, then re-sum.

**Practical threshold:** In ariel's training, monitor per-task `cos(φ_ij)` between the actor gradients at each PPO update epoch. If the hover–trajectory conflict fraction exceeds ~80% (as observed in PCGrad's empirical study), add PCGrad. If hover's gradient is consistently near-zero (the prior solves it), conflict may be absent and PCGrad adds no value.

**SB3 integration note:** SB3's PPO computes a single combined loss over all tasks in the rollout buffer. To apply PCGrad properly, the combined loss must be separated into per-task loss components before backward(). This requires a custom `train()` override in SB3's `OnPolicyAlgorithm`.

## See Also

- [[Residual_Policy_Learning]] — ariel's MTRL architecture where PCGrad would be applied
- [[single_controller_quadcopter]] / [[extreme_adapt_quadcopter]] — morphology-conditioned control papers that use per-task normalisation but not PCGrad
