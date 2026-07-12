---
type: concept_reference
tags: [concept, multitask, reinforcement-learning, optimization, gradient]
source: https://arxiv.org/abs/2001.06782
date_ingested: 2026-07-09
---

# multitask_gradient_interference

The phenomenon in multi-task learning where gradient updates computed for different tasks destructively interfere during shared-parameter optimisation, causing some tasks to stagnate, oscillate, or regress while others improve.

## Theory

Three root causes (the "tragic triad" — Yu et al. 2020):

### 1. Conflicting gradients
`cos(φ_ij) < 0` between task i and task j gradients. Updating along the combined gradient moves *away* from at least one task's optimum.

### 2. Gradient magnitude disparity
When `‖g_i‖ ≫ ‖g_j‖`, task i dominates the update regardless of relative importance. Task j may make no progress even with non-conflicting gradients.

```
Φ(g_i, g_j) = 2‖g_i‖‖g_j‖ / (‖g_i‖² + ‖g_j‖²)
# Φ ≈ 1 → balanced; Φ ≈ 0 → magnitude-dominated
```

### 3. High curvature
Even non-conflicting gradients can interfere if the loss landscape has high curvature in the shared parameter space — the linear approximation used by gradient descent breaks down.

## Remedies

| Problem | Remedy | Example |
|---|---|---|
| Conflicting directions | [[PCGrad_Gradient_Surgery]] — project conflicting gradients | Yu et al. 2020 |
| Magnitude disparity | Per-task reward / loss normalisation; GradNorm; [[PopArt_MultiTask_RL]] | ariel `_PerTaskRewardNormalizer` |
| All three | PCGrad + normalisation combined | Recommended for ariel |

## In Ariel

The MTRL residual controller (`37_train_residual_mtrl.py`) trains on 5 tasks simultaneously:

| Task | Reward structure | Gradient character |
|---|---|---|
| hover | Dense, continuous (+0.0125/step) | Smooth, low-magnitude |
| figure8 | Sparse gate spikes (+1 per gate) | High-variance, intermittent |
| slalom | Sparse gate spikes | High-variance, intermittent |
| shuttle-run | Sparse gate spikes | High-variance, intermittent |
| circle | Sparse gate spikes | High-variance, intermittent |

This is a high-risk configuration for gradient interference:
- Hover gradient is small and smooth; trajectory gradients are large and spiky.
- After per-task reward normalisation (which addresses magnitude disparity), directional conflict can still dominate — especially early in training before the residual has learned to track trajectories.

**Diagnostic:** Monitor `cos(φ_hover, φ_trajectory)` during PPO update epochs. Persistent negative cosine similarity (>50% of updates) is the actionable signal to add [[PCGrad_Gradient_Surgery]].

**Mitigation priority:**
1. Per-task reward normalisation (already implemented).
2. Per-task α scaling (already implemented — hover α=0.10 limits hover's action magnitude, which may also reduce hover's gradient magnitude relative to trajectory tasks).
3. PCGrad (not yet implemented — add if per-task reward stagnates after 20M steps).

## Practical Notes

- **Gradient conflict is episodic**, not constant. Early training (0–10M steps) tends to have more conflict as tasks have not yet been balanced. It typically reduces as training converges.
- **Per-task gradient cosine similarity is cheap to log** — compute it once per PPO epoch from the actor's gradient. Add to tensorboard as an early-warning signal.
- **Task weighting is an alternative to PCGrad**: manually up/down-weight task losses. Less principled than PCGrad but easier to implement in SB3.

## See Also

- [[PCGrad_Gradient_Surgery]] — the PCGrad algorithm for resolving directional conflict
- [[Residual_Policy_Learning]] — ariel's MTRL context
