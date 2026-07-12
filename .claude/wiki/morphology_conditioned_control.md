---
type: concept_reference
tags: [concept, morphology, drone, adaptation, reinforcement-learning, multitask]
source: https://arxiv.org/abs/2209.09232
date_ingested: 2026-07-09
---

# morphology_conditioned_control

The problem of training a **single policy** that can fly (or otherwise control) robots with substantially different morphologies — body geometry, mass distribution, actuator strengths — without per-morphology retraining.

## Theory

The core challenge: a fixed controller trained on one morphology will fail on another because the mapping from actions to dynamics is morphology-dependent. Three families of solutions:

### 1. Feed-forward morphology encoding
Provide the policy with an explicit morphology descriptor at inference time:

```
a_t = π(s_t, morph_features)
```

The policy learns to condition on `morph_features`. Requires knowing the morphology at deploy time. Works well if the descriptor is expressive enough and the morphology manifold is well-covered during training.

**Examples:**
- Ariel's `morph_features` (22-d hand-crafted, permutation-invariant) in `morphology_features.py`
- Any "body plan" encoding passed as a fixed obs component

### 2. Online latent adaptation (RMA-style)
Train a two-phase system: first a policy conditioned on a privileged latent `z` encoding morphology/env parameters; then an adaptation module that estimates `z` from observable state-action history:

```
z_t = φ({s_{t-k:t}, a_{t-k:t}})     # estimated from history
a_t = π(s_t, z_t)
```

Does not require knowing the morphology at deploy time — adapts online. Latency is ~1–2 window lengths.

**Examples:**
- [[single_controller_quadcopter]] (Zhang et al. 2023) — quadcopters with 4.5× mass variation
- [[extreme_adapt_quadcopter]] (Zhang et al. 2025) — extends 2023 with BC+RL, torque reward, design-informed randomization; 100× motor constant variation
- RMA (Kumar et al. 2021) — legged robots on diverse terrains

### 3. Prior + residual
Provide a morphology-specific prior controller (analytical or pre-computed) and learn a residual correction on top of it:

```
a_t = π_0(s_t; θ_morph) + α · π_θ(s_t, morph_features)
```

The prior handles "how to fly this body" from day zero; the residual learns task-tracking corrections. Fastest at deploy time if the prior is cheap to compute; requires an analytical or pre-tunable prior to exist.

**Examples:**
- Ariel's `ResidualDroneEnv` + `HoverPrior` — [[Residual_Policy_Learning]]
- Johannink et al. 2019 — impedance controller + SAC residual

## In Ariel

Ariel uses **approach 3 (prior + residual)** as the primary architecture, with **approach 1 (feed-forward morph encoding)** as the secondary conditioning signal to the residual:

```
a_t = prior(s_t; cmaes_params)        # 35c CMA-tuned per morph, 11-d
    + α_task · residual(s_t, task_oh, morph_features)   # PPO residual
```

- `cmaes_params` (11-d): per-morph analytical gains, hidden from actor, used only in prior
- `morph_features` (22-d): feed-forward, permutation-invariant, given to actor
- Library of 100 morphs in `__data__/hex_library/v1/library.npz`

**Fallback path to approach 2:** If `morph_features` prove insufficient for OOD morphs in Stage-4 evaluation, replace/augment with a TCN adaptation module (φ) estimating a compressed morph latent from recent state-action history — directly porting the Zhang et al. architecture.

## Practical Notes

- **Coverage matters more than representation:** A better morph descriptor doesn't help if the training distribution leaves corners of morphology space uncovered. Stratified sampling (as in ariel's `hex_sampler.py`) is necessary.
- **Feed-forward vs online:** Feed-forward is faster and simpler but fails on truly unseen morphologies. Online adaptation is slower (latency, history buffer) but more robust.
- **Prior quality dominates early training:** A strong analytical prior collapses the credit-assignment problem — the residual only needs to learn incremental corrections rather than "how to fly at all."
- **Latent dimensionality:** Zhang et al. show that 6-d latent outperforms direct 18-d parameter estimation. Ariel's 22-d morph_features may be over-specified — consider a learned 6–8-d compression for the actor if training converges slowly.

## See Also

- [[single_controller_quadcopter]] — Zhang et al. 2023, RMA-based approach
- [[Residual_Policy_Learning]] — Silver/Johannink 2018/2019, prior+residual approach
