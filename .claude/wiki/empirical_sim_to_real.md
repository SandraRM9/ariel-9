---
type: concept_reference
tags: [sim-to-real, reinforcement-learning, gaussian-process, knn, drone, perception, concept]
source: https://www.nature.com/articles/s41586-023-06419-4
date_ingested: 2026-07-09
---

# empirical_sim_to_real

Sim-to-real transfer method that replaces domain randomization with **empirical residual models** fit from a small number of real-world rollouts. Two separate models capture the two dominant sim-to-real gaps: perception noise (stochastic, temporally correlated) and dynamics mismatch (deterministic, state-dependent).

Introduced by Kaufmann et al. (Nature 2023) in [[Swift_Drone_Racing]]. The key argument: domain randomization forces the policy to be conservative across all randomized conditions, whereas empirical residuals let the policy be *optimistic* (aggressive) for the *actual* deployment environment.

## Theory

The sim-to-real gap decomposes into two independent channels:

```
gap_total = gap_perception + gap_dynamics

gap_perception: noise and drift in the state estimator (VIO)
                stochastic, temporally correlated, speed-dependent
                → model with Gaussian Process

gap_dynamics:   unmodeled aerodynamics, motor nonlinearities
                largely deterministic given (state, command)
                → model with k-NN regression
```

### Perception Residual — Gaussian Process

Nine independent 1D GPs (one per state component: position x/y/z, velocity x/y/z, attitude roll/pitch/yaw):

```
κ(z_i, z_j) = σ_f² · exp(-½ (z_i - z_j)^T L^{-2} (z_i - z_j)) + σ_n²

z     : input features (time, flight speed, manoeuvre type)
L     : diagonal length-scale matrix (learned)
σ_f   : data noise std (learned)
σ_n   : prior noise std (learned)
```

Hyperparameters optimized by marginal likelihood maximization. The GP generates **temporally consistent sample paths** during fine-tuning: each simulated rollout draws a full correlated noise trajectory rather than i.i.d. noise per step. This is critical for reproducing the drift behaviour of VIO at high speeds.

**Data requirement:** ~3 real rollouts (~50 seconds each). The small data size is possible because the GP captures temporal structure rather than per-step statistics.

### Dynamics Residual — k-NN Regression

```
a_res = KNN_k5(s_t, c_t)

s_t : platform state (position, velocity, attitude, body rates)
c_t : commanded collective thrust
a_res : residual linear acceleration (3-d)
k   : 5
```

**Dataset size:** 800–1,000 state-action-residual tuples (one track's worth of flight). The residual is deterministic, so a simple non-parametric regressor suffices. kNN is preferred over a parametric model because the residual surface is irregular (ground effect, prop wash, structural resonances) and hard to parameterize.

## In Ariel

Ariel currently uses **no empirical sim-to-real** — the hex drone pipeline is simulation-only. If real hardware is added, the Swift methodology applies directly:

1. Deploy the Stage-3 PPO policy on real hardware with motion capture.
2. Log `(state_estimated_VIO, state_groundtruth_mocap, action)` for ~3 flights.
3. Fit 9 GPs on `(state_groundtruth - state_estimated)` traces.
4. Fit kNN on `(s_t, c_t) → a_residual` where `a_residual = a_measured - a_simulated`.
5. Inject both models into `TorchDroneGateEnv` and fine-tune for ~2×10⁷ steps.

For the **hex morphology** case (ariel's setting), separate GP and kNN models per morphology class may be needed, since aerodynamic residuals differ substantially across arm-lengths and spin patterns.

## Practical Notes

- **One fine-tuning iteration is sufficient.** Kaufmann et al. show that a second iteration changes lap time by only 0.02 ± 0.02 s and distance variance by 0.09 ± 0.08 m — negligible.
- **GP temporal consistency matters more than GP accuracy.** The key is that the noise is correlated in time (mimicking VIO drift), not that the GP is calibrated. An i.i.d. noise injection gives a much weaker sim-to-real bridge.
- **kNN is sufficient for dynamics.** There is no evidence that a neural residual model outperforms kNN here; the dataset is small and the benefit of interpolation is low.
- **Domain randomization is a viable fallback** but produces policies that are 20–40% slower on real hardware because they hedge against worst-case parameter perturbations that never occur in practice.
- **Empirical residuals require a stable base policy first.** The Phase-1 PPO policy must already complete laps reliably in idealized sim before data collection. Collecting data with an unstable policy produces uninformative residual samples.

## See Also

- [[Swift_Drone_Racing]] — full system context; reward function includes `r_perc` to keep gates in view for VIO stability
- [[Residual_Policy_Learning]] — different use of "residual": additive policy correction on a prior controller, not a sim-to-real model
- [[single_controller_quadcopter]] — uses domain randomization instead; trades peak performance for morphology generalization
